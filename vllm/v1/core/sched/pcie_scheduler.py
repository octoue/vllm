# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCIe transfer scheduler for coordinating KV cache transfers with PP pipeline phases.

Coordinates H2D (Prefetch/Restore), D2H (Evict), and P2P (Pipeline Parallel) operations
to reduce PCIe bandwidth contention in single-node multi-GPU environments.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Callable

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.kv_offload.worker.worker import TransferSpec

logger = init_logger(__name__)


class PPPhase(IntEnum):
    """Pipeline Parallel execution phase."""

    RECV = 0
    """PP_Recv in progress - avoid scheduling H2D on GPU1."""
    FORWARD = 1
    """GPU forward pass."""
    SEND = 2
    """PP_Send in progress."""
    IDLE = 3
    """PCIe idle window - ideal for H2D transfers."""


class TransferPriority(IntEnum):
    """Priority for PCIe transfer operations. Lower value = higher priority."""

    RESTORE = 0
    """Highest: Restore preempted request KV cache - affects TTFT."""
    PREFETCH = 1
    """Medium: Prefetch future request KV cache - affects warming."""
    EVICT = 2
    """Lowest: Evict to CPU (D2H) - least latency critical."""


@dataclass
class TransferRequest:
    """Encapsulates a transfer request for the PCIe scheduler."""

    priority: int
    transfer_spec: Any
    label: str
    req_id: str | None = None
    submit_time: float = field(default_factory=time.monotonic)
    _sequence: int = field(default=0, repr=False)
    extra: dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: TransferRequest) -> bool:
        # heapq is min-heap; lower priority value = higher priority
        if self.priority != other.priority:
            return self.priority < other.priority
        return self._sequence < other._sequence


class PCIeTransferScheduler:
    """Coordinates KV cache transfers with PP pipeline phases.

    Maintains a priority queue for pending transfers and controls when they
    are dispatched to the underlying offloading handler based on PP phase
    and concurrent transfer limits.
    """

    def __init__(
        self,
        max_concurrent_h2d: int = 2,
        prefetch_block_threshold: int = 50,
        enable_pp_phase_aware: bool = True,
        evict_batch_size: int = 4,
        dispatch_fn: Callable[[TransferRequest], bool] | None = None,
    ):
        """Initialize the PCIe transfer scheduler.

        Args:
            max_concurrent_h2d: Maximum concurrent H2D (Prefetch/Restore) transfers.
            prefetch_block_threshold: Defer prefetch when free_blocks < this value.
            enable_pp_phase_aware: Whether to align H2D with PP idle windows.
            evict_batch_size: Batch size for Evict operations (reserved for future).
            dispatch_fn: Callback to execute a transfer; receives (spec, label).
        """
        self.max_concurrent_h2d = max_concurrent_h2d
        self.prefetch_block_threshold = prefetch_block_threshold
        self.enable_pp_phase_aware = enable_pp_phase_aware
        self.evict_batch_size = evict_batch_size
        self._dispatch_fn = dispatch_fn
        self._pending_transfers: list[tuple[int, int, TransferRequest]] = []
        self._sequence_counter = 0
        self._active_h2d_count = 0
        self._pp_phase = PPPhase.IDLE

        # Statistics for observability
        self._stats = {
            "prefetch_deferred": 0,      # Prefetch requests deferred due to block pressure
            "restore_dispatched": 0,     # Restore transfers dispatched (high priority)
            "prefetch_dispatched": 0,    # Prefetch transfers dispatched
            "evict_dispatched": 0,       # Evict transfers dispatched (low priority)
            "max_queue_depth": 0,        # Maximum pending transfer queue depth
            "h2d_throttled": 0,          # H2D transfers throttled due to concurrency limit
            "total_submitted": 0,        # Total transfers submitted
            "pp_idle_flushes": 0,        # Number of flushes triggered by PP idle phase
        }

    def should_defer_prefetch(self, free_blocks: int) -> bool:
        """Determine whether to defer a prefetch request based on GPU block pressure.

        When free_blocks is below the threshold, defer prefetch to allow normal
        requests to allocate blocks first.

        Args:
            free_blocks: Current number of free GPU blocks.

        Returns:
            True if prefetch should be deferred, False otherwise.
        """
        should_defer = free_blocks < self.prefetch_block_threshold
        if should_defer:
            self._stats["prefetch_deferred"] += 1
        return should_defer

    def submit_transfer(
        self,
        transfer_spec: "TransferSpec",
        priority: int,
        label: str,
        req_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        """Submit a transfer request to the scheduler.

        If PP-phase-aware scheduling is enabled and we are not in IDLE phase,
        the transfer may be queued. Otherwise it is dispatched immediately
        (if dispatch_fn is set).

        Args:
            transfer_spec: The transfer specification (src, dst block IDs).
            priority: TransferPriority value (RESTORE=0, PREFETCH=1, EVICT=2).
            label: Human-readable label (e.g., "Prefetch", "Restore", "Evict").
            req_id: Optional request ID for tracking.

        Returns:
            True if transfer was dispatched, False if queued (caller should not
            rely on immediate execution).
        """
        self._sequence_counter += 1
        req = TransferRequest(
            priority=priority,
            transfer_spec=transfer_spec,
            label=label,
            req_id=req_id,
            _sequence=self._sequence_counter,
            extra=extra or {},
        )
        heapq.heappush(self._pending_transfers, (priority, req._sequence, req))

        # Update statistics
        self._stats["total_submitted"] += 1
        if self._stats["total_submitted"] % 20 == 0:
            logger.debug(f"PCIe Scheduler: {self._stats['total_submitted']} submitted, "
                         f"queue={len(self._pending_transfers)}, phase={self._pp_phase.name}")
        current_depth = len(self._pending_transfers)
        if current_depth > self._stats["max_queue_depth"]:
            self._stats["max_queue_depth"] = current_depth

        if self._dispatch_fn is None:
            return True

        # Only queue; caller must call flush() to dispatch in priority order.
        return False

    def _flush_pending_transfers(self) -> bool:
        """Dispatch pending transfers up to the concurrency limit.

        Evict (D2H) is not limited by max_concurrent_h2d and can run alongside
        H2D transfers. H2D (Prefetch/Restore) respect the limit.

        Returns:
            True if at least one transfer was dispatched.
        """
        if self._dispatch_fn is None:
            return False
        dispatched = False
        max_iter = len(self._pending_transfers) + 1  # prevent infinite loop
        for _ in range(max_iter):
            if not self._pending_transfers:
                break
            _, _, req = heapq.heappop(self._pending_transfers)
            is_h2d = req.label in ("Prefetch", "Restore")
            if is_h2d and self._active_h2d_count >= self.max_concurrent_h2d:
                # At H2D limit; skip this one and try next (may be Evict).
                self._stats["h2d_throttled"] += 1
                heapq.heappush(
                    self._pending_transfers,
                    (req.priority, req._sequence, req),
                )
                continue
            if is_h2d:
                self._active_h2d_count += 1
            success = self._dispatch_fn(req)
            if success:
                dispatched = True
                # Update dispatch statistics
                if req.label == "Restore":
                    self._stats["restore_dispatched"] += 1
                elif req.label == "Prefetch":
                    self._stats["prefetch_dispatched"] += 1
                elif req.label == "Evict":
                    self._stats["evict_dispatched"] += 1
            else:
                if is_h2d:
                    self._active_h2d_count -= 1
                heapq.heappush(
                    self._pending_transfers,
                    (req.priority, req._sequence, req),
                )
                break
        return dispatched

    def _flush_evict_only(self) -> bool:
        """仅分发Evict (D2H)传输，不受PP阶段限制"""
        if self._dispatch_fn is None:
            return False

        dispatched = False
        max_iter = len(self._pending_transfers)
        for _ in range(max_iter):
            if not self._pending_transfers:
                break

            # 查找优先级队列中的Evict请求
            _, _, req = heapq.heappop(self._pending_transfers)

            if req.label != "Evict":
                # 非Evict请求，放回队列
                heapq.heappush(
                    self._pending_transfers,
                    (req.priority, req._sequence, req),
                )
                continue

            # 执行Evict传输
            success = self._dispatch_fn(req)
            if success:
                dispatched = True
                self._stats["evict_dispatched"] += 1
            else:
                # 失败则放回
                heapq.heappush(
                    self._pending_transfers,
                    (req.priority, req._sequence, req),
                )
                break

        return dispatched

    def _flush_h2d_transfers(self) -> bool:
        """分发H2D (Prefetch/Restore)传输，仅在IDLE阶段调用

        遵守max_concurrent_h2d限制
        """
        if self._dispatch_fn is None:
            return False

        dispatched = False
        max_iter = len(self._pending_transfers)

        for _ in range(max_iter):
            if not self._pending_transfers:
                break

            # 检查H2D并发限制
            if self._active_h2d_count >= self.max_concurrent_h2d:
                self._stats["h2d_throttled"] += 1
                break

            _, _, req = heapq.heappop(self._pending_transfers)

            # 只处理H2D传输
            if req.label not in ("Prefetch", "Restore"):
                heapq.heappush(
                    self._pending_transfers,
                    (req.priority, req._sequence, req),
                )
                continue

            # 执行H2D传输
            self._active_h2d_count += 1
            success = self._dispatch_fn(req)

            if success:
                dispatched = True
                if req.label == "Restore":
                    self._stats["restore_dispatched"] += 1
                elif req.label == "Prefetch":
                    self._stats["prefetch_dispatched"] += 1
            else:
                self._active_h2d_count -= 1
                heapq.heappush(
                    self._pending_transfers,
                    (req.priority, req._sequence, req),
                )
                break

        return dispatched

    def flush(self) -> bool:
        """分发待处理传输，考虑PP阶段和传输类型

        策略:
        1. D2H (Evict) 随时可分发
        2. H2D (Prefetch/Restore) 仅在IDLE阶段分发
        3. Restore优先级最高，可考虑放宽限制
        """
        if not self._pending_transfers:
            return False

        # 如果禁用PP感知，直接分发所有
        if not self.enable_pp_phase_aware:
            return self._flush_pending_transfers()

        # D2H (Evict) 随时可分发
        evict_dispatched = self._flush_evict_only()

        # H2D (Prefetch/Restore) 仅在IDLE阶段分发
        h2d_dispatched = False
        if self._pp_phase == PPPhase.IDLE:
            h2d_dispatched = self._flush_h2d_transfers()
        else:
            # 增加调试日志
            h2d_count = sum(1 for _, _, r in self._pending_transfers if r.label in ('Prefetch', 'Restore'))
            d2h_count = sum(1 for _, _, r in self._pending_transfers if r.label == 'Evict')
            if h2d_count > 0:
                logger.debug(
                    f"Flush blocked: phase={self._pp_phase.name}, "
                    f"pending={len(self._pending_transfers)} "
                    f"(H2D={h2d_count}, D2H={d2h_count})"
                )

        return evict_dispatched or h2d_dispatched

    def on_transfer_completed(self, label: str) -> None:
        """Notify that an H2D transfer has completed.

        Call from the offloading handler when a Prefetch or Restore finishes.
        """
        if label in ("Prefetch", "Restore") and self._active_h2d_count > 0:
            self._active_h2d_count -= 1

    def on_pp_phase_change(self, new_phase: PPPhase) -> None:
        """处理PP阶段变化

        - RECV: 阻止H2D分发，但允许D2H
        - SEND: 阻止H2D分发，但允许D2H
        - IDLE: 重置H2D计数，分发所有待处理传输
        """
        if new_phase != self._pp_phase:
            logger.debug(f"PP Phase: {self._pp_phase.name} → {new_phase.name}, "
                         f"pending={len(self._pending_transfers)}")

        self._pp_phase = new_phase

        if new_phase == PPPhase.RECV or new_phase == PPPhase.SEND:
            # RECV/SEND阶段: 仅分发D2H (Evict)
            self._flush_evict_only()

        elif new_phase == PPPhase.IDLE:
            # IDLE阶段: 重置H2D计数 + 分发所有传输
            self._active_h2d_count = 0
            if self._pending_transfers:  # 仅当队列非空时flush
                self._stats["pp_idle_flushes"] += 1
                self.flush()  # 分发D2H和H2D

    @property
    def has_pending_transfers(self) -> bool:
        """Return True if there are queued transfers."""
        return len(self._pending_transfers) > 0

    def get_stats(self) -> dict[str, int]:
        """Return a copy of current statistics for reporting."""
        return self._stats.copy()

    def log_stats(self) -> None:
        """Log current statistics for debugging and analysis."""
        logger.info(
            "PCIe Scheduler Stats: submitted=%d, deferred=%d, "
            "restore=%d, prefetch=%d, evict=%d, max_queue=%d, "
            "throttled=%d, pp_idle_flushes=%d",
            self._stats["total_submitted"],
            self._stats["prefetch_deferred"],
            self._stats["restore_dispatched"],
            self._stats["prefetch_dispatched"],
            self._stats["evict_dispatched"],
            self._stats["max_queue_depth"],
            self._stats["h2d_throttled"],
            self._stats["pp_idle_flushes"],
        )
