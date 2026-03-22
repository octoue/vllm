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
        self._pending_h2d: list[tuple[int, int, TransferRequest]] = []  # Restore/Prefetch, min-heap
        self._pending_d2h: list[TransferRequest] = []  # Evict, FIFO
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
        if label == "Evict":
            self._pending_d2h.append(req)
        else:
            heapq.heappush(self._pending_h2d, (priority, req._sequence, req))

        # Update statistics
        self._stats["total_submitted"] += 1
        current_depth = len(self._pending_h2d) + len(self._pending_d2h)
        if current_depth > self._stats["max_queue_depth"]:
            self._stats["max_queue_depth"] = current_depth
        if self._stats["total_submitted"] % 20 == 0:
            logger.debug(f"PCIe Scheduler: {self._stats['total_submitted']} submitted, "
                         f"H2D={len(self._pending_h2d)}, D2H={len(self._pending_d2h)}, "
                         f"phase={self._pp_phase.name}")

        if self._dispatch_fn is None:
            return True

        # Only queue; caller must call flush() to dispatch in priority order.
        return False

    def _flush_pending_transfers(self) -> bool:
        """Dispatch all pending (non-PP-aware mode). D2H first, then H2D."""
        d2h_ok = self._flush_d2h()
        h2d_ok = self._flush_h2d_transfers()
        return d2h_ok or h2d_ok

    def _flush_d2h(self) -> bool:
        """Dispatch Evict (D2H) transfers. Not limited by PP phase."""
        if self._dispatch_fn is None or not self._pending_d2h:
            return False

        dispatched = False
        while self._pending_d2h:
            req = self._pending_d2h.pop(0)
            success = self._dispatch_fn(req)
            if success:
                dispatched = True
                self._stats["evict_dispatched"] += 1
            else:
                self._pending_d2h.insert(0, req)
                break
        return dispatched

    def _flush_restore_only(self) -> bool:
        """Dispatch only Restore (not Prefetch). For start_kv_transfers() step start."""
        if self._dispatch_fn is None or not self._pending_h2d:
            return False

        dispatched = False
        to_push_back: list[tuple[int, int, TransferRequest]] = []
        while self._pending_h2d:
            _, _, req = heapq.heappop(self._pending_h2d)
            if req.label != "Restore":
                to_push_back.append((req.priority, req._sequence, req))
                continue
            if self._active_h2d_count >= self.max_concurrent_h2d:
                to_push_back.append((req.priority, req._sequence, req))
                break
            self._active_h2d_count += 1
            success = self._dispatch_fn(req)
            if success:
                dispatched = True
                self._stats["restore_dispatched"] += 1
            else:
                self._active_h2d_count -= 1
                to_push_back.append((req.priority, req._sequence, req))
                break
        for item in to_push_back:
            heapq.heappush(self._pending_h2d, item)
        return dispatched

    def flush_evict_and_restore(self) -> bool:
        """Dispatch Evict + Restore immediately. Prefetch stays queued for PP idle."""
        d2h_ok = self._flush_d2h()
        restore_ok = self._flush_restore_only()
        return d2h_ok or restore_ok

    def _flush_h2d_transfers(self) -> bool:
        """Dispatch H2D (Prefetch/Restore) transfers. Called only in IDLE phase.

        Respects max_concurrent_h2d limit. Restore (priority 0) before Prefetch (1).
        """
        if self._dispatch_fn is None or not self._pending_h2d:
            return False

        dispatched = False
        while self._pending_h2d:
            if self._active_h2d_count >= self.max_concurrent_h2d:
                self._stats["h2d_throttled"] += 1
                break

            _, _, req = heapq.heappop(self._pending_h2d)
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
                heapq.heappush(self._pending_h2d, (req.priority, req._sequence, req))
                break

        return dispatched

    def flush(self) -> bool:
        """Dispatch pending transfers, considering PP phase and transfer type.

        Strategy:
        1. D2H (Evict) - dispatch anytime, Evict-first to free blocks early
        2. H2D (Prefetch/Restore) - only in IDLE phase
        """
        if not self._pending_h2d and not self._pending_d2h:
            return False

        # If PP-aware disabled, dispatch all
        if not self.enable_pp_phase_aware:
            return self._flush_pending_transfers()

        # D2H (Evict) first - frees blocks before H2D consumes them
        evict_dispatched = self._flush_d2h()

        # H2D only in IDLE phase
        h2d_dispatched = False
        if self._pp_phase == PPPhase.IDLE:
            h2d_dispatched = self._flush_h2d_transfers()
        elif self._pending_h2d:
            logger.debug(
                f"Flush blocked: phase={self._pp_phase.name}, "
                f"H2D={len(self._pending_h2d)}, D2H={len(self._pending_d2h)}"
            )

        return evict_dispatched or h2d_dispatched

    def on_transfer_completed(self, label: str) -> None:
        """Notify that an H2D transfer has completed.

        Call from the offloading handler when a Prefetch or Restore finishes.
        """
        if label in ("Prefetch", "Restore") and self._active_h2d_count > 0:
            self._active_h2d_count -= 1

    def on_pp_phase_change(self, new_phase: PPPhase) -> None:
        """Handle PP phase changes.

        - RECV/SEND: Only dispatch D2H (Evict)
        - IDLE: Reset H2D count, flush all (D2H + H2D)
        """
        if new_phase != self._pp_phase:
            logger.debug(f"PP Phase: {self._pp_phase.name} → {new_phase.name}, "
                         f"H2D={len(self._pending_h2d)}, D2H={len(self._pending_d2h)}")

        self._pp_phase = new_phase

        if new_phase == PPPhase.RECV or new_phase == PPPhase.SEND:
            self._flush_d2h()

        elif new_phase == PPPhase.IDLE:
            self._active_h2d_count = 0
            if self._pending_h2d or self._pending_d2h:
                self._stats["pp_idle_flushes"] += 1
                self.flush()

    @property
    def has_pending_transfers(self) -> bool:
        """Return True if there are queued transfers."""
        return len(self._pending_h2d) > 0 or len(self._pending_d2h) > 0

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
