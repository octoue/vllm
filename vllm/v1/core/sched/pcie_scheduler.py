# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCIe transfer scheduler for coordinating KV cache transfers with PP/EPLB phases.

Coordinates H2D (Prefetch/Restore), D2H (Evict), and P2P (Pipeline Parallel) operations
to reduce PCIe bandwidth contention in single-node multi-GPU environments.

EPLB-Phase-Aware scheduling:
  - Sync rearrangement: pause all H2D during rearrangement, flush on completion
  - Async migration: reduce H2D concurrency while async worker transfers weights
"""

from __future__ import annotations

import heapq
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

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


class LoadLevel(IntEnum):
    """Load level for adaptive Phase-Aware scheduling."""

    LOW = 0
    """Queue fits in one IDLE window → strict Phase-Aware + window limits."""
    MEDIUM = 1
    """Moderate backlog → relax window limits (2x count budget)."""
    HIGH = 2
    """Severe backlog → disable phase constraints (G2-mode)."""


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
    dispatch_phase: str = ""
    dispatch_time: float = 0.0

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
        max_queue_wait_ms: int = 30,
        max_h2d_per_idle_window: int = 3,
        idle_window_budget_ms: float = 7.0,
        no_priority_queue: bool = False,
        no_evict_first: bool = False,
        enable_eplb_phase_aware: bool = False,
        eplb_async_h2d_limit: int = 1,
        dispatch_fn: Callable[[TransferRequest], bool] | None = None,
    ):
        """Initialize the PCIe transfer scheduler.

        Args:
            max_concurrent_h2d: Maximum concurrent H2D (Prefetch/Restore) transfers.
            prefetch_block_threshold: Defer prefetch when free_blocks < this value.
            enable_pp_phase_aware: Whether to align H2D with PP idle windows.
            evict_batch_size: Batch size for Evict operations (reserved for future).
            max_queue_wait_ms: Prefetch queue wait before bypassing phase limits (0=off).
            max_h2d_per_idle_window: Max H2D dispatches per IDLE window (0=unlimited).
                Also used as denominator for load-level computation.
            idle_window_budget_ms: Time budget per IDLE window in ms (0=unlimited).
            enable_eplb_phase_aware: Pause H2D during EPLB rearrangement.
            eplb_async_h2d_limit: Max concurrent H2D during async EPLB migration.
            dispatch_fn: Callback to execute a transfer; receives (spec, label).
        """
        self.max_concurrent_h2d = max_concurrent_h2d
        self.prefetch_block_threshold = prefetch_block_threshold
        self.enable_pp_phase_aware = enable_pp_phase_aware
        self.evict_batch_size = evict_batch_size
        self.max_queue_wait_ms = max_queue_wait_ms
        self.max_h2d_per_idle_window = max_h2d_per_idle_window
        self.idle_window_budget_ms = idle_window_budget_ms
        self.no_priority_queue = no_priority_queue
        self.no_evict_first = no_evict_first
        self.enable_eplb_phase_aware = enable_eplb_phase_aware
        self.eplb_async_h2d_limit = eplb_async_h2d_limit
        self._dispatch_fn = dispatch_fn
        self._pending_h2d: list[tuple[int, int, TransferRequest]] = []  # Restore/Prefetch, min-heap
        self._pending_d2h: list[TransferRequest] = []  # Evict, FIFO
        self._sequence_counter = 0
        self._active_h2d_count = 0
        self._pp_phase = PPPhase.IDLE

        # EPLB phase state
        self._eplb_rearranging = False  # sync rearrangement in progress
        self._eplb_async_migrating = False  # async weight migration in progress

        # Idle window capacity tracking
        self._idle_window_h2d_count = 0  # H2D dispatched in current IDLE window
        self._idle_window_start_time = 0.0  # When current IDLE window started
        self._idle_window_utilizations: list[float] = []  # Rolling utilization ratios
        self._idle_window_max_history = 20

        # Per-phase H2D latency tracking (dispatch → completion)
        self._phase_latency: dict[str, list[float]] = defaultdict(list)

        # Statistics for observability
        self._stats = {
            "prefetch_deferred": 0,
            "restore_dispatched": 0,
            "prefetch_dispatched": 0,
            "evict_dispatched": 0,
            "max_queue_depth": 0,
            "h2d_throttled": 0,
            "total_submitted": 0,
            "pp_idle_flushes": 0,
            "prefetch_starved_dispatched": 0,
            "idle_window_budget_exhausted": 0,
            "idle_window_count_exhausted": 0,
            "load_level_low": 0,
            "load_level_medium": 0,
            "load_level_high": 0,
            "eplb_rearrange_pauses": 0,
            "eplb_h2d_deferred_during_rearrange": 0,
            "eplb_async_cc_reductions": 0,
            "eplb_post_rearrange_flushes": 0,
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

    def _compute_load_level(self) -> LoadLevel:
        """Compute current load level based on pending H2D queue depth.

        Uses max_h2d_per_idle_window as capacity estimate for one window.
        - LOW: pending fits in one window
        - MEDIUM: needs a few windows
        - HIGH: severely backlogged, disable phase constraints
        """
        pending = len(self._pending_h2d)
        capacity = (
            self.max_h2d_per_idle_window
            if self.max_h2d_per_idle_window > 0
            else self.max_concurrent_h2d
        )
        if pending <= capacity:
            return LoadLevel.LOW
        elif pending <= capacity * 3:
            return LoadLevel.MEDIUM
        return LoadLevel.HIGH

    def _effective_max_h2d_for_phase(self) -> int:
        """Load-adaptive PP-phase limit.

        - LOW/MEDIUM: full concurrency in IDLE/FORWARD; at most 1 in RECV/SEND
        - HIGH: full concurrency in all phases (G2-mode, no phase restriction)
        """
        if not self.enable_pp_phase_aware:
            return self.max_concurrent_h2d
        load = self._compute_load_level()
        if load == LoadLevel.HIGH:
            return self.max_concurrent_h2d
        if self._pp_phase in (PPPhase.IDLE, PPPhase.FORWARD):
            return self.max_concurrent_h2d
        return min(1, self.max_concurrent_h2d)

    def _idle_window_has_budget(
        self, load_override: LoadLevel | None = None
    ) -> bool:
        """IDLE window capacity check.

        Enforces count and time budget limits when in IDLE phase,
        unless load is HIGH (phase constraints fully disabled).

        Args:
            load_override: If provided, use this load level instead of
                recomputing. Prevents mid-flush level drift as items drain.
        """
        if self._pp_phase != PPPhase.IDLE:
            return True

        load = load_override if load_override is not None else self._compute_load_level()
        if load == LoadLevel.HIGH:
            return True

        # MEDIUM: apply window limits to prevent overload
        if self.max_h2d_per_idle_window > 0:
            if self._idle_window_h2d_count >= self.max_h2d_per_idle_window:
                self._stats["idle_window_count_exhausted"] += 1
                return False

        if self.idle_window_budget_ms > 0 and self._idle_window_start_time > 0:
            elapsed_ms = (time.monotonic() - self._idle_window_start_time) * 1000.0
            if elapsed_ms >= self.idle_window_budget_ms:
                self._stats["idle_window_budget_exhausted"] += 1
                return False

        return True

    def _record_idle_window_utilization(self) -> None:
        """Record utilization of the ending IDLE window for adaptive tuning."""
        if self._idle_window_start_time <= 0 or self.max_h2d_per_idle_window <= 0:
            return
        utilization = self._idle_window_h2d_count / self.max_h2d_per_idle_window
        self._idle_window_utilizations.append(utilization)
        if len(self._idle_window_utilizations) > self._idle_window_max_history:
            self._idle_window_utilizations.pop(0)

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
            if self.no_priority_queue:
                self._pending_h2d.append((priority, req._sequence, req))
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
        if self.no_evict_first:
            h2d_ok = self._flush_h2d_transfers()
            d2h_ok = self._flush_d2h()
        else:
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
            if self.no_priority_queue:
                _, _, req = self._pending_h2d.pop(0)
            else:
                _, _, req = heapq.heappop(self._pending_h2d)
            if req.label != "Restore":
                to_push_back.append((req.priority, req._sequence, req))
                continue
            if self._active_h2d_count >= self.max_concurrent_h2d:
                to_push_back.append((req.priority, req._sequence, req))
                break
            req.dispatch_phase = self._pp_phase.name
            req.dispatch_time = time.monotonic()
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
            if self.no_priority_queue:
                self._pending_h2d.append(item)
            else:
                heapq.heappush(self._pending_h2d, item)
        return dispatched

    def flush_evict_and_restore(self) -> bool:
        """Dispatch Evict + Restore immediately. Prefetch if in IDLE/FORWARD."""
        if self.no_evict_first:
            restore_ok = self._flush_restore_only()
            prefetch_ok = False
            if self._pp_phase in (PPPhase.IDLE, PPPhase.FORWARD):
                prefetch_ok = self._flush_h2d_transfers()
            d2h_ok = self._flush_d2h()
        else:
            d2h_ok = self._flush_d2h()
            restore_ok = self._flush_restore_only()
            prefetch_ok = False
            if self._pp_phase in (PPPhase.IDLE, PPPhase.FORWARD):
                prefetch_ok = self._flush_h2d_transfers()
        return d2h_ok or restore_ok or prefetch_ok

    def _flush_h2d_transfers(self, effective_max: int | None = None) -> bool:
        """Dispatch H2D (Prefetch/Restore) transfers.

        Respects effective_max concurrent active H2D (defaults to max_concurrent_h2d).
        Restore (priority 0) before Prefetch (1).
        Phase 1: Also respects IDLE window capacity limits.
        EPLB-aware: blocks H2D during sync rearrangement; reduces CC during async.
        """
        if self._dispatch_fn is None or not self._pending_h2d:
            return False

        # EPLB sync rearrangement: completely pause H2D
        if self._eplb_rearranging:
            self._stats["eplb_h2d_deferred_during_rearrange"] += len(self._pending_h2d)
            return False

        limit = (
            effective_max
            if effective_max is not None
            else self.max_concurrent_h2d
        )

        # EPLB async migration: cap concurrency
        if self._eplb_async_migrating:
            limit = min(limit, self.eplb_async_h2d_limit)

        # Snapshot load level once to prevent mid-flush drift as items drain
        snapped_load = self._compute_load_level()

        dispatched = False
        while self._pending_h2d:
            if self._active_h2d_count >= limit:
                self._stats["h2d_throttled"] += 1
                break

            # Phase 1: Check IDLE window budget before dispatching
            if not self._idle_window_has_budget(load_override=snapped_load):
                break

            if self.no_priority_queue:
                _, _, req = self._pending_h2d.pop(0)
            else:
                _, _, req = heapq.heappop(self._pending_h2d)
            req.dispatch_phase = self._pp_phase.name
            req.dispatch_time = time.monotonic()
            self._active_h2d_count += 1
            success = self._dispatch_fn(req)

            if success:
                dispatched = True
                if req.label == "Restore":
                    self._stats["restore_dispatched"] += 1
                elif req.label == "Prefetch":
                    self._stats["prefetch_dispatched"] += 1
                # Phase 1: Track IDLE window H2D count
                if self._pp_phase == PPPhase.IDLE:
                    self._idle_window_h2d_count += 1
            else:
                self._active_h2d_count -= 1
                heapq.heappush(self._pending_h2d, (req.priority, req._sequence, req))
                break

        return dispatched

    def _flush_starved_prefetches(self) -> bool:
        """Dispatch Prefetch requests that exceeded max_queue_wait_ms in the queue.

        Uses full max_concurrent_h2d for the dispatch limit (bypasses phase soft cap).
        """
        if (
            self._dispatch_fn is None
            or not self._pending_h2d
            or self.max_queue_wait_ms <= 0
        ):
            return False

        threshold_sec = self.max_queue_wait_ms / 1000.0
        now = time.monotonic()

        items: list[tuple[int, int, TransferRequest]] = []
        while self._pending_h2d:
            if self.no_priority_queue:
                items.append(self._pending_h2d.pop(0))
            else:
                items.append(heapq.heappop(self._pending_h2d))

        starved: list[tuple[int, int, TransferRequest]] = []
        rest: list[tuple[int, int, TransferRequest]] = []
        for t in items:
            req = t[2]
            if (
                req.label == "Prefetch"
                and (now - req.submit_time) >= threshold_sec
            ):
                starved.append(t)
            else:
                rest.append(t)

        starved.sort(key=lambda x: x[2].submit_time)
        dispatched = False
        for t in starved:
            _pri, _seq, req = t
            if self._active_h2d_count >= self.max_concurrent_h2d:
                rest.append(t)
                continue
            req.dispatch_phase = self._pp_phase.name
            req.dispatch_time = time.monotonic()
            self._active_h2d_count += 1
            success = self._dispatch_fn(req)
            if success:
                dispatched = True
                self._stats["prefetch_dispatched"] += 1
                self._stats["prefetch_starved_dispatched"] += 1
            else:
                self._active_h2d_count -= 1
                rest.append(t)
                break

        for t in rest:
            if self.no_priority_queue:
                self._pending_h2d.append(t)
            else:
                heapq.heappush(self._pending_h2d, t)
        return dispatched

    def flush(self) -> bool:
        """Dispatch pending transfers with load-adaptive Phase-Aware scheduling.

        Strategy:
        1. D2H (Evict) - dispatch anytime, Evict-first to free blocks
        2. H2D - load-adaptive phase limit:
           - LOW/MEDIUM: IDLE/FORWARD full concurrency, RECV/SEND at most 1
             + IDLE window count/time budget
           - HIGH: no phase restriction, no window limits (G2-mode)
        3. Starved Prefetch - bypass phase cap after max_queue_wait_ms
        """
        if not self._pending_h2d and not self._pending_d2h:
            return False

        # If PP-aware disabled, dispatch all
        if not self.enable_pp_phase_aware:
            return self._flush_pending_transfers()

        # Track load level for observability
        load = self._compute_load_level()
        if load == LoadLevel.LOW:
            self._stats["load_level_low"] += 1
        elif load == LoadLevel.MEDIUM:
            self._stats["load_level_medium"] += 1
        else:
            self._stats["load_level_high"] += 1

        effective = self._effective_max_h2d_for_phase()
        if self.no_evict_first:
            # Ablation: H2D before D2H
            h2d_dispatched = self._flush_h2d_transfers(effective_max=effective)
            h2d_dispatched |= self._flush_starved_prefetches()
            evict_dispatched = self._flush_d2h()
        else:
            # Default: D2H (Evict) first - frees blocks before H2D consumes them
            evict_dispatched = self._flush_d2h()
            h2d_dispatched = self._flush_h2d_transfers(effective_max=effective)
            h2d_dispatched |= self._flush_starved_prefetches()

        if not h2d_dispatched and self._pending_h2d:
            logger.debug(
                f"Flush: phase={self._pp_phase.name}, load={load.name}, "
                f"effective_h2d_cap={effective}, "
                f"idle_window_h2d={self._idle_window_h2d_count}, "
                f"H2D={len(self._pending_h2d)}, D2H={len(self._pending_d2h)}"
            )

        return evict_dispatched or h2d_dispatched

    def on_transfer_completed(
        self,
        label: str,
        dispatch_phase: str = "",
        dispatch_time: float = 0.0,
    ) -> None:
        """Notify that an H2D transfer has completed.

        Call from the offloading handler when a Prefetch or Restore finishes.
        Records per-phase latency when dispatch_phase/dispatch_time are provided.
        """
        if label in ("Prefetch", "Restore") and self._active_h2d_count > 0:
            self._active_h2d_count -= 1
        if dispatch_phase and dispatch_time > 0:
            latency_ms = (time.monotonic() - dispatch_time) * 1000.0
            self._phase_latency[dispatch_phase].append(latency_ms)

    def on_pp_phase_change(self, new_phase: PPPhase) -> None:
        """Handle PP phase changes with load-adaptive behavior.

        - RECV/SEND at LOW/MEDIUM: Only dispatch D2H (Evict); H2D held back
        - RECV/SEND at HIGH: Full flush (phase limits disabled in G2-mode)
        - FORWARD: Evict + H2D with soft limits
        - IDLE: Reset H2D count + window budget, flush all
        """
        if new_phase != self._pp_phase:
            logger.debug(f"PP Phase: {self._pp_phase.name} → {new_phase.name}, "
                         f"H2D={len(self._pending_h2d)}, D2H={len(self._pending_d2h)}")

        # Record utilization of ending IDLE window
        if self._pp_phase == PPPhase.IDLE and new_phase != PPPhase.IDLE:
            self._record_idle_window_utilization()

        self._pp_phase = new_phase

        if new_phase == PPPhase.RECV or new_phase == PPPhase.SEND:
            self._flush_d2h()
            # HIGH load: phase constraints are off, flush H2D too
            if (self._pending_h2d
                    and self._compute_load_level() == LoadLevel.HIGH):
                self.flush()

        elif new_phase == PPPhase.FORWARD:
            if self._pending_h2d or self._pending_d2h:
                self.flush()

        elif new_phase == PPPhase.IDLE:
            self._active_h2d_count = 0
            self._idle_window_h2d_count = 0
            self._idle_window_start_time = time.monotonic()
            if self._pending_h2d or self._pending_d2h:
                self._stats["pp_idle_flushes"] += 1
                self.flush()

    # ------------------------------------------------------------------
    # EPLB Phase-Aware hooks
    # ------------------------------------------------------------------

    def notify_eplb_rearrange_start(self) -> None:
        """Called when EPLB sync rearrangement begins.

        Pauses all H2D dispatches to avoid contending with P2P expert weight
        transfers on the PCIe link. D2H (Evict) is still allowed since it
        frees GPU memory that rearrangement may need.
        """
        if not self.enable_eplb_phase_aware:
            return
        self._eplb_rearranging = True
        self._stats["eplb_rearrange_pauses"] += 1
        logger.debug("EPLB rearrange START — H2D paused")

    def notify_eplb_rearrange_end(self) -> None:
        """Called when EPLB sync rearrangement completes.

        Resumes H2D dispatches and immediately flushes any transfers that
        accumulated during the pause.
        """
        if not self.enable_eplb_phase_aware:
            return
        self._eplb_rearranging = False
        logger.debug("EPLB rearrange END — flushing %d pending H2D",
                     len(self._pending_h2d))
        if self._pending_h2d or self._pending_d2h:
            self._stats["eplb_post_rearrange_flushes"] += 1
            self.flush()

    def notify_eplb_async_migration_start(self) -> None:
        """Called when EPLB async worker begins transferring expert weights.

        Reduces H2D concurrency to eplb_async_h2d_limit (default 1) so that
        KV cache transfers don't fully contend with the background P2P stream.
        """
        if not self.enable_eplb_phase_aware:
            return
        self._eplb_async_migrating = True
        self._stats["eplb_async_cc_reductions"] += 1
        logger.debug("EPLB async migration START — H2D CC reduced to %d",
                     self.eplb_async_h2d_limit)

    def notify_eplb_async_migration_end(self) -> None:
        """Called when EPLB async worker finishes all layer transfers.

        Restores full H2D concurrency and flushes pending transfers.
        """
        if not self.enable_eplb_phase_aware:
            return
        self._eplb_async_migrating = False
        logger.debug("EPLB async migration END — H2D CC restored to %d",
                     self.max_concurrent_h2d)
        if self._pending_h2d or self._pending_d2h:
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
            "restore=%d, prefetch=%d (starved=%d), evict=%d, max_queue=%d, "
            "throttled=%d, pp_idle_flushes=%d, "
            "idle_budget_exhausted=%d, idle_count_exhausted=%d, "
            "load_levels=[L=%d M=%d H=%d], "
            "eplb=[pauses=%d deferred=%d async_cc=%d flushes=%d]",
            self._stats["total_submitted"],
            self._stats["prefetch_deferred"],
            self._stats["restore_dispatched"],
            self._stats["prefetch_dispatched"],
            self._stats["prefetch_starved_dispatched"],
            self._stats["evict_dispatched"],
            self._stats["max_queue_depth"],
            self._stats["h2d_throttled"],
            self._stats["pp_idle_flushes"],
            self._stats["idle_window_budget_exhausted"],
            self._stats["idle_window_count_exhausted"],
            self._stats["load_level_low"],
            self._stats["load_level_medium"],
            self._stats["load_level_high"],
            self._stats["eplb_rearrange_pauses"],
            self._stats["eplb_h2d_deferred_during_rearrange"],
            self._stats["eplb_async_cc_reductions"],
            self._stats["eplb_post_rearrange_flushes"],
        )
        # Log IDLE window utilization summary
        if self._idle_window_utilizations:
            arr = np.array(self._idle_window_utilizations)
            logger.info(
                "IDLE Window Utilization: n=%d, mean=%.2f, "
                "P50=%.2f, P95=%.2f, max=%.2f",
                len(arr), float(np.mean(arr)),
                float(np.percentile(arr, 50)),
                float(np.percentile(arr, 95)),
                float(np.max(arr)),
            )
        # Per-phase H2D latency summary
        for phase, latencies in sorted(self._phase_latency.items()):
            if not latencies:
                continue
            arr = np.array(latencies)
            logger.info(
                "H2D Phase Latency [%s]: n=%d, mean=%.2fms, "
                "P50=%.2fms, P95=%.2fms, P99=%.2fms",
                phase, len(arr), float(np.mean(arr)),
                float(np.percentile(arr, 50)),
                float(np.percentile(arr, 95)),
                float(np.percentile(arr, 99)),
            )
