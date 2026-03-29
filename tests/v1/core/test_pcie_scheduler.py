# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the PCIe transfer scheduler and its integration."""

import time

import pytest

from vllm.v1.core.sched.pcie_scheduler import (
    LoadLevel,
    PCIeTransferScheduler,
    PPPhase,
    TransferPriority,
    TransferRequest,
)
from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler, mock_kv

pytestmark = pytest.mark.cpu_test


# ------------------------------------------------------------------
# PCIeTransferScheduler core logic
# ------------------------------------------------------------------


def test_priority_order():
    """D2H (Evict) dispatched first to free blocks; then H2D: RESTORE > PREFETCH."""
    dispatched = []

    def capture_dispatch(req: TransferRequest) -> bool:
        dispatched.append(req.label)
        return True

    sched = PCIeTransferScheduler(
        max_concurrent_h2d=10,
        dispatch_fn=capture_dispatch,
    )
    sched.submit_transfer(None, TransferPriority.EVICT, "Evict")
    sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch")
    sched.submit_transfer(None, TransferPriority.RESTORE, "Restore")
    sched.flush()

    assert dispatched == ["Evict", "Restore", "Prefetch"]


def test_same_priority_fifo():
    """Same-priority requests are dispatched in FIFO order."""
    dispatched = []

    def capture_dispatch(req: TransferRequest) -> bool:
        dispatched.append(req.req_id or req._sequence)
        return True

    sched = PCIeTransferScheduler(
        max_concurrent_h2d=10,
        dispatch_fn=capture_dispatch,
    )
    sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch", req_id="a")
    sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch", req_id="b")
    sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch", req_id="c")
    sched.flush()

    assert [r for r in dispatched if isinstance(r, str)] == ["a", "b", "c"]


def test_concurrency_limit():
    """max_concurrent_h2d limits active H2D transfers; third stays pending."""
    dispatched = []

    def capture_dispatch(req: TransferRequest) -> bool:
        dispatched.append(req.label)
        return True

    sched = PCIeTransferScheduler(
        max_concurrent_h2d=2,
        dispatch_fn=capture_dispatch,
    )
    sched.submit_transfer(None, TransferPriority.RESTORE, "Restore")
    sched.submit_transfer(None, TransferPriority.RESTORE, "Restore")
    sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch")
    sched.flush()

    assert len(dispatched) == 2
    assert sched.has_pending_transfers is True
    assert sched._active_h2d_count == 2

    sched.on_transfer_completed("Restore")
    sched.flush()
    assert len(dispatched) == 3
    assert sched.has_pending_transfers is False


def test_evict_not_limited_by_h2d():
    """Evict is not limited by max_concurrent_h2d; Evict dispatched first."""
    dispatched = []

    def capture_dispatch(req: TransferRequest) -> bool:
        dispatched.append(req.label)
        return True

    sched = PCIeTransferScheduler(
        max_concurrent_h2d=2,
        dispatch_fn=capture_dispatch,
    )
    sched.submit_transfer(None, TransferPriority.RESTORE, "Restore")
    sched.submit_transfer(None, TransferPriority.RESTORE, "Restore")
    sched.submit_transfer(None, TransferPriority.EVICT, "Evict")
    sched.flush()

    assert "Evict" in dispatched
    assert dispatched == ["Evict", "Restore", "Restore"]


def test_should_defer_prefetch():
    """should_defer_prefetch returns True when free_blocks < threshold."""
    sched = PCIeTransferScheduler(prefetch_block_threshold=50)
    assert sched.should_defer_prefetch(49) is True
    assert sched.should_defer_prefetch(50) is False
    assert sched.should_defer_prefetch(51) is False


def test_pp_phase_idle_triggers_flush():
    """on_pp_phase_change(IDLE) triggers flush of pending transfers."""
    dispatched = []

    def capture_dispatch(req: TransferRequest) -> bool:
        dispatched.append(req.label)
        return True

    sched = PCIeTransferScheduler(
        max_concurrent_h2d=1,
        dispatch_fn=capture_dispatch,
    )
    sched.submit_transfer(None, TransferPriority.RESTORE, "Restore")
    sched.submit_transfer(None, TransferPriority.RESTORE, "Restore")
    sched.flush()
    assert len(dispatched) == 1

    sched.on_pp_phase_change(PPPhase.RECV)
    assert len(dispatched) == 1

    sched.on_pp_phase_change(PPPhase.IDLE)
    assert len(dispatched) == 2


def test_prefetch_starved_dispatched_after_queue_wait():
    """Prefetch waiting longer than max_queue_wait_ms bypasses RECV soft cap."""
    dispatched: list[str] = []

    def capture_dispatch(req: TransferRequest) -> bool:
        dispatched.append(req.req_id or "")
        return True

    sched = PCIeTransferScheduler(
        max_concurrent_h2d=2,
        enable_pp_phase_aware=True,
        max_queue_wait_ms=30,
        dispatch_fn=capture_dispatch,
    )
    sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch", req_id="first")
    sched.flush()
    assert dispatched == ["first"]
    assert sched._active_h2d_count == 1

    sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch", req_id="second")
    sched.on_pp_phase_change(PPPhase.RECV)
    sched.flush()
    assert dispatched == ["first"]

    time.sleep(0.04)
    sched.flush()
    assert dispatched == ["first", "second"]
    assert sched._stats["prefetch_starved_dispatched"] >= 1


def test_forward_phase_allows_full_h2d_concurrency():
    """FORWARD uses full max_concurrent_h2d (soft limit, not RECV cap of 1)."""
    dispatched: list[str] = []

    def capture_dispatch(req: TransferRequest) -> bool:
        dispatched.append(req.label)
        return True

    sched = PCIeTransferScheduler(
        max_concurrent_h2d=2,
        enable_pp_phase_aware=True,
        max_queue_wait_ms=0,
        dispatch_fn=capture_dispatch,
    )
    sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch", req_id="a")
    sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch", req_id="b")
    sched.on_pp_phase_change(PPPhase.FORWARD)
    sched.flush()
    assert len(dispatched) == 2
    assert sched._active_h2d_count == 2


def test_pp_phase_recv_no_flush():
    """on_pp_phase_change(RECV) does not flush."""
    dispatched = []

    def capture_dispatch(req: TransferRequest) -> bool:
        dispatched.append(req.label)
        return True

    sched = PCIeTransferScheduler(
        max_concurrent_h2d=1,
        dispatch_fn=capture_dispatch,
    )
    sched.submit_transfer(None, TransferPriority.RESTORE, "Restore")
    sched.flush()
    sched.submit_transfer(None, TransferPriority.RESTORE, "Restore")
    assert len(dispatched) == 1

    sched.on_pp_phase_change(PPPhase.RECV)
    assert len(dispatched) == 1
    assert sched.has_pending_transfers is True


def test_flush_evict_and_restore_dispatches_prefetch_in_idle():
    """flush_evict_and_restore() dispatches Prefetch when PP phase is IDLE."""
    dispatched: list[str] = []

    def capture_dispatch(req: TransferRequest) -> bool:
        dispatched.append(req.label)
        return True

    sched = PCIeTransferScheduler(
        max_concurrent_h2d=10,
        enable_pp_phase_aware=True,
        dispatch_fn=capture_dispatch,
    )
    # IDLE is the default phase
    assert sched._pp_phase == PPPhase.IDLE

    sched.submit_transfer(None, TransferPriority.EVICT, "Evict")
    sched.submit_transfer(None, TransferPriority.RESTORE, "Restore")
    sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch")
    sched.flush_evict_and_restore()

    assert "Evict" in dispatched
    assert "Restore" in dispatched
    assert "Prefetch" in dispatched
    assert len(dispatched) == 3


def test_flush_evict_and_restore_dispatches_prefetch_in_forward():
    """flush_evict_and_restore() dispatches Prefetch when PP phase is FORWARD."""
    dispatched: list[str] = []

    def capture_dispatch(req: TransferRequest) -> bool:
        dispatched.append(req.label)
        return True

    sched = PCIeTransferScheduler(
        max_concurrent_h2d=10,
        enable_pp_phase_aware=True,
        dispatch_fn=capture_dispatch,
    )
    sched.on_pp_phase_change(PPPhase.FORWARD)

    sched.submit_transfer(None, TransferPriority.EVICT, "Evict")
    sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch")
    sched.flush_evict_and_restore()

    assert "Evict" in dispatched
    assert "Prefetch" in dispatched


def test_flush_evict_and_restore_no_prefetch_in_recv():
    """flush_evict_and_restore() does NOT dispatch Prefetch when PP phase is RECV."""
    dispatched: list[str] = []

    def capture_dispatch(req: TransferRequest) -> bool:
        dispatched.append(req.label)
        return True

    sched = PCIeTransferScheduler(
        max_concurrent_h2d=10,
        enable_pp_phase_aware=True,
        dispatch_fn=capture_dispatch,
    )
    sched.on_pp_phase_change(PPPhase.RECV)

    sched.submit_transfer(None, TransferPriority.EVICT, "Evict")
    sched.submit_transfer(None, TransferPriority.RESTORE, "Restore")
    sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch")
    sched.flush_evict_and_restore()

    assert "Evict" in dispatched
    assert "Restore" in dispatched
    assert "Prefetch" not in dispatched
    assert sched.has_pending_transfers is True


def test_flush_evict_and_restore_no_prefetch_in_send():
    """flush_evict_and_restore() does NOT dispatch Prefetch when PP phase is SEND."""
    dispatched: list[str] = []

    def capture_dispatch(req: TransferRequest) -> bool:
        dispatched.append(req.label)
        return True

    sched = PCIeTransferScheduler(
        max_concurrent_h2d=10,
        enable_pp_phase_aware=True,
        dispatch_fn=capture_dispatch,
    )
    sched.on_pp_phase_change(PPPhase.SEND)

    sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch")
    sched.flush_evict_and_restore()

    assert "Prefetch" not in dispatched
    assert sched.has_pending_transfers is True


def test_dispatch_fn_false_requeues():
    """When dispatch_fn returns False, request is requeued."""
    call_count = 0

    def fail_twice_then_ok(req: TransferRequest) -> bool:
        nonlocal call_count
        call_count += 1
        return call_count >= 3

    sched = PCIeTransferScheduler(
        max_concurrent_h2d=2,
        dispatch_fn=fail_twice_then_ok,
    )
    sched.submit_transfer(None, TransferPriority.RESTORE, "Restore")
    result = sched.flush()
    assert result is False
    assert call_count == 1

    sched.flush()
    assert call_count == 2
    assert sched.has_pending_transfers is True

    sched.flush()
    assert call_count == 3
    assert sched.has_pending_transfers is False


# ------------------------------------------------------------------
# Scheduler integration: prefetch deferral
# ------------------------------------------------------------------


def test_prefetch_deferred_when_blocks_tight():
    """Prefetch is deferred when free_blocks < prefetch_block_threshold."""
    num_blocks = 30
    prefetch_threshold = 25
    # 4 requests * 2 blocks each (32 tokens) = 8 blocks used, free=22 < 25
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        use_kv_connector=mock_kv(matched_tokens=32, is_async=True),
        enable_pcie_scheduling=True,
        prefetch_block_threshold=prefetch_threshold,
        num_blocks=num_blocks,
    )
    requests = create_requests(
        num_requests=5, num_tokens=32, max_tokens=16, same_prompt=True
    )
    requests[4].prefetch_only = True

    for i in range(4):
        scheduler.add_request(requests[i])

    sched_out = scheduler.schedule()
    model_out = ModelRunnerOutput(
        req_ids=list(sched_out.num_scheduled_tokens.keys()),
        req_id_to_index={
            r: i for i, r in enumerate(sched_out.num_scheduled_tokens)
        },
        sampled_token_ids=[[1], [1], [1], [1]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(sched_out, model_out)

    free_before = scheduler.kv_cache_manager.get_num_free_blocks()
    assert free_before < prefetch_threshold, (
        f"Need free_blocks < {prefetch_threshold} for deferral, got {free_before}"
    )

    scheduler.add_request(requests[4])
    sched_out2 = scheduler.schedule()

    assert requests[4].request_id not in sched_out2.num_scheduled_tokens
    assert requests[4].request_id in [r.request_id for r in scheduler.waiting]


def test_prefetch_not_deferred_when_blocks_available():
    """Prefetch proceeds when free_blocks >= prefetch_block_threshold."""
    num_blocks = 100
    prefetch_threshold = 25
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        use_kv_connector=mock_kv(matched_tokens=32, is_async=True),
        enable_pcie_scheduling=True,
        prefetch_block_threshold=prefetch_threshold,
        num_blocks=num_blocks,
    )
    requests = create_requests(
        num_requests=1, num_tokens=32, max_tokens=16
    )
    requests[0].prefetch_only = True

    scheduler.add_request(requests[0])
    sched_out = scheduler.schedule()

    assert scheduler.kv_cache_manager.get_num_free_blocks() >= prefetch_threshold
    assert requests[0].request_id in sched_out.num_scheduled_tokens


# ------------------------------------------------------------------
# OffloadingConnectorWorker integration (mock-based)
# ------------------------------------------------------------------


def test_offloading_connector_creates_pcie_scheduler_when_enabled():
    """When enable_pcie_scheduling, OffloadingConnectorWorker creates PCIeTransferScheduler."""
    from unittest.mock import MagicMock

    from vllm.config import SchedulerConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
        OffloadingConnectorWorker,
    )

    sc = SchedulerConfig(
        max_num_seqs=8,
        max_num_batched_tokens=1024,
        max_model_len=1024,
        is_encoder_decoder=False,
        enable_pcie_scheduling=True,
    )
    spec = MagicMock()
    spec.vllm_config.scheduler_config = sc
    spec.vllm_config.kv_connector_config = None
    spec.get_manager.return_value = MagicMock()

    conn = OffloadingConnectorWorker(spec=spec)
    assert conn._pcie_scheduler is not None
    assert isinstance(conn._pcie_scheduler, PCIeTransferScheduler)


# ------------------------------------------------------------------
# Phase 1: Idle window capacity awareness
# ------------------------------------------------------------------


def test_idle_window_count_limit():
    """max_h2d_per_idle_window limits H2D dispatches within a single IDLE window."""
    dispatched: list[str] = []

    def capture_dispatch(req: TransferRequest) -> bool:
        dispatched.append(req.label)
        return True

    sched = PCIeTransferScheduler(
        max_concurrent_h2d=10,
        max_h2d_per_idle_window=2,
        idle_window_budget_ms=0,  # disable time budget
        enable_pp_phase_aware=True,
        dispatch_fn=capture_dispatch,
    )
    # Default phase is IDLE
    assert sched._pp_phase == PPPhase.IDLE
    sched._idle_window_start_time = time.monotonic()

    sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch", req_id="a")
    sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch", req_id="b")
    sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch", req_id="c")
    sched.flush()

    # Only 2 dispatched due to window count limit
    assert len(dispatched) == 2
    assert sched.has_pending_transfers is True
    assert sched._idle_window_h2d_count == 2

    # New IDLE window resets counter
    sched.on_pp_phase_change(PPPhase.RECV)
    sched.on_pp_phase_change(PPPhase.IDLE)
    assert sched._idle_window_h2d_count == 1  # third was dispatched in new window
    assert len(dispatched) == 3


def test_idle_window_budget_limit():
    """idle_window_budget_ms stops dispatching when time budget is exhausted."""
    dispatched: list[str] = []

    def capture_dispatch(req: TransferRequest) -> bool:
        dispatched.append(req.label)
        return True

    sched = PCIeTransferScheduler(
        max_concurrent_h2d=10,
        max_h2d_per_idle_window=0,  # disable count limit
        idle_window_budget_ms=5.0,  # 5ms budget
        enable_pp_phase_aware=True,
        dispatch_fn=capture_dispatch,
    )
    # Simulate IDLE window that started 10ms ago (budget exhausted)
    sched._idle_window_start_time = time.monotonic() - 0.010

    sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch", req_id="a")
    sched.flush()

    # Budget exhausted, nothing dispatched
    assert len(dispatched) == 0
    assert sched.has_pending_transfers is True
    assert sched._stats["idle_window_budget_exhausted"] >= 1


def test_idle_window_limit_not_applied_in_forward():
    """IDLE window limits only apply in IDLE phase, not FORWARD."""
    dispatched: list[str] = []

    def capture_dispatch(req: TransferRequest) -> bool:
        dispatched.append(req.label)
        return True

    sched = PCIeTransferScheduler(
        max_concurrent_h2d=10,
        max_h2d_per_idle_window=1,
        enable_pp_phase_aware=True,
        dispatch_fn=capture_dispatch,
    )
    sched.on_pp_phase_change(PPPhase.FORWARD)

    sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch", req_id="a")
    sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch", req_id="b")
    sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch", req_id="c")
    sched.flush()

    # All dispatched because we're in FORWARD, not IDLE
    assert len(dispatched) == 3


# ------------------------------------------------------------------
# Load-adaptive Phase-Aware scheduling
# ------------------------------------------------------------------


def test_load_level_low_strict_phase_aware():
    """LOW load: RECV blocks H2D (strict Phase-Aware)."""
    dispatched: list[str] = []

    def capture_dispatch(req: TransferRequest) -> bool:
        dispatched.append(req.label)
        return True

    sched = PCIeTransferScheduler(
        max_concurrent_h2d=10,
        max_h2d_per_idle_window=3,
        enable_pp_phase_aware=True,
        max_queue_wait_ms=0,
        dispatch_fn=capture_dispatch,
    )
    sched.on_pp_phase_change(PPPhase.RECV)

    # Submit 2 (≤ capacity of 3) → LOW load
    sched.submit_transfer(None, TransferPriority.RESTORE, "Restore", req_id="r1")
    sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch", req_id="p1")

    assert sched._compute_load_level() == LoadLevel.LOW
    sched.flush()

    # RECV at LOW: soft cap = 1, so at most 1 H2D dispatched
    assert len(dispatched) <= 1


def test_load_level_high_ignores_phase():
    """HIGH load: RECV allows full H2D concurrency (G2-mode)."""
    dispatched: list[str] = []

    def capture_dispatch(req: TransferRequest) -> bool:
        dispatched.append(req.label)
        return True

    sched = PCIeTransferScheduler(
        max_concurrent_h2d=10,
        max_h2d_per_idle_window=3,
        enable_pp_phase_aware=True,
        max_queue_wait_ms=0,
        dispatch_fn=capture_dispatch,
    )
    sched.on_pp_phase_change(PPPhase.RECV)

    # Submit 10 (> 3*3=9) → HIGH load
    for i in range(10):
        sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch",
                              req_id=f"p{i}")

    assert sched._compute_load_level() == LoadLevel.HIGH
    sched.flush()

    # HIGH: phase limits off, all dispatched up to max_concurrent_h2d
    assert len(dispatched) == 10


def test_load_level_medium_applies_idle_budget():
    """MEDIUM load: IDLE window count limit is strictly enforced."""
    dispatched: list[str] = []

    def capture_dispatch(req: TransferRequest) -> bool:
        dispatched.append(req.label)
        return True

    sched = PCIeTransferScheduler(
        max_concurrent_h2d=10,
        max_h2d_per_idle_window=2,
        idle_window_budget_ms=0,  # disable time budget
        enable_pp_phase_aware=True,
        dispatch_fn=capture_dispatch,
    )
    sched._idle_window_start_time = time.monotonic()

    # Submit 5 (2 < 5 ≤ 6=2*3) → MEDIUM load
    for i in range(5):
        sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch",
                              req_id=f"p{i}")

    assert sched._compute_load_level() == LoadLevel.MEDIUM
    sched.flush()

    # MEDIUM enforces count limit = 2
    assert len(dispatched) == 2
    assert sched.has_pending_transfers is True


def test_load_level_high_disables_idle_budget():
    """HIGH load: IDLE window limits completely disabled."""
    dispatched: list[str] = []

    def capture_dispatch(req: TransferRequest) -> bool:
        dispatched.append(req.label)
        return True

    sched = PCIeTransferScheduler(
        max_concurrent_h2d=10,
        max_h2d_per_idle_window=2,
        idle_window_budget_ms=0,
        enable_pp_phase_aware=True,
        dispatch_fn=capture_dispatch,
    )
    sched._idle_window_start_time = time.monotonic()

    # Submit 10 (> 2*3=6) → HIGH load
    for i in range(10):
        sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch",
                              req_id=f"p{i}")

    assert sched._compute_load_level() == LoadLevel.HIGH
    sched.flush()

    # HIGH: no window limits, all 10 dispatched
    assert len(dispatched) == 10


def test_load_level_transitions_tracked():
    """Stats correctly count load level occurrences on each flush."""
    def noop(req: TransferRequest) -> bool:
        return True

    sched = PCIeTransferScheduler(
        max_concurrent_h2d=10,
        max_h2d_per_idle_window=2,
        enable_pp_phase_aware=True,
        dispatch_fn=noop,
    )

    # LOW flush
    sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch")
    sched.flush()
    assert sched._stats["load_level_low"] == 1

    # MEDIUM flush (submit 4 > 2, ≤ 6)
    for _ in range(4):
        sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch")
    sched.flush()
    assert sched._stats["load_level_medium"] == 1

    # HIGH flush (submit 10 > 6)
    for _ in range(10):
        sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch")
    sched.flush()
    assert sched._stats["load_level_high"] == 1


def test_high_load_recv_triggers_flush():
    """on_pp_phase_change(RECV) flushes H2D when load is HIGH."""
    dispatched: list[str] = []

    def capture_dispatch(req: TransferRequest) -> bool:
        dispatched.append(req.label)
        return True

    sched = PCIeTransferScheduler(
        max_concurrent_h2d=10,
        max_h2d_per_idle_window=2,
        enable_pp_phase_aware=True,
        max_queue_wait_ms=0,
        dispatch_fn=capture_dispatch,
    )

    # Submit 10 items to reach HIGH load
    for i in range(10):
        sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch",
                              req_id=f"p{i}")

    # Transition to RECV — should flush because HIGH
    sched.on_pp_phase_change(PPPhase.RECV)

    assert len(dispatched) == 10
    assert sched._stats["load_level_high"] >= 1


# ------------------------------------------------------------------
# OffloadingConnectorWorker integration (mock-based)
# ------------------------------------------------------------------


def test_offloading_connector_no_pcie_scheduler_when_disabled():
    """When enable_pcie_scheduling=False, OffloadingConnectorWorker has no PCIe scheduler."""
    from unittest.mock import MagicMock

    from vllm.config import SchedulerConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
        OffloadingConnectorWorker,
    )

    sc = SchedulerConfig(
        max_num_seqs=8,
        max_num_batched_tokens=1024,
        max_model_len=1024,
        is_encoder_decoder=False,
        enable_pcie_scheduling=False,
    )
    spec = MagicMock()
    spec.vllm_config.scheduler_config = sc
    spec.vllm_config.kv_connector_config = None
    spec.get_manager.return_value = MagicMock()

    conn = OffloadingConnectorWorker(spec=spec)
    assert conn._pcie_scheduler is None
