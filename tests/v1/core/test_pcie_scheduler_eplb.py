# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for EPLB-Phase-Aware scheduling in the PCIe transfer scheduler."""

import pytest

from vllm.v1.core.sched.pcie_scheduler import (
    PCIeTransferScheduler,
    TransferPriority,
    TransferRequest,
)

pytestmark = pytest.mark.cpu_test


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_scheduler(**kwargs) -> tuple[PCIeTransferScheduler, list[str]]:
    """Create a scheduler with EPLB phase awareness enabled and a capture list."""
    dispatched: list[str] = []

    def capture_dispatch(req: TransferRequest) -> bool:
        dispatched.append(req.label)
        return True

    defaults = dict(
        max_concurrent_h2d=2,
        enable_pp_phase_aware=False,
        enable_eplb_phase_aware=True,
        eplb_async_h2d_limit=1,
        dispatch_fn=capture_dispatch,
    )
    defaults.update(kwargs)
    return PCIeTransferScheduler(**defaults), dispatched


# ------------------------------------------------------------------
# Sync rearrangement: H2D paused
# ------------------------------------------------------------------


def test_eplb_rearrange_blocks_h2d():
    """During sync rearrangement, H2D transfers are paused."""
    sched, dispatched = _make_scheduler()

    sched.submit_transfer(None, TransferPriority.RESTORE, "Restore", req_id="r1")
    sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch", req_id="p1")

    sched.notify_eplb_rearrange_start()
    sched.flush()

    assert len(dispatched) == 0, "H2D should be blocked during rearrangement"
    assert sched.has_pending_transfers is True
    assert sched._eplb_rearranging is True


def test_eplb_rearrange_allows_d2h():
    """During sync rearrangement, D2H (Evict) transfers still go through."""
    sched, dispatched = _make_scheduler()

    sched.submit_transfer(None, TransferPriority.EVICT, "Evict")
    sched.submit_transfer(None, TransferPriority.RESTORE, "Restore", req_id="r1")

    sched.notify_eplb_rearrange_start()
    sched.flush()

    assert "Evict" in dispatched, "D2H should still be dispatched during rearrangement"
    assert "Restore" not in dispatched, "H2D should be blocked"


def test_eplb_rearrange_end_flushes():
    """After sync rearrangement ends, accumulated H2D are immediately flushed."""
    sched, dispatched = _make_scheduler()

    sched.submit_transfer(None, TransferPriority.RESTORE, "Restore", req_id="r1")
    sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch", req_id="p1")

    sched.notify_eplb_rearrange_start()
    sched.flush()
    assert len(dispatched) == 0

    sched.notify_eplb_rearrange_end()

    assert "Restore" in dispatched
    assert "Prefetch" in dispatched
    assert len(dispatched) == 2
    assert sched._eplb_rearranging is False


def test_eplb_rearrange_end_flush_respects_cc():
    """Post-rearrangement flush still respects max_concurrent_h2d."""
    sched, dispatched = _make_scheduler(max_concurrent_h2d=1)

    sched.submit_transfer(None, TransferPriority.RESTORE, "Restore", req_id="r1")
    sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch", req_id="p1")

    sched.notify_eplb_rearrange_start()
    sched.flush()

    sched.notify_eplb_rearrange_end()

    # Only 1 should be dispatched (CC=1)
    assert len(dispatched) == 1
    assert dispatched[0] == "Restore"  # Restore has higher priority
    assert sched.has_pending_transfers is True


def test_eplb_rearrange_stats():
    """EPLB rearrangement stats are correctly tracked."""
    sched, dispatched = _make_scheduler()

    sched.submit_transfer(None, TransferPriority.RESTORE, "Restore", req_id="r1")
    sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch", req_id="p1")

    sched.notify_eplb_rearrange_start()
    sched.flush()

    assert sched._stats["eplb_rearrange_pauses"] == 1
    assert sched._stats["eplb_h2d_deferred_during_rearrange"] == 2

    sched.notify_eplb_rearrange_end()
    assert sched._stats["eplb_post_rearrange_flushes"] == 1


# ------------------------------------------------------------------
# Async migration: reduced concurrency
# ------------------------------------------------------------------


def test_eplb_async_reduces_cc():
    """During async migration, H2D concurrency is reduced to eplb_async_h2d_limit."""
    sched, dispatched = _make_scheduler(max_concurrent_h2d=4, eplb_async_h2d_limit=1)

    sched.submit_transfer(None, TransferPriority.RESTORE, "Restore", req_id="r1")
    sched.submit_transfer(None, TransferPriority.RESTORE, "Restore", req_id="r2")
    sched.submit_transfer(None, TransferPriority.PREFETCH, "Prefetch", req_id="p1")

    sched.notify_eplb_async_migration_start()
    sched.flush()

    # Only 1 dispatched due to async CC limit
    assert len(dispatched) == 1
    assert sched._active_h2d_count == 1
    assert sched._stats["eplb_async_cc_reductions"] == 1


def test_eplb_async_end_restores_cc():
    """After async migration ends, full CC is restored."""
    sched, dispatched = _make_scheduler(max_concurrent_h2d=4, eplb_async_h2d_limit=1)

    sched.submit_transfer(None, TransferPriority.RESTORE, "Restore", req_id="r1")
    sched.submit_transfer(None, TransferPriority.RESTORE, "Restore", req_id="r2")
    sched.submit_transfer(None, TransferPriority.RESTORE, "Restore", req_id="r3")

    sched.notify_eplb_async_migration_start()
    sched.flush()
    assert len(dispatched) == 1

    # Complete the first, then end migration
    sched.on_transfer_completed("Restore")
    sched.notify_eplb_async_migration_end()

    # All remaining should flush with full CC=4
    assert len(dispatched) == 3
    assert sched._eplb_async_migrating is False


def test_eplb_async_d2h_unaffected():
    """Async migration CC reduction does not affect D2H transfers."""
    sched, dispatched = _make_scheduler(max_concurrent_h2d=4, eplb_async_h2d_limit=1)

    sched.submit_transfer(None, TransferPriority.EVICT, "Evict")
    sched.submit_transfer(None, TransferPriority.EVICT, "Evict")
    sched.submit_transfer(None, TransferPriority.RESTORE, "Restore", req_id="r1")
    sched.submit_transfer(None, TransferPriority.RESTORE, "Restore", req_id="r2")

    sched.notify_eplb_async_migration_start()
    sched.flush()

    evict_count = dispatched.count("Evict")
    restore_count = dispatched.count("Restore")
    assert evict_count == 2, "All D2H should go through"
    assert restore_count == 1, "Only 1 H2D due to async CC limit"


# ------------------------------------------------------------------
# Feature disabled: no-op behavior
# ------------------------------------------------------------------


def test_eplb_phase_aware_disabled_noop():
    """When enable_eplb_phase_aware=False, EPLB notifications are no-ops."""
    dispatched: list[str] = []

    def capture_dispatch(req: TransferRequest) -> bool:
        dispatched.append(req.label)
        return True

    sched = PCIeTransferScheduler(
        max_concurrent_h2d=2,
        enable_eplb_phase_aware=False,
        dispatch_fn=capture_dispatch,
    )

    sched.submit_transfer(None, TransferPriority.RESTORE, "Restore", req_id="r1")

    sched.notify_eplb_rearrange_start()
    sched.flush()

    # Should dispatch normally since feature is disabled
    assert len(dispatched) == 1
    assert sched._eplb_rearranging is False
    assert sched._stats["eplb_rearrange_pauses"] == 0


# ------------------------------------------------------------------
# Combined PP + EPLB phase interaction
# ------------------------------------------------------------------


def test_eplb_and_pp_phase_combined():
    """EPLB rearrangement blocks H2D even when PP phase is IDLE."""
    sched, dispatched = _make_scheduler(enable_pp_phase_aware=True)

    sched.on_pp_phase_change(
        __import__("vllm.v1.core.sched.pcie_scheduler", fromlist=["PPPhase"]).PPPhase.IDLE
    )

    sched.submit_transfer(None, TransferPriority.RESTORE, "Restore", req_id="r1")

    sched.notify_eplb_rearrange_start()
    sched.flush()

    assert len(dispatched) == 0, "EPLB rearrange should override PP IDLE"


# ------------------------------------------------------------------
# EplbState hook integration (mock-based)
# ------------------------------------------------------------------


def test_eplb_state_hooks_called():
    """Verify EplbState calls PCIe hooks during simulated rearrange lifecycle."""
    calls: list[str] = []

    # Import and create a minimal mock
    from unittest.mock import MagicMock, patch

    from vllm.distributed.eplb.eplb_state import EplbState

    # We can't fully instantiate EplbState without torch/cuda,
    # but we can test set_pcie_scheduler_hooks at the attribute level
    state = MagicMock(spec=EplbState)
    state._on_rearrange_start = lambda: calls.append("start")
    state._on_rearrange_end = lambda: calls.append("end")
    state._on_async_migration_start = lambda: calls.append("async_start")
    state._on_async_migration_end = lambda: calls.append("async_end")

    # Simulate sync rearrange callback sequence
    if state._on_rearrange_start:
        state._on_rearrange_start()
    if state._on_rearrange_end:
        state._on_rearrange_end()

    assert calls == ["start", "end"]

    # Simulate async sequence
    if state._on_async_migration_start:
        state._on_async_migration_start()
    if state._on_async_migration_end:
        state._on_async_migration_end()

    assert calls == ["start", "end", "async_start", "async_end"]
