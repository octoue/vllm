# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
PCIe bandwidth profiling tracer for vLLM.

Records transfer events (KV offload Evict/Restore, Prefetch, PP) with
CUDA Event timing for Gantt chart visualization. Enabled via
VLLM_PCIE_TRACE=1.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class PCIeEvent:
    """A single PCIe transfer event.

    op_type: "Evict" | "Restore" | "Prefetch" | "PP_P2P_Send" | "PP_P2P_Recv"
             | "PP_TP_AllGather_Reconstruct" | "PP_Transfer" (legacy aggregate)
    wire_bytes: actual bytes on the link (for bandwidth); defaults to size_bytes
    logical_bytes: logical tensor size (e.g. full tensor before send-allgather slice)
    """

    op_type: str
    gpu_id: int
    direction: str  # "H2D" | "D2H" | "P2P"
    start_us: float
    end_us: float
    size_bytes: int  # kept for backward compat; equals wire_bytes when not split
    wire_bytes: int | None = None
    logical_bytes: int | None = None
    src_rank: int | None = None
    dst_rank: int | None = None
    group: str | None = None
    transport_scope: str | None = None  # "intra_node" | "inter_node"

    def to_dict(self) -> dict:
        duration_ms = (self.end_us - self.start_us) / 1000.0
        wire = self.wire_bytes if self.wire_bytes is not None else self.size_bytes
        size_mb = wire / (1024 * 1024)
        bandwidth_gbps = (
            (wire / (1024**3)) / (duration_ms / 1000.0)
            if duration_ms > 0
            else 0.0
        )
        out: dict = {
            "op_type": self.op_type,
            "gpu_id": self.gpu_id,
            "direction": self.direction,
            "start_us": self.start_us,
            "end_us": self.end_us,
            "start_ms": self.start_us / 1000.0,
            "duration_ms": duration_ms,
            "size_bytes": self.size_bytes,
            "size_mb": size_mb,
            "bandwidth_gbps": bandwidth_gbps,
        }
        if self.wire_bytes is not None:
            out["wire_bytes"] = self.wire_bytes
        if self.logical_bytes is not None:
            out["logical_bytes"] = self.logical_bytes
        if self.src_rank is not None:
            out["src_rank"] = self.src_rank
        if self.dst_rank is not None:
            out["dst_rank"] = self.dst_rank
        if self.group is not None:
            out["group"] = self.group
        if self.transport_scope is not None:
            out["transport_scope"] = self.transport_scope
        return out


class PCIeTracer:
    """
    Thread-safe singleton for recording PCIe transfer events.
    Zero overhead when VLLM_PCIE_TRACE != "1".
    """

    _instance: PCIeTracer | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._events: list[PCIeEvent] = []
        self._events_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> PCIeTracer | None:
        if os.environ.get("VLLM_PCIE_TRACE", "0") != "1":
            return None
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def record_event(
        self,
        op_type: str,
        gpu_id: int,
        direction: str,
        start_us: float,
        end_us: float,
        size_bytes: int,
        *,
        wire_bytes: int | None = None,
        logical_bytes: int | None = None,
        src_rank: int | None = None,
        dst_rank: int | None = None,
        group: str | None = None,
        transport_scope: str | None = None,
    ) -> None:
        with self._events_lock:
            self._events.append(
                PCIeEvent(
                    op_type=op_type,
                    gpu_id=gpu_id,
                    direction=direction,
                    start_us=start_us,
                    end_us=end_us,
                    size_bytes=size_bytes,
                    wire_bytes=wire_bytes,
                    logical_bytes=logical_bytes,
                    src_rank=src_rank,
                    dst_rank=dst_rank,
                    group=group,
                    transport_scope=transport_scope,
                )
            )

    def save_json(self, path: str) -> None:
        with self._events_lock:
            data = [e.to_dict() for e in self._events]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def clear(self) -> None:
        with self._events_lock:
            self._events.clear()


def get_pcie_tracer() -> PCIeTracer | None:
    """Return the PCIeTracer singleton if VLLM_PCIE_TRACE=1, else None."""
    return PCIeTracer.get_instance()
