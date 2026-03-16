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
    """A single PCIe transfer event."""

    op_type: str  # "Evict" | "Restore" | "Prefetch" | "PP_Transfer"
    gpu_id: int
    direction: str  # "H2D" | "D2H" | "P2P"
    start_us: float
    end_us: float
    size_bytes: int

    def to_dict(self) -> dict:
        duration_ms = (self.end_us - self.start_us) / 1000.0
        size_mb = self.size_bytes / (1024 * 1024)
        bandwidth_gbps = (
            (self.size_bytes / (1024**3)) / (duration_ms / 1000.0)
            if duration_ms > 0
            else 0.0
        )
        return {
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
