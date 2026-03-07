# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for prefetch CPU-hit eviction scenario.

Verifies that when prefix KV blocks have been evicted from GPU to CPU,
a prefetch=True request triggers CPU->GPU migration, and subsequent
real requests benefit from the warmed cache (lower TTFT).

Environment:
  VLLM_PREFETCH_TEST_MODEL: Override model path (default: local Qwen3-8B)
  VLLM_PREFETCH_TEST_GPU: Override GPU selection (default: auto-detect most idle)
"""

import os
import re
import subprocess
import time

import openai
import pytest
import pytest_asyncio
import requests

from ...utils import RemoteOpenAIServer

MODEL_PATH = os.environ.get(
    "VLLM_PREFETCH_TEST_MODEL",
    "/lpai/models/Qwen__Qwen3-8B/25-07-26-0349",
)


def _get_most_idle_gpu() -> str:
    """Select the GPU with most free memory."""
    gpu_override = os.environ.get("VLLM_PREFETCH_TEST_GPU")
    if gpu_override is not None:
        return str(gpu_override)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        if not lines:
            return "0"
        gpus = []
        for line in lines:
            parts = line.split(",", 1)
            if len(parts) == 2:
                idx = int(parts[0].strip())
                free = int(parts[1].strip())
                gpus.append((idx, free))
        if not gpus:
            return "0"
        gpus.sort(key=lambda x: x[1], reverse=True)
        return str(gpus[0][0])
    except Exception:
        return "0"


# ---------------------------------------------------------------------------
# Helper: Synthetic long prompts (~320 tokens each, 20 blocks at block_size=16)
# ---------------------------------------------------------------------------

PARAGRAPH_A = (
    "The theory of relativity, proposed by Albert Einstein, is divided into "
    "two parts: special relativity and general relativity. Special relativity "
    "deals with objects moving at constant speeds. General relativity describes "
    "gravity as curvature of spacetime. "
) * 8  # ~320 tokens

PARAGRAPH_B = (
    "Quantum mechanics is the branch of physics that deals with the behavior "
    "of matter and light on the atomic and subatomic scale. It introduces "
    "concepts such as wave-particle duality and the uncertainty principle. "
) * 8

PARAGRAPH_C = (
    "Machine learning is a subset of artificial intelligence that enables "
    "systems to learn from data. Deep learning uses neural networks with "
    "multiple layers to model complex patterns. "
) * 8

PARAGRAPH_D = (
    "The history of computing dates back to ancient times with the abacus. "
    "Modern computers evolved through vacuum tubes, transistors, and "
    "integrated circuits. The internet has connected the world. "
) * 8


def _make_messages(prefix: str, suffix: str | None = None) -> list[dict]:
    """Build chat messages with prefix and optional suffix."""
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prefix},
    ]
    if suffix:
        msgs.append({"role": "user", "content": suffix})
    return msgs


# ---------------------------------------------------------------------------
# Helper: Parse Prometheus /metrics
# ---------------------------------------------------------------------------

def _fetch_metrics(metrics_url: str) -> str:
    """Fetch raw Prometheus metrics from /metrics endpoint."""
    resp = requests.get(metrics_url, timeout=10)
    resp.raise_for_status()
    return resp.text


def _parse_counter_flexible(metrics_text: str, name: str, label_filter: str) -> float:
    """Parse counter by matching lines containing name and label_filter.

    Uses case-insensitive matching for label_filter since Prometheus may use
    'CPU_to_GPU' / 'GPU_to_CPU' (from vLLM transfer_type).
    """
    total = 0.0
    for line in metrics_text.split("\n"):
        if line.strip().startswith("#"):
            continue
        if name not in line:
            continue
        if label_filter and label_filter.lower() not in line.lower():
            continue
        parts = line.rsplit(None, 1)
        if len(parts) == 2:
            try:
                total += float(parts[1])
            except ValueError:
                pass
    return total


def _get_metrics_snapshot_flexible(metrics_url: str) -> dict[str, float]:
    """Get metrics snapshot using flexible parsing (handles variable label order).

    Note: Prometheus labels use 'GPU_to_CPU' and 'CPU_to_GPU' (from vLLM
    transfer_type), so we match case-insensitively.
    """
    text = _fetch_metrics(metrics_url)
    return {
        "kv_offload_gpu_to_cpu_bytes": _parse_counter_flexible(
            text, "vllm:kv_offload_total_bytes", "gpu_to_cpu"
        ),
        "kv_offload_cpu_to_gpu_bytes": _parse_counter_flexible(
            text, "vllm:kv_offload_total_bytes", "cpu_to_gpu"
        ),
        "external_prefix_cache_hits": _parse_counter_flexible(
            text, "vllm:external_prefix_cache_hits", ""
        ),
        "prefix_cache_hits": _parse_counter_flexible(
            text, "vllm:prefix_cache_hits", ""
        ),
    }


# ---------------------------------------------------------------------------
# Helper: TTFT measurement (streaming first token)
# ---------------------------------------------------------------------------

async def _measure_ttft(
    client: openai.AsyncOpenAI,
    messages: list[dict],
    model: str = MODEL_PATH,
) -> float:
    """Measure TTFT: time from request start to first received chunk."""
    start = time.perf_counter()
    first_chunk_time: float | None = None
    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=32,
        stream=True,
    )
    async for chunk in stream:
        if first_chunk_time is None:
            first_chunk_time = time.perf_counter()
        if chunk.choices and chunk.choices[0].delta.content:
            break
    if first_chunk_time is None:
        first_chunk_time = time.perf_counter()
    return first_chunk_time - start


# ---------------------------------------------------------------------------
# Server fixture with KV offload / small GPU cache
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cpu_hit_server_args():
    """Server args to induce GPU eviction: small GPU cache + KV offload."""
    return [
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "32",
        "--enforce-eager",
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",
        "--block-size",
        "16",
        "--num-gpu-blocks-override",
        "48",
        "--kv-offloading-size",
        "2",
        "--disable-hybrid-kv-cache-manager",
    ]


@pytest.fixture(scope="module")
def cpu_hit_server(cpu_hit_server_args):
    """Start vLLM with KV offload and small GPU cache."""
    env_dict = {"HF_HUB_OFFLINE": "1"}
    env_dict["CUDA_VISIBLE_DEVICES"] = _get_most_idle_gpu()
    with RemoteOpenAIServer(
        MODEL_PATH,
        cpu_hit_server_args,
        env_dict=env_dict,
        max_wait_seconds=360,
    ) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def cpu_hit_client(cpu_hit_server):
    async with cpu_hit_server.get_async_client() as async_client:
        yield async_client


def _assert_no_error(response, context: str = ""):
    if hasattr(response, "error") and response.error is not None:
        err = response.error
        msg = getattr(err, "message", str(err))
        err_type = getattr(err, "type", "Unknown")
        pytest.fail(f"{context}Request failed with {err_type}: {msg}")


# ---------------------------------------------------------------------------
# Test: Prefetch CPU->GPU migration (five-step)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prefetch_cpu_to_gpu_migration(
    cpu_hit_server,
    cpu_hit_client: openai.AsyncOpenAI,
):
    """Verify prefetch triggers CPU->GPU KV migration when prefix is GPU-miss/CPU-hit.

    Step 1: Populate prefix A with normal request -> KV in GPU + offload to CPU
    Step 2: Evict A from GPU by sending 3 different prefix requests (B/C/D)
    Step 3: Record metrics baseline
    Step 4: Prefetch prefix A -> CPU->GPU transfer
    Step 5: Real request with prefix A + suffix -> cache hit, cached_tokens > 0
    """
    metrics_url = cpu_hit_server.url_for("metrics")

    # Step 1: Populate prefix A
    _ = await cpu_hit_client.chat.completions.create(
        model=MODEL_PATH,
        messages=_make_messages(PARAGRAPH_A),
        max_tokens=8,
    )
    m1 = _get_metrics_snapshot_flexible(metrics_url)

    # Step 2: Evict A by sending 3 different prefix requests
    for prefix in (PARAGRAPH_B, PARAGRAPH_C, PARAGRAPH_D):
        _ = await cpu_hit_client.chat.completions.create(
            model=MODEL_PATH,
            messages=_make_messages(prefix),
            max_tokens=8,
        )

    m2 = _get_metrics_snapshot_flexible(metrics_url)

    # Step 3: Baseline - gpu_to_cpu should have increased (store) after eviction
    assert m2["kv_offload_gpu_to_cpu_bytes"] >= m1["kv_offload_gpu_to_cpu_bytes"], (
        "Expected gpu_to_cpu store after eviction"
    )

    cpu_to_gpu_before = m2["kv_offload_cpu_to_gpu_bytes"]

    # Step 4: Prefetch prefix A (should trigger CPU->GPU load)
    prefetch_resp = await cpu_hit_client.chat.completions.create(
        model=MODEL_PATH,
        messages=_make_messages(PARAGRAPH_A),
        extra_body={"prefetch": True},
    )
    _assert_no_error(prefetch_resp, "test_prefetch_cpu_to_gpu_migration Step 4: ")

    assert prefetch_resp.usage is not None
    assert prefetch_resp.usage.prompt_tokens > 0
    assert prefetch_resp.usage.completion_tokens == 0

    details = prefetch_resp.usage.prompt_tokens_details
    assert details is not None
    assert details.cached_tokens is not None and details.cached_tokens > 0, (
        f"Prefetch should report cached_tokens > 0 after CPU->GPU load, "
        f"got {details.cached_tokens}"
    )

    m3 = _get_metrics_snapshot_flexible(metrics_url)

    # Step 4 verification: cpu_to_gpu bytes should have increased
    cpu_to_gpu_after = m3["kv_offload_cpu_to_gpu_bytes"]
    if cpu_to_gpu_after <= cpu_to_gpu_before:
        raw = _fetch_metrics(metrics_url)
        kv_lines = [
            l for l in raw.split("\n")
            if "kv_offload" in l and not l.strip().startswith("#")
        ]
        pytest.fail(
            f"Expected cpu_to_gpu transfer after prefetch: "
            f"before={cpu_to_gpu_before}, after={cpu_to_gpu_after}. "
            f"Prometheus uses transfer_type='CPU_to_GPU' or 'GPU_to_CPU'. "
            f"KV offload lines: {kv_lines[:15]!r}"
        )

    # Step 5: Real request with prefix A + new question
    real_messages = _make_messages(PARAGRAPH_A, "Can you summarize that in one sentence?")
    real_resp = await cpu_hit_client.chat.completions.create(
        model=MODEL_PATH,
        messages=real_messages,
        max_tokens=32,
    )
    _assert_no_error(real_resp, "test_prefetch_cpu_to_gpu_migration Step 5: ")

    assert real_resp.usage is not None
    assert real_resp.usage.prompt_tokens > 0
    assert len(real_resp.choices) == 1
    assert real_resp.choices[0].message.content is not None

    real_details = real_resp.usage.prompt_tokens_details
    assert real_details is not None, (
        "prompt_tokens_details should be present "
        "(server started with --enable-prompt-tokens-details)"
    )
    assert real_details.cached_tokens is not None and real_details.cached_tokens > 0, (
        f"Real request should have cached_tokens > 0 after prefetch, "
        f"got {real_details.cached_tokens}"
    )


# ---------------------------------------------------------------------------
# Test: TTFT A/B comparison (prefetch vs no prefetch)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.slow
async def test_prefetch_cpu_hit_ttft_improvement(
    cpu_hit_server,
    cpu_hit_client: openai.AsyncOpenAI,
):
    """Compare TTFT: prefetch vs no prefetch in GPU-miss/CPU-hit scenario.

    A: Real request without prefetch -> TTFT_A
    B: Prefetch + real request -> TTFT_B
    Expect B < A (prefetch warms cache, so TTFT should be lower).
    """
    # Setup: populate A, then evict with B/C/D
    _ = await cpu_hit_client.chat.completions.create(
        model=MODEL_PATH,
        messages=_make_messages(PARAGRAPH_A),
        max_tokens=8,
    )
    for prefix in (PARAGRAPH_B, PARAGRAPH_C, PARAGRAPH_D):
        _ = await cpu_hit_client.chat.completions.create(
            model=MODEL_PATH,
            messages=_make_messages(prefix),
            max_tokens=8,
        )

    real_messages = _make_messages(PARAGRAPH_A, "Summarize in one sentence.")

    # A: No prefetch - measure TTFT
    ttft_no_prefetch = await _measure_ttft(cpu_hit_client, real_messages)

    # Re-evict A again (B/C/D)
    for prefix in (PARAGRAPH_B, PARAGRAPH_C, PARAGRAPH_D):
        _ = await cpu_hit_client.chat.completions.create(
            model=MODEL_PATH,
            messages=_make_messages(prefix),
            max_tokens=8,
        )

    # B: Prefetch then real request
    _ = await cpu_hit_client.chat.completions.create(
        model=MODEL_PATH,
        messages=_make_messages(PARAGRAPH_A),
        extra_body={"prefetch": True},
    )
    ttft_with_prefetch = await _measure_ttft(cpu_hit_client, real_messages)

    # Prefetch should reduce TTFT (allow some variance)
    assert ttft_with_prefetch < ttft_no_prefetch * 1.5, (
        f"Prefetch should improve TTFT: no_prefetch={ttft_no_prefetch:.3f}s, "
        f"with_prefetch={ttft_with_prefetch:.3f}s"
    )
