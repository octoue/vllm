# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the prefetch_only mechanism in the scheduler.

Prefetch requests never enter model execution. They are finished or discarded
in the scheduler based on cache hits (GPU prefix cache or CPU offload).
These tests verify that:
1. Prefetch with no cache hit is discarded and returns valid output.
2. Prefetch with GPU cache hit finishes immediately.
3. The EngineCoreRequest.prefetch_only field serializes correctly.
4. Request.from_engine_core_request propagates the flag.
"""

import msgspec
import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.engine import EngineCoreRequest, FinishReason
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from .utils import EOS_TOKEN_ID, create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


# ------------------------------------------------------------------
# Test: prefetch with no cache hit is discarded
# ------------------------------------------------------------------
def test_prefetch_no_hit_discards_request():
    """A prefetch_only request with no GPU or CPU cache hit should be
    discarded in the scheduler and return a valid output (no model execution)."""

    scheduler = create_scheduler(enable_prefix_caching=True)
    requests = create_requests(num_requests=1, num_tokens=32, max_tokens=16)
    requests[0].prefetch_only = True

    scheduler.add_request(requests[0])
    assert len(scheduler.waiting) == 1

    scheduler_output = scheduler.schedule()
    # Prefetch was discarded, so nothing scheduled for model execution.
    assert requests[0].request_id not in scheduler_output.num_scheduled_tokens

    model_output = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    engine_core_outputs = scheduler.update_from_output(
        scheduler_output, model_output
    )

    all_outputs = []
    for client_outputs in engine_core_outputs.values():
        all_outputs.extend(client_outputs.outputs)
    prefetch_output = [
        o for o in all_outputs if o.request_id == requests[0].request_id
    ]
    assert len(prefetch_output) == 1
    assert prefetch_output[0].finish_reason == FinishReason.STOP
    assert prefetch_output[0].new_token_ids == []
    assert requests[0].request_id in scheduler.finished_req_ids


# ------------------------------------------------------------------
# Test: prefetch with GPU prefix cache hit finishes immediately
# ------------------------------------------------------------------
def test_prefetch_gpu_hit_finishes_immediately():
    """A prefetch_only request with GPU prefix cache hit should finish
    immediately without model execution."""

    scheduler = create_scheduler(enable_prefix_caching=True)
    requests = create_requests(
        num_requests=2, num_tokens=32, max_tokens=16, same_prompt=True
    )
    requests[0].prefetch_only = True

    # First run a normal request to populate prefix cache with same prompt.
    scheduler.add_request(requests[1])
    sched_out = scheduler.schedule()
    assert len(sched_out.num_scheduled_tokens) > 0
    # Simulate model completion for the normal request.
    model_out = ModelRunnerOutput(
        req_ids=list(sched_out.num_scheduled_tokens.keys()),
        req_id_to_index={r: i for i, r in enumerate(sched_out.num_scheduled_tokens)},
        sampled_token_ids=[[EOS_TOKEN_ID]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(sched_out, model_out)

    # Now add prefetch with same prompt (same block hashes).
    scheduler.add_request(requests[0])
    sched_out2 = scheduler.schedule()

    model_out2 = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    engine_core_outputs = scheduler.update_from_output(sched_out2, model_out2)

    all_outputs = []
    for client_outputs in engine_core_outputs.values():
        all_outputs.extend(client_outputs.outputs)
    prefetch_output = [
        o for o in all_outputs if o.request_id == requests[0].request_id
    ]
    assert len(prefetch_output) == 1
    assert prefetch_output[0].finish_reason == FinishReason.STOP
    assert prefetch_output[0].new_token_ids == []
    assert prefetch_output[0].num_cached_tokens > 0


# ------------------------------------------------------------------
# Test: prefetch_only flag on Request
# ------------------------------------------------------------------
def test_request_prefetch_only_default():
    """Request.prefetch_only should default to False."""
    req = create_requests(num_requests=1)[0]
    assert req.prefetch_only is False


def test_request_prefetch_only_set():
    """Request.prefetch_only can be explicitly set."""
    req = create_requests(num_requests=1)[0]
    req.prefetch_only = True
    assert req.prefetch_only is True


# ------------------------------------------------------------------
# Test: EngineCoreRequest serialization round-trip
# ------------------------------------------------------------------
def test_engine_core_request_prefetch_only_serialization():
    """EngineCoreRequest.prefetch_only should survive msgspec
    encode/decode."""

    original = EngineCoreRequest(
        request_id="test-prefetch-001",
        prompt_token_ids=[1, 2, 3, 4],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        eos_token_id=EOS_TOKEN_ID,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        prefetch_only=True,
    )

    encoded = msgspec.msgpack.encode(original)
    decoded = msgspec.msgpack.decode(encoded, type=EngineCoreRequest)

    assert decoded.prefetch_only is True
    assert decoded.request_id == original.request_id
    assert decoded.prompt_token_ids == original.prompt_token_ids


def test_engine_core_request_prefetch_only_default():
    """When prefetch_only is not set, it should default to False."""

    original = EngineCoreRequest(
        request_id="test-normal-001",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=16),
        pooling_params=None,
        eos_token_id=EOS_TOKEN_ID,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )

    assert original.prefetch_only is False

    encoded = msgspec.msgpack.encode(original)
    decoded = msgspec.msgpack.decode(encoded, type=EngineCoreRequest)
    assert decoded.prefetch_only is False


# ------------------------------------------------------------------
# Test: from_engine_core_request propagates prefetch_only
# ------------------------------------------------------------------
def test_from_engine_core_request_propagates_prefetch():
    """Request.from_engine_core_request should copy the prefetch_only
    flag from the EngineCoreRequest."""

    ecr = EngineCoreRequest(
        request_id="test-prop-001",
        prompt_token_ids=[10, 20, 30],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        eos_token_id=EOS_TOKEN_ID,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        prefetch_only=True,
    )

    req = Request.from_engine_core_request(ecr, block_hasher=None)
    assert req.prefetch_only is True


def test_from_engine_core_request_default_prefetch():
    """When prefetch_only is False, Request should also be False."""

    ecr = EngineCoreRequest(
        request_id="test-prop-002",
        prompt_token_ids=[10, 20, 30],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=16),
        pooling_params=None,
        eos_token_id=EOS_TOKEN_ID,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )

    req = Request.from_engine_core_request(ecr, block_hasher=None)
    assert req.prefetch_only is False
