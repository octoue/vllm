# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the prefetch_only mechanism in the scheduler.

These tests verify that:
1. A request with prefetch_only=True is immediately finished after
   the first step (prefill), with generated tokens discarded.
2. The EngineCoreRequest.prefetch_only field serializes correctly.
3. Request.from_engine_core_request propagates the flag.
"""

import msgspec
import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from .utils import EOS_TOKEN_ID, create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


# ------------------------------------------------------------------
# Test: prefetch_only request is finished immediately
# ------------------------------------------------------------------
def test_prefetch_only_finishes_immediately():
    """A prefetch_only request should be marked FINISHED_STOPPED after
    the first update_from_output, with its generated tokens discarded."""

    scheduler = create_scheduler()

    requests = create_requests(num_requests=2, max_tokens=16)
    # Mark the first request as prefetch_only
    requests[0].prefetch_only = True

    for req in requests:
        req.num_computed_tokens = req.num_tokens
        scheduler.requests[req.request_id] = req
        scheduler.running.append(req)
        req.status = RequestStatus.RUNNING

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={
            requests[0].request_id: 1,
            requests[1].request_id: 1,
        },
        total_num_scheduled_tokens=2,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={req.request_id: i for i, req in enumerate(requests)},
        sampled_token_ids=[[42], [99]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    engine_core_outputs = scheduler.update_from_output(
        scheduler_output, model_output
    )

    # The prefetch request should be finished and removed from running.
    assert requests[0].status == RequestStatus.FINISHED_STOPPED
    assert requests[0].request_id in scheduler.finished_req_ids
    # Generated tokens should be discarded for the prefetch request.
    assert list(requests[0].output_token_ids) == []

    # The normal request should still be running with its token appended.
    assert len(scheduler.running) == 1
    assert scheduler.running[0].request_id == requests[1].request_id
    assert list(requests[1].output_token_ids) == [99]

    # Verify the engine core output includes a finish_reason for the
    # prefetch request (so the API layer can construct its response).
    all_outputs = []
    for client_outputs in engine_core_outputs.values():
        all_outputs.extend(client_outputs.outputs)
    prefetch_output = [
        o for o in all_outputs if o.request_id == requests[0].request_id
    ]
    assert len(prefetch_output) == 1
    assert prefetch_output[0].finish_reason is not None


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
