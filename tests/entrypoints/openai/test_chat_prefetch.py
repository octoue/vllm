# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the KV cache prefetch / preheat mechanism.

The prefetch feature lets clients pre-compute the KV cache for a
conversation history while the user is still typing.  When the real
request arrives with the same prefix, it benefits from prefix-cache
hits and has a lower TTFT.
"""

import openai
import pytest
import pytest_asyncio

from ...utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

HISTORY_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": (
            "Please explain the theory of relativity in detail, "
            "covering both special and general relativity, "
            "including the key equations and their implications."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "The theory of relativity, proposed by Albert Einstein, "
            "is divided into two parts: special relativity (1905) and "
            "general relativity (1915). Special relativity deals with "
            "objects moving at constant speeds, particularly near the "
            "speed of light. Its key postulates are that the laws of "
            "physics are the same in all inertial frames and that the "
            "speed of light is constant. The famous equation E=mc^2 "
            "shows the equivalence of mass and energy."
        ),
    },
]


@pytest.fixture(scope="module")
def default_server_args():
    return [
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "128",
        "--enforce-eager",
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",
    ]


@pytest.fixture(scope="module")
def server(default_server_args):
    with RemoteOpenAIServer(MODEL_NAME, default_server_args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


# ------------------------------------------------------------------
# Test 1: Basic prefetch request returns a valid response
# ------------------------------------------------------------------
@pytest.mark.asyncio
async def test_prefetch_returns_valid_response(
    client: openai.AsyncOpenAI,
):
    """A prefetch request should succeed and return a well-formed
    ChatCompletionResponse with completion_tokens == 0."""

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=HISTORY_MESSAGES,
        extra_body={"prefetch": True},
    )

    assert response.id is not None
    assert response.model == MODEL_NAME
    assert len(response.choices) == 1

    choice = response.choices[0]
    assert choice.finish_reason == "stop"
    assert choice.message.role == "assistant"

    assert response.usage is not None
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens == 0


# ------------------------------------------------------------------
# Test 2: Prefix cache hit after prefetch
# ------------------------------------------------------------------
@pytest.mark.asyncio
async def test_prefetch_populates_prefix_cache(
    client: openai.AsyncOpenAI,
):
    """After a prefetch, a follow-up request with the same history
    should report cached_tokens in prompt_tokens_details."""

    # Step 1: prefetch the conversation history
    prefetch_resp = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=HISTORY_MESSAGES,
        extra_body={"prefetch": True},
    )
    assert prefetch_resp.usage is not None
    prefetch_prompt_tokens = prefetch_resp.usage.prompt_tokens
    assert prefetch_prompt_tokens > 0

    # Step 2: send a real request with the same history + new user message
    real_messages = HISTORY_MESSAGES + [
        {"role": "user", "content": "Can you summarize that in one sentence?"},
    ]
    real_resp = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=real_messages,
        max_tokens=32,
    )

    assert real_resp.usage is not None
    assert real_resp.usage.prompt_tokens > 0
    assert len(real_resp.choices) == 1
    assert real_resp.choices[0].message.content is not None

    # The real request should have a prefix cache hit for the history
    # portion.  cached_tokens should be > 0 (the exact count depends
    # on block alignment, so we just check it's positive).
    details = real_resp.usage.prompt_tokens_details
    assert details is not None, (
        "prompt_tokens_details should be present "
        "(server started with --enable-prompt-tokens-details)"
    )
    assert details.cached_tokens is not None and details.cached_tokens > 0, (
        f"Expected cached_tokens > 0 after prefetch, got {details.cached_tokens}"
    )


# ------------------------------------------------------------------
# Test 3: Prefetch does NOT generate real content
# ------------------------------------------------------------------
@pytest.mark.asyncio
async def test_prefetch_no_generated_content(
    client: openai.AsyncOpenAI,
):
    """Prefetch should not produce any meaningful generated text."""

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": "Hello, how are you?"},
        ],
        extra_body={"prefetch": True},
    )

    choice = response.choices[0]
    # content should be None (no tokens generated)
    assert choice.message.content is None
