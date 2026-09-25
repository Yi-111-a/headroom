from __future__ import annotations

import json
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from headroom.proxy.anthropic_wire import (
    AnthropicSSEEnvelope,
    build_anthropic_upstream_url,
    has_dangerous_tool_use_beta,
    is_safeguard_capable_request,
    strip_safeguard_payload,
)
from headroom.proxy.handlers.streaming import StreamingMixin
from headroom.proxy.server import ProxyConfig, create_app

FIXTURES = Path(__file__).parent / "fixtures" / "anthropic"


def _fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def test_native_url_preserves_path_and_raw_query() -> None:
    assert (
        build_anthropic_upstream_url(
            "https://api.anthropic.com", "/v1/messages", "beta=true&beta=false"
        )
        == "https://api.anthropic.com/v1/messages?beta=true&beta=false"
    )


def test_classifier_request_detection_does_not_need_safeguard_schema() -> None:
    request = json.loads(_fixture("claude_code_auto_mode_request.json"))
    assert is_safeguard_capable_request(request, "prompt-caching-2024-07-31")
    assert has_dangerous_tool_use_beta("prompt-caching-2024-07-31,dangerous-tool-use-2026-09")
    assert not is_safeguard_capable_request({}, "prompt-caching-2024-07-31")


def test_sse_round_trip_preserves_opaque_result_order_and_ids() -> None:
    envelope = AnthropicSSEEnvelope.parse(_fixture("claude_code_auto_mode_stream.sse"))
    assert envelope.message["safeguard_results"]["decision"] == "allow"
    assert envelope.message["content"][0]["id"] == "toolu_auto_001"

    rendered = b"".join(envelope.render(envelope.message))
    safeguard_offset = rendered.index(b"event: safeguard_results")
    delta_offset = rendered.index(b"event: message_delta")
    stop_offset = rendered.index(b"event: message_stop")
    assert safeguard_offset < delta_offset < stop_offset
    assert b'"id": "toolu_auto_001"' in rendered


def test_sse_round_trip_keeps_results_across_a_ccr_continuation() -> None:
    envelope = AnthropicSSEEnvelope.parse(_fixture("claude_code_auto_mode_stream.sse"))
    continuation = {
        "id": "msg_continuation_001",
        "type": "message",
        "role": "assistant",
        "model": "claude-test-20260924",
        "content": [{"type": "text", "text": "retrieval complete"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 20, "output_tokens": 4},
    }

    rendered = b"".join(envelope.render(continuation))

    assert b"safeguard_results" in rendered
    assert b"toolu_auto_001" in rendered
    assert b"msg_continuation_001" in rendered


def test_unknown_content_delta_is_complete_and_replayed_verbatim() -> None:
    raw = b"""event: message_start
data: {"type":"message_start","message":{"id":"msg_future","type":"message","role":"assistant","content":[]}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"future_delta","value":"opaque"}}

event: message_stop
data: {"type":"message_stop"}

"""

    envelope = AnthropicSSEEnvelope.parse(raw)

    assert envelope.is_complete()
    assert b'"type":"future_delta"' in b"".join(envelope.render())


def test_non_stream_response_keeps_unknown_top_level_fields() -> None:
    response = json.loads(_fixture("claude_code_auto_mode_response.json"))
    rendered = b"".join(StreamingMixin()._response_to_sse(response, "anthropic"))
    assert b"safeguard_results" in rendered
    assert b"toolu_auto_001" in rendered


def test_observation_copy_removes_classifier_payloads() -> None:
    value = {
        "safeguards": {"sentinel": "do-not-log"},
        "nested": [{"safeguard_results": {"sentinel": "do-not-log"}}],
        "safe": "kept",
    }
    observed = strip_safeguard_payload(value)
    assert observed == {"nested": [{}], "safe": "kept"}
    assert value["safeguards"]["sentinel"] == "do-not-log"


class _AutoModeFixtureStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield _fixture("claude_code_auto_mode_stream.sse")


class _CapturingAnthropicTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.url: str | None = None
        self.headers: dict[str, str] | None = None
        self.body: bytes | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.url = str(request.url)
        self.headers = dict(request.headers.items())
        self.body = b"".join([chunk async for chunk in request.stream])
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_AutoModeFixtureStream(),
        )


def test_native_handler_preserves_classifier_request_envelope() -> None:
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )
    app = create_app(config)
    proxy = app.state.proxy
    transport = _CapturingAnthropicTransport()
    proxy.http_client = httpx.AsyncClient(transport=transport)
    request = json.loads(_fixture("claude_code_auto_mode_request.json"))
    request["model"] = "claude-sonnet-4-6"

    with TestClient(app).stream(
        "POST",
        "/v1/messages?beta=true",
        headers={
            "x-api-key": "test-key",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "claude-code-20250219,dangerous-tool-use-2026-09-03",
            "content-type": "application/json",
        },
        content=json.dumps(request).encode(),
    ) as response:
        assert response.status_code == 200
        assert b"safeguard_results" in b"".join(response.iter_bytes())

    assert transport.url == "https://api.anthropic.com/v1/messages?beta=true"
    assert transport.headers is not None
    assert (
        transport.headers["anthropic-beta"] == "claude-code-20250219,dangerous-tool-use-2026-09-03"
    )
    assert transport.body is not None
    assert json.loads(transport.body)["safeguards"] == request["safeguards"]
