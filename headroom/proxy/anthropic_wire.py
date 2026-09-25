"""Loss-minimizing helpers for the native Anthropic Messages wire envelope.

The proxy is allowed to understand the message content it optimizes, but it
must not need to understand every field Anthropic adds to the surrounding
protocol.  This module keeps that distinction explicit: known message events
are reconstructed into a normal response dictionary, while unknown SSE frames
are retained as opaque bytes and replayed by :meth:`AnthropicSSEEnvelope.render`.

The opaque data is deliberately kept out of the provider JSON.  It is an
internal rendering detail and can therefore never accidentally be serialized
upstream as a synthetic Headroom field.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any

from headroom.copilot_auth import build_copilot_upstream_url

_DANGEROUS_TOOL_USE_RE = re.compile(r"(?:^|,)\s*dangerous-tool-use-[^,\s]+", re.IGNORECASE)
_KNOWN_EVENTS = frozenset(
    {
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    }
)
_KNOWN_DELTA_TYPES = frozenset(
    {
        "text_delta",
        "input_json_delta",
        "thinking_delta",
        "signature_delta",
        "citations_delta",
    }
)
_KNOWN_MESSAGE_FIELDS = frozenset(
    {
        "id",
        "type",
        "role",
        "model",
        "content",
        "stop_reason",
        "stop_sequence",
        "stop_details",
        "usage",
    }
)


@dataclass(frozen=True)
class _SSEFrame:
    """One raw SSE frame and its position among recognized message frames."""

    raw: bytes
    event_type: str
    payload: dict[str, Any] | None
    known_before: int


def build_anthropic_upstream_url(base_url: str, path: str, raw_query: str = "") -> str:
    """Build an Anthropic upstream URL without dropping the incoming query.

    ``raw_query`` is intentionally not parsed and re-encoded.  Beta query
    parameters are protocol data, so preserving their spelling and ordering is
    safer than normalizing them through a mapping.
    """

    url = build_copilot_upstream_url(base_url, path)
    if raw_query:
        return f"{url}?{raw_query}"
    return url


def has_dangerous_tool_use_beta(value: str | None) -> bool:
    """Return whether a beta header contains an auto-mode capability token."""

    return bool(value and _DANGEROUS_TOOL_USE_RE.search(value))


def is_safeguard_capable_request(body: Any, anthropic_beta: str | None) -> bool:
    """Classify a request without inspecting or retaining the safeguard value."""

    return isinstance(body, dict) and (
        "safeguards" in body or has_dangerous_tool_use_beta(anthropic_beta)
    )


def strip_safeguard_payload(value: Any) -> Any:
    """Copy an observation while removing classifier payload fields.

    This is only for logs, pipeline events, and diagnostics.  The actual
    upstream response is never passed through this helper, so wire fidelity is
    unaffected.
    """

    if isinstance(value, dict):
        return {
            key: strip_safeguard_payload(item)
            for key, item in value.items()
            if key not in {"safeguards", "safeguard_results"}
        }
    if isinstance(value, list):
        return [strip_safeguard_payload(item) for item in value]
    return value


def preserve_opaque_response_fields(
    original: dict[str, Any], replacement: dict[str, Any]
) -> dict[str, Any]:
    """Retain unknown Anthropic response attachments across a transform.

    Response transforms such as CCR may replace the message produced by the
    first upstream call with a continuation message.  Capability attachments
    belong to the surrounding protocol rather than to Headroom's mutable
    message view, so keep them unless the replacement explicitly supplies its
    own value.
    """

    merged = copy.deepcopy(replacement)
    for key, value in original.items():
        if key not in _KNOWN_MESSAGE_FIELDS:
            merged.setdefault(key, copy.deepcopy(value))
    return merged


def _split_frames(raw: bytes) -> list[bytes]:
    """Split SSE bytes while retaining each frame's original delimiters."""

    frames: list[bytes] = []
    start = 0
    for match in re.finditer(rb"\r\n\r\n|\n\n|\r\r", raw):
        end = match.end()
        if raw[start : match.start()].strip():
            frames.append(raw[start:end])
        start = end
    if raw[start:].strip():
        frames.append(raw[start:])
    return frames


def _frame_parts(raw: bytes) -> tuple[str, str]:
    event_name = ""
    data_lines: list[str] = []
    for line in raw.splitlines():
        if line.startswith(b"event:"):
            event_name = line[6:].strip().decode("utf-8", "replace")
        elif line.startswith(b"data:"):
            data_lines.append(line[5:].lstrip().decode("utf-8", "replace"))
    return event_name, "\n".join(data_lines)


def _json_payload(raw: bytes) -> tuple[str, dict[str, Any] | None]:
    event_name, data = _frame_parts(raw)
    if not data or data == "[DONE]":
        return event_name or "[DONE]", None
    try:
        payload = json.loads(data)
    except (TypeError, json.JSONDecodeError):
        return event_name, None
    if not isinstance(payload, dict):
        return event_name, None
    return event_name or str(payload.get("type", "")), payload


def _sse_event(payload: dict[str, Any], *, event_name: str | None = None) -> bytes:
    name = event_name or str(payload.get("type", "message"))
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


def _response_from_events(
    frames: list[_SSEFrame],
) -> tuple[dict[str, Any], bool, bool, bool, set[int]]:
    response: dict[str, Any] = {"content": [], "usage": {}}
    blocks_by_index: dict[int, dict[str, Any]] = {}
    current_block: dict[str, Any] | None = None
    appended: set[int] = set()
    open_blocks: set[int] = set()
    saw_start = saw_stop = saw_error = False

    for frame in frames:
        data = frame.payload
        event_type = frame.event_type
        if data is None:
            continue
        if event_type == "message_start":
            saw_start = True
            message = data.get("message")
            if isinstance(message, dict):
                # Start with the complete upstream message object.  This is
                # what preserves future top-level attachments.
                response.update(copy.deepcopy(message))
                # The stream's content blocks below are authoritative.  A
                # defensive non-empty content array in message_start must not
                # be duplicated when those blocks are replayed.
                response["content"] = []
                response.setdefault("usage", {})
        elif event_type == "content_block_start":
            block = data.get("content_block")
            if not isinstance(block, dict):
                continue
            index = data.get("index", len(response["content"]))
            try:
                index = int(index)
            except (TypeError, ValueError):
                index = len(response["content"])
            current_block = copy.deepcopy(block)
            blocks_by_index[index] = current_block
            open_blocks.add(index)
        elif event_type == "content_block_delta":
            delta = data.get("delta")
            if not isinstance(delta, dict):
                continue
            index = data.get("index")
            target = blocks_by_index.get(index) if index is not None else current_block
            if target is None:
                continue
            dtype = delta.get("type")
            if dtype == "text_delta":
                target["text"] = target.get("text", "") + (delta.get("text") or "")
            elif dtype == "input_json_delta":
                target["_partial_json"] = target.get("_partial_json", "") + (
                    delta.get("partial_json") or ""
                )
            elif dtype == "thinking_delta":
                target["thinking"] = target.get("thinking", "") + (delta.get("thinking") or "")
            elif dtype == "signature_delta":
                if "signature" in delta:
                    target["signature"] = delta["signature"]
            elif dtype == "citations_delta":
                citation = delta.get("citation")
                if citation is not None:
                    target.setdefault("citations", []).append(citation)
        elif event_type == "content_block_stop":
            index = data.get("index")
            target = blocks_by_index.get(index) if index is not None else current_block
            if target is None:
                continue
            partial = target.pop("_partial_json", None)
            if partial is not None:
                try:
                    target["input"] = json.loads(partial) if partial else {}
                except (TypeError, json.JSONDecodeError):
                    target["input"] = {}
            key = index if index is not None else id(target)
            if key not in appended:
                response["content"].append(target)
                appended.add(key)
            if index is not None:
                open_blocks.discard(index)
            current_block = None
        elif event_type == "message_delta":
            delta = data.get("delta")
            if isinstance(delta, dict):
                for key in ("stop_reason", "stop_sequence", "stop_details"):
                    if key in delta:
                        response[key] = copy.deepcopy(delta[key])
            usage = data.get("usage")
            if isinstance(usage, dict):
                response.setdefault("usage", {}).update(copy.deepcopy(usage))
            # Some Anthropic additions are attached directly to the delta
            # event rather than nested below ``delta``. Preserve them as
            # response fields without interpreting their schemas.
            for key, value in data.items():
                if key not in {"type", "delta", "usage"}:
                    response[key] = copy.deepcopy(value)
        elif event_type == "message_stop":
            saw_stop = True
        elif event_type == "error":
            saw_error = True

    return response, saw_start, saw_stop, saw_error, open_blocks


@dataclass
class AnthropicSSEEnvelope:
    """Parsed Anthropic response plus opaque frames retained for replay."""

    message: dict[str, Any]
    _frames: list[_SSEFrame]
    _message_delta_extras: dict[str, Any]
    saw_message_start: bool
    saw_message_stop: bool
    saw_error: bool
    open_block_indices: set[int]

    @classmethod
    def parse(cls, raw_sse_bytes: bytes) -> AnthropicSSEEnvelope:
        frames: list[_SSEFrame] = []
        known_count = 0
        for raw in _split_frames(raw_sse_bytes):
            event_type, payload = _json_payload(raw)
            is_known = event_type in _KNOWN_EVENTS
            if event_type == "content_block_delta" and isinstance(payload, dict):
                delta = payload.get("delta")
                is_known = isinstance(delta, dict) and delta.get("type") in _KNOWN_DELTA_TYPES
            frames.append(_SSEFrame(raw, event_type, payload, known_count))
            if is_known:
                known_count += 1

        message, saw_start, saw_stop, saw_error, open_blocks = _response_from_events(frames)
        delta_extras: dict[str, Any] = {}
        for frame in frames:
            if frame.event_type == "message_delta" and isinstance(frame.payload, dict):
                for key, value in frame.payload.items():
                    if key not in {"type", "delta", "usage"}:
                        delta_extras[key] = copy.deepcopy(value)
        return cls(
            message,
            frames,
            delta_extras,
            saw_start,
            saw_stop,
            saw_error,
            open_blocks,
        )

    @classmethod
    def from_events(cls, events: list[dict[str, Any]]) -> AnthropicSSEEnvelope:
        """Build an envelope for callers that already decoded event JSON."""

        raw = b"".join(_sse_event(event) for event in events if isinstance(event, dict))
        return cls.parse(raw)

    def is_complete(self) -> bool:
        return (
            self.saw_message_start
            and self.saw_message_stop
            and not self.saw_error
            and not self.open_block_indices
        )

    def render(self, message: dict[str, Any] | None = None) -> list[bytes]:
        """Render a changed message and replay unknown frames at their anchors."""

        response = preserve_opaque_response_fields(
            self.message,
            message if isinstance(message, dict) else self.message,
        )
        standard = _render_known_response(response, self._message_delta_extras)
        opaque = [
            frame
            for frame in self._frames
            if frame.event_type not in _KNOWN_EVENTS
            or (
                frame.event_type == "content_block_delta"
                and not (
                    isinstance(frame.payload, dict)
                    and isinstance(frame.payload.get("delta"), dict)
                    and frame.payload["delta"].get("type") in _KNOWN_DELTA_TYPES
                )
            )
        ]
        if opaque:
            by_anchor: dict[int, list[bytes]] = {}
            for frame in opaque:
                by_anchor.setdefault(min(frame.known_before, len(standard)), []).append(frame.raw)
            out: list[bytes] = []
            for index in range(len(standard) + 1):
                out.extend(by_anchor.get(index, []))
                if index < len(standard):
                    out.append(standard[index])
            return out
        return standard


def _render_known_response(
    response: dict[str, Any], message_delta_extras: dict[str, Any] | None = None
) -> list[bytes]:
    message = copy.deepcopy(response)
    content = message.pop("content", [])
    usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
    msg_start_message = copy.deepcopy(message)
    msg_start_message.setdefault("type", "message")
    msg_start_message.setdefault("role", "assistant")
    msg_start_message.setdefault("content", [])
    msg_start_message.setdefault("stop_reason", None)
    msg_start_message["usage"] = usage
    events = [_sse_event({"type": "message_start", "message": msg_start_message})]

    if isinstance(content, list):
        for index, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            start_block = copy.deepcopy(block)
            if block_type == "text":
                start_block["text"] = ""
            elif block_type in {"tool_use", "server_tool_use"}:
                start_block["input"] = {}
            start = {"type": "content_block_start", "index": index, "content_block": start_block}
            events.append(_sse_event(start))
            if block_type == "text" and block.get("text"):
                events.append(
                    _sse_event(
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {"type": "text_delta", "text": block["text"]},
                        }
                    )
                )
                for citation in block.get("citations", []) or []:
                    events.append(
                        _sse_event(
                            {
                                "type": "content_block_delta",
                                "index": index,
                                "delta": {"type": "citations_delta", "citation": citation},
                            }
                        )
                    )
            elif block_type in {"tool_use", "server_tool_use"} and "input" in block:
                events.append(
                    _sse_event(
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": json.dumps(
                                    block.get("input") or {}, ensure_ascii=False
                                ),
                            },
                        }
                    )
                )
            elif block_type == "thinking":
                if block.get("thinking"):
                    events.append(
                        _sse_event(
                            {
                                "type": "content_block_delta",
                                "index": index,
                                "delta": {"type": "thinking_delta", "thinking": block["thinking"]},
                            }
                        )
                    )
                if block.get("signature"):
                    events.append(
                        _sse_event(
                            {
                                "type": "content_block_delta",
                                "index": index,
                                "delta": {
                                    "type": "signature_delta",
                                    "signature": block["signature"],
                                },
                            }
                        )
                    )
            events.append(_sse_event({"type": "content_block_stop", "index": index}))

    delta: dict[str, Any] = {
        "type": "message_delta",
        "delta": {},
        "usage": {"output_tokens": usage.get("output_tokens", 0)},
    }
    for key in ("stop_reason", "stop_sequence", "stop_details"):
        if key in response:
            delta["delta"][key] = response[key]
    if message_delta_extras:
        delta.update(copy.deepcopy(message_delta_extras))
    events.append(_sse_event(delta))
    events.append(_sse_event({"type": "message_stop"}))
    return events
