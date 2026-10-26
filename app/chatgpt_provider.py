"""Official ChatGPT-plan Responses transport; no API-key or billing fallback.

Messages remain local until one explicitly configured inference request. A failed
or interrupted POST is never replayed. Only a completed SSE response is delivered.
"""
from __future__ import annotations

import asyncio
import codecs
import hashlib
import json
import os
import re
import time
from pathlib import Path
from urllib.request import getproxies

import aiohttp

from chatgpt_auth import PlanAuthError, PlanProfile, default_profile_dir

RESOURCE = "https://api.openai.com/v1"
MAX_ERROR_BYTES = 65536
MAX_STREAM_BYTES = 2 * 1024 * 1024
MAX_EVENT_CHARS = 262144
MAX_OUTPUT_CHARS = 65536


def _tag(value, maximum=128):
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:/;+ =,-]{1," + str(maximum) + r"}", value):
        return value
    return ""


class PlanProviderError(Exception):
    """Contains machine-readable, non-secret diagnostics only."""
    def __init__(self, code, diagnostics=None):
        self.code = _tag(code, 100) or "PLAN_REQUEST_FAILED"
        self.diagnostics = dict(diagnostics or {})
        self.diagnostics["error_code"] = self.code
        super().__init__(self.code)


def build_payload(messages: list[dict], model: str) -> dict:
    """Translate text Chat Completions history without altering participant roles."""
    if not isinstance(messages, list) or not messages:
        raise PlanProviderError("INVALID_MESSAGES")
    instructions, rows = [], []
    for item in messages:
        if not isinstance(item, dict) or not isinstance(item.get("content"), str):
            raise PlanProviderError("UNSUPPORTED_MESSAGE_CONTENT")
        role, text = item.get("role"), item["content"]
        if role == "system":
            instructions.append(text)
        elif role in {"developer", "user", "assistant"}:
            rows.append({"role": role, "content": text})
        else:
            raise PlanProviderError("UNSUPPORTED_MESSAGE_ROLE")
    if not rows:
        raise PlanProviderError("NO_INPUT_MESSAGES")
    payload = {"model": model, "input": rows, "store": False, "stream": True}
    if instructions:
        payload["instructions"] = "\n\n".join(instructions)
    return payload


async def _chunks(response):
    source = response.content
    iterator = source.iter_any() if hasattr(source, "iter_any") else source
    async for chunk in iterator:
        if not isinstance(chunk, bytes):
            raise PlanProviderError("INVALID_RESPONSE_BYTES")
        if chunk:
            yield chunk


async def _limited_body(iterator, prefix=b"", limit=MAX_ERROR_BYTES):
    result = bytearray(prefix[:limit])
    if len(prefix) > limit:
        return bytes(result)
    async for chunk in iterator:
        result.extend(chunk[:max(0, limit - len(result))])
        if len(result) >= limit:
            break
    return bytes(result)


def _body_diagnostics(raw: bytes) -> dict:
    """Do not retain upstream text, arbitrary object keys, or error messages."""
    text = raw.decode("utf-8", errors="replace").lstrip()
    if not text:
        return {"body_shape": "empty"}
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return {"body_shape": "html" if text.startswith("<") else "non_json_text"}
    result = {"body_shape": "object" if isinstance(data, dict) else "array" if isinstance(data, list) else "json_scalar"}
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            result["body_shape"] = "error_object"
            code = _tag(error.get("code") or error.get("type"), 100)
            param = _tag(error.get("param"), 100)
            if code:
                result["upstream_error_code"] = code
            if param:
                result["error_param"] = param
        elif "detail" in data:
            result["body_shape"] = "detail_object"
    return result


def _response_diagnostics(response) -> dict:
    headers = response.headers
    return {"http_status": int(response.status),
            "request_id": _tag(headers.get("x-request-id") or headers.get("openai-request-id"), 160),
            "content_type": _tag(headers.get("Content-Type", ""), 160)}


async def consume_response(response, diagnostics: dict | None = None) -> str:
    """Parse actual SSE, even when its MIME header is wrong; never accept JSON as success."""
    diag = diagnostics if diagnostics is not None else {}
    diag.update(_response_diagnostics(response))
    iterator = _chunks(response).__aiter__()
    if response.status != 200:
        diag.update(_body_diagnostics(await _limited_body(iterator)))
        raise PlanProviderError("RESPONSES_HTTP_" + str(response.status), diag)

    # Inspect the body rather than treating a proxy's HTML/JSON HTTP 200 as SSE.
    # This also handles genuine SSE incorrectly labelled application/octet-stream.
    prefix = bytearray()
    while len(prefix) < 16 and b"\n" not in prefix and b"\r" not in prefix:
        try:
            prefix.extend(await anext(iterator))
        except StopAsyncIteration:
            break
    probe = bytes(prefix).removeprefix(codecs.BOM_UTF8).lstrip(b"\r\n")
    if probe and not probe.startswith((b"data:", b"event:", b":", b"id:", b"retry:")):
        diag.update(_body_diagnostics(await _limited_body(iterator, bytes(prefix))))
        raise PlanProviderError("UNEXPECTED_RESPONSE_FORMAT", diag)
    diag["body_shape"] = "sse"
    mime = str(response.headers.get("Content-Type", "")).split(";", 1)[0].strip().lower()
    diag["sse_mime_mismatch"] = mime != "text/event-stream"
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    buffer, data_lines, output = "", [], []
    output_length = 0
    event_length = 0
    total_bytes = 0
    first_line = True

    def dispatch():
        nonlocal output_length, event_length
        if not data_lines:
            return False
        raw = "\n".join(data_lines)
        data_lines.clear()
        event_length = 0
        if raw == "[DONE]":
            return False
        try:
            obj = json.loads(raw)
        except (ValueError, TypeError):
            raise PlanProviderError("MALFORMED_STREAM_EVENT", diag) from None
        if not isinstance(obj, dict):
            raise PlanProviderError("MALFORMED_STREAM_EVENT", diag)
        kind = obj.get("type", "")
        if kind == "response.output_text.delta":
            delta = obj.get("delta")
            if not isinstance(delta, str):
                raise PlanProviderError("INVALID_TEXT_DELTA", diag)
            output_length += len(delta)
            if output_length > MAX_OUTPUT_CHARS:
                raise PlanProviderError("OUTPUT_SIZE_LIMIT", diag)
            output.append(delta)
        elif kind in {"error", "response.failed", "response.incomplete"}:
            response_obj = obj.get("response") if isinstance(obj.get("response"), dict) else obj
            error = response_obj.get("error")
            if isinstance(error, dict):
                upstream = _tag(error.get("code") or error.get("type"), 100)
                if upstream:
                    diag["upstream_error_code"] = upstream
            raise PlanProviderError("INFERENCE_" + kind.replace(".", "_").upper(), diag)
        elif kind == "response.completed":
            response_obj = obj.get("response")
            if isinstance(response_obj, dict) and response_obj.get("status", "completed") != "completed":
                raise PlanProviderError("INVALID_COMPLETED_STATUS", diag)
            diag["completed_event_seen"] = True
            return True
        return False

    def line_received(line):
        nonlocal first_line, event_length
        if first_line:
            line = line.removeprefix("\ufeff")
            first_line = False
        if not line:
            return dispatch()
        if line.startswith(":"):
            return False
        field, separator, value = line.partition(":")
        if field == "data":
            if not separator:
                value = ""
            elif value.startswith(" "):
                value = value[1:]
            data_lines.append(value)
            event_length += len(value) + 1
            if event_length > MAX_EVENT_CHARS:
                raise PlanProviderError("STREAM_EVENT_SIZE_LIMIT", diag)
        # Other SSE fields are transport metadata, not generated text.
        return False

    async def all_chunks():
        if prefix:
            yield bytes(prefix)
        async for chunk in iterator:
            yield chunk

    try:
        async for chunk in all_chunks():
            total_bytes += len(chunk)
            if total_bytes > MAX_STREAM_BYTES:
                raise PlanProviderError("STREAM_SIZE_LIMIT", diag)
            buffer += decoder.decode(chunk)
            while True:
                delimiter = re.search(r"[\r\n]", buffer)
                if delimiter is None:
                    break
                index = delimiter.start()
                if buffer[index] == "\r" and index + 1 == len(buffer):
                    break
                width = 2 if buffer[index:index + 2] == "\r\n" else 1
                line, buffer = buffer[:index], buffer[index + width:]
                if line_received(line):
                    text = "".join(output).strip()
                    if not text:
                        raise PlanProviderError("COMPLETED_WITHOUT_VISIBLE_TEXT", diag)
                    return text
            if len(buffer) > MAX_EVENT_CHARS:
                raise PlanProviderError("STREAM_EVENT_SIZE_LIMIT", diag)
        buffer += decoder.decode(b"", final=True)
        # A final undelimited line is not an SSE-dispatched terminal event.
        # Even a plausible draft must not be delivered after an interrupted stream.
        if buffer.endswith("\r") and line_received(buffer[:-1]):
            text = "".join(output).strip()
            if text:
                return text
        raise PlanProviderError("STREAM_ENDED_WITHOUT_COMPLETED", diag)
    except UnicodeDecodeError:
        raise PlanProviderError("INVALID_STREAM_UTF8", diag) from None


class ChatGPTPlanProvider:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.profile = PlanProfile(cfg.get("chatgpt_profile") or None)
        self.last_diagnostics: dict = {}
        self._catalog = []
        self._catalog_identity = None
        self._catalog_time = 0.0
        self._catalog_lock = asyncio.Lock()
        self._diagnostic_dir = Path(cfg.get("chatgpt_profile") or default_profile_dir()).expanduser()

    def _record(self, diag):
        self.last_diagnostics = dict(diag, recorded_at=time.time())
        # Auth creates and restricts this directory. Do not create an unprotected
        # directory or put diagnostics in the source repository as a fallback.
        directory = self._diagnostic_dir
        if not directory.is_dir() or directory.is_symlink():
            return
        path = directory / "provider-diagnostics.jsonl"
        if path.is_symlink():
            return
        try:
            if path.exists() and path.stat().st_size > 1024 * 1024:
                return
            line = (json.dumps(self.last_diagnostics, ensure_ascii=True) + "\n").encode()
            fd = os.open(path, os.O_APPEND | os.O_WRONLY | os.O_CREAT, 0o600)
            try:
                os.write(fd, line)
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            # In-memory safe diagnostics are still available; no token is lost.
            self.last_diagnostics["diagnostic_write_failed"] = True

    def _timeout(self):
        return aiohttp.ClientTimeout(total=max(5.0, min(300.0, float(self.cfg.get("chatgpt_timeout", 90)))))

    async def _credential(self, session):
        try:
            access = await self.profile.access_token(session)
        except PlanAuthError as exc:
            raise PlanProviderError("AUTH_" + exc.code) from None
        if not isinstance(access, str) or not access:
            raise PlanProviderError("AUTH_MISSING_ACCESS_TOKEN")
        identity = getattr(self.profile, "identity_key", None)
        # A mock or older own-profile implementation may not expose identity yet.
        # Token hash stays only in memory, never in diagnostics or source files.
        key = (identity, hashlib.sha256(access.encode()).digest())
        return access, key

    async def _models(self, session, access, identity):
        async with self._catalog_lock:
            if identity == self._catalog_identity and time.monotonic() - self._catalog_time < 300:
                return list(self._catalog)
            diag = {"phase": "model_catalog", "inference_attempted": False}
            try:
                async with session.get(RESOURCE + "/models", headers={"Authorization": "Bearer " + access,
                        "Accept": "application/json"}, proxy=getproxies().get("https"),
                        timeout=self._timeout(), allow_redirects=False) as response:
                    diag.update(_response_diagnostics(response))
                    raw = await _limited_body(_chunks(response).__aiter__(), limit=MAX_STREAM_BYTES)
                    if response.status != 200:
                        diag.update(_body_diagnostics(raw))
                        raise PlanProviderError("MODEL_CATALOG_HTTP_" + str(response.status), diag)
                    try:
                        data = json.loads(raw)
                    except (ValueError, TypeError):
                        diag.update(_body_diagnostics(raw))
                        raise PlanProviderError("INVALID_MODEL_CATALOG", diag) from None
                    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
                        raise PlanProviderError("INVALID_MODEL_CATALOG", diag)
                    visible = [{"slug": m["slug"], "display_name": str(m.get("display_name") or m["slug"]), "visibility": "list"}
                        for m in data["models"] if isinstance(m, dict) and m.get("visibility") == "list"
                        and isinstance(m.get("slug"), str) and m["slug"] and len(m["slug"]) <= 200]
                    if not visible:
                        raise PlanProviderError("NO_VISIBLE_PLAN_MODELS", diag)
                    self._catalog, self._catalog_identity, self._catalog_time = visible, identity, time.monotonic()
                    diag["body_shape"] = "model_catalog"
                    self._record(diag)
                    return list(visible)
            except PlanProviderError as exc:
                self._record(dict(diag, **exc.diagnostics))
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError):
                self._record(dict(diag, error_code="MODEL_CATALOG_TRANSPORT_FAILED"))
                raise PlanProviderError("MODEL_CATALOG_TRANSPORT_FAILED", self.last_diagnostics) from None

    async def list_models(self, session) -> list[dict]:
        access, identity = await self._credential(session)
        return await self._models(session, access, identity)

    async def complete(self, session, messages: list[dict]) -> str:
        diag = {"phase": "inference", "inference_attempted": False, "completed_event_seen": False}
        self.last_diagnostics = dict(diag)
        try:
            # Validate before refreshing credentials or starting network I/O.
            payload = build_payload(messages, "pending")
            access, identity = await self._credential(session)
            models = await self._models(session, access, identity)
            requested = self.cfg.get("chatgpt_model") or "auto"
            model = models[0]["slug"] if requested == "auto" else requested
            if not isinstance(model, str) or model not in {m["slug"] for m in models}:
                raise PlanProviderError("MODEL_NOT_AVAILABLE_TO_ACCOUNT", diag)
            payload["model"] = model
            diag["model"] = _tag(model, 120)
            diag["inference_attempted"] = True
            self._record(diag)
            async with session.post(RESOURCE + "/responses", json=payload,
                    headers={"Authorization": "Bearer " + access, "Content-Type": "application/json",
                             "Accept": "text/event-stream"}, proxy=getproxies().get("https"),
                    timeout=self._timeout(), allow_redirects=False) as response:
                text = await consume_response(response, diag)
            self._record(diag)
            return text
        except PlanProviderError as exc:
            self._record(dict(diag, **exc.diagnostics))
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError):
            self._record(dict(diag, error_code="INFERENCE_TRANSPORT_FAILED"))
            raise PlanProviderError("INFERENCE_TRANSPORT_FAILED", self.last_diagnostics) from None
