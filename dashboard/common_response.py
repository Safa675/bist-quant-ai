"""Shared response envelope helpers for Python engine endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def generate_run_id(prefix: str = "run") -> str:
    safe_prefix = "".join(ch if ch.isalnum() else "_" for ch in prefix).strip("_") or "run"
    return f"{safe_prefix}_{uuid4().hex[:12]}"


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_run_id(run_id: str | None = None) -> str:
    if run_id is None:
        return generate_run_id()
    candidate = str(run_id).strip()
    if not candidate:
        return generate_run_id()
    return candidate


def _response_meta(meta: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(meta or {})
    # Keep server timestamp authoritative even if caller passes timestamp in meta.
    payload["timestamp"] = _utc_iso_now()
    return payload


def success_response(
    result: Any,
    *,
    run_id: str | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": _resolve_run_id(run_id),
        "meta": _response_meta(meta),
        "result": result,
        "error": None,
    }


def error_response(
    message: str,
    *,
    code: str = "ENGINE_ERROR",
    details: Any = None,
    run_id: str | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": _resolve_run_id(run_id),
        "meta": _response_meta(meta),
        "result": None,
        "error": {
            "code": code,
            "message": str(message),
            "details": details,
        },
    }
