"""Optional LangSmith tracing plus a local-first OpenTelemetry fallback.

Hosted LangSmith is off by default. Core RAG works with tracing disabled.
Document text is not sent to LangSmith unless LANGSMITH_SEND_CONTENT=true.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import Any, Iterator

from config import settings
from catalog.database import log_trace

_TRACER = None


def configure_observability() -> dict:
    """Apply env flags. Never enables hosted LangSmith without an API key."""
    tracing = bool(settings.langsmith_tracing and settings.langsmith_api_key)
    if tracing:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
        if not settings.langsmith_send_content:
            os.environ["LANGCHAIN_HIDE_INPUTS"] = "true"
            os.environ["LANGCHAIN_HIDE_OUTPUTS"] = "true"
    else:
        os.environ.setdefault("LANGSMITH_TRACING", "false")
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
    _init_local_tracer()
    return {
        "langsmith_hosted": tracing,
        "langsmith_project": settings.langsmith_project if tracing else None,
        "otel_local": settings.otel_enabled,
        "send_document_content": bool(tracing and settings.langsmith_send_content),
    }


def _init_local_tracer() -> None:
    global _TRACER
    if _TRACER is not None or not settings.otel_enabled:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": "nasa-orbit"}))
        if os.getenv("ORBIT_OTEL_CONSOLE", "").lower() == "true":
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        _TRACER = trace.get_tracer("nasa-orbit")
    except Exception:
        _TRACER = False


def langchain_callbacks() -> list:
    if not (settings.langsmith_tracing and settings.langsmith_api_key):
        return []
    try:
        from langchain_core.tracers.langchain import LangChainTracer

        return [LangChainTracer(project_name=settings.langsmith_project)]
    except Exception:
        return []


def redact_payload(payload: Any) -> Any:
    if settings.langsmith_send_content:
        return payload
    if isinstance(payload, str) and len(payload) > 80:
        return {"redacted": True, "chars": len(payload)}
    if isinstance(payload, dict):
        return {k: redact_payload(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [redact_payload(v) for v in payload[:12]]
    return payload


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    started = time.perf_counter()
    error = None
    tracer_span = None
    if _TRACER and _TRACER is not False:
        tracer_span = _TRACER.start_as_current_span(name)
        tracer_span.__enter__()
        for key, value in attributes.items():
            try:
                tracer_span.set_attribute(key, json.dumps(value) if isinstance(value, (dict, list)) else value)
            except Exception:
                pass
    try:
        yield
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        duration_ms = (time.perf_counter() - started) * 1000
        try:
            log_trace(name, duration_ms, redact_payload(attributes), error)
        except Exception:
            pass
        if tracer_span is not None:
            tracer_span.__exit__(None, None, None)
