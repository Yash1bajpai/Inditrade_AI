"""
Tests for the RAG query endpoint's error semantics.

The audit found two problems this file pins down:
  1. When both the Hugging Face and Groq upstreams were unavailable, the handler
     returned HTTP 200 with {"source": "Error"} and the error text in the answer
     field, so clients had to string-match the body to detect failure.
  2. RAG retrieval failure was swallowed and the request continued with an empty
     context, making an ungrounded answer indistinguishable from a grounded one.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def query_client(monkeypatch):
    """TestClient over the query router with Supabase logging disabled."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.backend.api import query as query_mod

    monkeypatch.setattr(query_mod, "log_chat_to_supabase", lambda *a, **k: None)
    # Reset the in-memory rate limiter so test ordering can't trip the 429.
    monkeypatch.setattr(query_mod, "ip_requests", {})
    # No Qdrant in tests -> retrieval is skipped, so grounded must be False.
    monkeypatch.setattr(query_mod, "qdrant", None, raising=False)

    # Keep the suite hermetic and fast: never let the HF branch hit the network.
    # Without this the handler spends its full 10s timeout per test before
    # falling through to the Groq path.
    import requests

    def no_network(*args, **kwargs):
        raise requests.exceptions.ConnectionError("network disabled in tests")

    monkeypatch.setattr(query_mod.requests, "post", no_network)

    app = FastAPI()
    app.include_router(query_mod.router, prefix="/api/query")
    return TestClient(app, raise_server_exceptions=False), query_mod


def test_no_upstream_credentials_returns_503_not_200(monkeypatch, query_client):
    """AUDIT REGRESSION: this used to be a 200 carrying an error string."""
    client, _ = query_client
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY1", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    resp = client.post("/api/query/", json={"question": "What is the duty on cotton exports?"})

    assert resp.status_code == 503, \
        f"expected 503 when both upstreams are down, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert "detail" in body
    assert "unavailable" in body["detail"].lower()
    # The old shape leaked the failure through a 200 body — make sure it's gone.
    assert body.get("source") != "Error"


def test_short_groq_key_is_treated_as_absent(monkeypatch, query_client):
    """A truncated/placeholder key must not be attempted against the API."""
    client, _ = query_client
    monkeypatch.setenv("GROQ_API_KEY", "abc")  # len < 10
    monkeypatch.delenv("HF_TOKEN", raising=False)

    resp = client.post("/api/query/", json={"question": "Explain FTP 2023 briefly."})
    assert resp.status_code == 503


def test_successful_groq_answer_reports_grounded_false(monkeypatch, query_client):
    """With no Qdrant, an answer is ungrounded and must say so explicitly."""
    client, query_mod = query_client
    monkeypatch.setenv("GROQ_API_KEY", "x" * 40)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    async def fake_fallback(question, context, citation_str="", client_ip="anonymous", grounded=False):
        return {"answer": "- Stub policy answer", "source": "Groq",
                "citation": citation_str, "grounded": grounded}

    monkeypatch.setattr(query_mod, "fallback_query", fake_fallback)

    resp = client.post("/api/query/", json={"question": "Is rice export restricted?"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "grounded" in body, "response must expose whether it was RAG-grounded"
    assert body["grounded"] is False
    assert body["source"] == "Groq"


def test_rate_limit_still_enforced(monkeypatch, query_client):
    """The 429 path must survive the error-semantics refactor."""
    client, query_mod = query_client
    monkeypatch.setenv("GROQ_API_KEY", "x" * 40)

    async def fake_fallback(question, context, citation_str="", client_ip="anonymous", grounded=False):
        return {"answer": "ok", "source": "Groq", "citation": "", "grounded": grounded}

    monkeypatch.setattr(query_mod, "fallback_query", fake_fallback)

    statuses = [
        client.post("/api/query/", json={"question": f"q{i}"}).status_code
        for i in range(query_mod.RATE_LIMIT + 2)
    ]
    assert 429 in statuses, f"rate limiter never fired: {statuses}"


def test_question_length_is_validated(query_client):
    """The 500-char cap on question is enforced by pydantic (422)."""
    client, _ = query_client
    resp = client.post("/api/query/", json={"question": "x" * 501})
    assert resp.status_code == 422
