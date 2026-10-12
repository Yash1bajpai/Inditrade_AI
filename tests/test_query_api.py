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
    # Hermetic: also disable the tracked BM25 artifact (real local retrieval
    # would flip grounded to True), keeping the suite offline-only.
    monkeypatch.setattr(query_mod, "get_sparse_index", lambda: None)

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


def test_fallback_uses_supported_model_and_low_reasoning(monkeypatch, query_client):
    """Pin the verified production model and leave budget for a visible answer."""
    from types import SimpleNamespace
    import groq
    import asyncio
    _, query_mod = query_client
    monkeypatch.setenv("GROQ_API_KEY", "x" * 40)
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"quotes":["DGFT administers foreign trade policy."]}'), finish_reason="stop")])

    monkeypatch.setattr(groq, "Groq", lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    body = asyncio.run(query_mod.fallback_query("What is DGFT?", "DGFT administers foreign trade policy.", grounded=True))
    assert captured["model"] == "openai/gpt-oss-20b"
    assert captured["reasoning_effort"] == "low"
    assert captured["max_completion_tokens"] == 4096
    assert body["source"] == "Groq"
    assert body["grounded"] is True
    assert "DGFT" in body["answer"]


@pytest.mark.parametrize("content, finish_reason", [("", "stop"), (None, "stop"), ("Partial answer", "length")])
def test_fallback_rejects_empty_or_truncated_answers(monkeypatch, query_client, content, finish_reason):
    from types import SimpleNamespace
    from fastapi import HTTPException
    import groq
    import asyncio
    _, query_mod = query_client
    monkeypatch.setenv("GROQ_API_KEY", "x" * 40)
    completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)])
    monkeypatch.setattr(groq, "Groq", lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: completion))))
    with pytest.raises(HTTPException) as error:
        asyncio.run(query_mod.fallback_query("What is DGFT?", "DGFT administers foreign trade policy.", grounded=True))
    assert error.value.status_code == 502


@pytest.mark.parametrize('embedding_failure', ['status', 'timeout', 'malformed', 'qdrant'])
def test_dense_failure_keeps_sparse_grounding(monkeypatch, query_client, embedding_failure):
    from types import SimpleNamespace
    import requests
    client, mod = query_client
    sparse_hit = SimpleNamespace(payload={'doc_id': 'sparse', 'title': 'Local policy', 'text': 'Sparse policy text'})
    monkeypatch.setattr(mod, 'get_sparse_index', lambda: SimpleNamespace(search=lambda *a, **k: [sparse_hit]))
    def dense(**kwargs):
        raise RuntimeError('cloud unavailable')
    monkeypatch.setattr(mod, 'qdrant', SimpleNamespace(query_points=dense))
    def post(url, **kwargs):
        if 'feature-extraction' not in url or embedding_failure == 'timeout':
            raise requests.exceptions.Timeout('offline')
        return SimpleNamespace(status_code=402 if embedding_failure == 'status' else 200,
                               json=lambda: [[1, 2]] if embedding_failure == 'malformed' else [1.] * 384)
    monkeypatch.setattr(mod.requests, 'post', post)
    async def fallback(question, context, citation_str='', client_ip='anonymous', grounded=False):
        return {'answer': context, 'grounded': grounded, 'citation': citation_str}
    monkeypatch.setattr(mod, 'fallback_query', fallback)
    resp = client.post('/api/query/', json={'question': 'What is DGFT?'})
    assert resp.status_code == 200
    assert resp.json()['grounded'] is True
    assert 'Sparse policy text' in resp.json()['answer']


def test_real_current_qdrant_client_dense_query(monkeypatch, query_client):
    from types import SimpleNamespace
    from qdrant_client import QdrantClient, models
    import requests
    client, mod = query_client
    dense = QdrantClient(':memory:')
    dense.create_collection('trade_policy_compliance', vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
    dense.upsert('trade_policy_compliance', [models.PointStruct(id=1, vector=[1.] * 384,
        payload={'doc_id': 'dense', 'title': 'Dense policy', 'text': 'Real Qdrant policy text'})])
    monkeypatch.setattr(mod, 'qdrant', dense)
    def post(url, **kwargs):
        if 'feature-extraction' in url:
            return SimpleNamespace(status_code=200, json=lambda: [[[1.] * 384]])
        raise requests.exceptions.ConnectionError('offline generation')
    monkeypatch.setattr(mod.requests, 'post', post)
    async def fallback(question, context, citation_str='', client_ip='anonymous', grounded=False):
        return {'answer': context, 'grounded': grounded, 'citation': citation_str}
    monkeypatch.setattr(mod, 'fallback_query', fallback)
    try:
        resp = client.post('/api/query/', json={'question': 'What is DGFT?'})
        assert resp.status_code == 200
        assert resp.json()['grounded'] is True
        assert 'Real Qdrant policy text' in resp.json()['answer']
    finally:
        dense.close()


@pytest.mark.parametrize('bad_vector', [[0.] * 384, [1.] * 383, [[1.] * 384] * 2,
                                      [float('nan')] * 384, [True] * 384, ['1'] * 384])
def test_invalid_embeddings_rejected(bad_vector):
    from src.backend.api.query import _validated_query_vector
    with pytest.raises(ValueError):
        _validated_query_vector(bad_vector)


def test_cloud_e5_uses_matching_document_and_collection(monkeypatch, query_client):
    from types import SimpleNamespace
    from qdrant_client.models import Document
    client, mod = query_client
    monkeypatch.setenv('QDRANT_EMBEDDING_MODE', 'cloud_e5')
    captured = {}
    def search(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(points=[SimpleNamespace(payload={'doc_id':'e5', 'text':'E5 policy evidence', 'title':'E5 policy'})])
    monkeypatch.setattr(mod, 'qdrant', SimpleNamespace(query_points=search))
    async def fallback(question, context, citation_str='', client_ip='anonymous', grounded=False):
        return {'answer':context, 'grounded':grounded}
    monkeypatch.setattr(mod, 'fallback_query', fallback)
    response=client.post('/api/query/',json={'question':'Sugar policy?'})
    assert response.status_code == 200
    assert captured['collection_name'] == 'trade_policy_e5_small_v1'
    assert isinstance(captured['query'], Document)
    assert captured['query'].model == 'intfloat/multilingual-e5-small'


def test_chat_history_missing_client_is_not_success(monkeypatch):
    from src.backend import database
    from src.backend.api.query import log_chat_to_supabase
    monkeypatch.setattr(database, 'supabase', None)
    assert log_chat_to_supabase('s','q','a') is False


def test_chat_history_real_insert_success_and_failure(monkeypatch):
    from types import SimpleNamespace
    from src.backend import database
    from src.backend.api.query import log_chat_to_supabase
    captured={}
    def insert(row):
        captured.update(row)
        return SimpleNamespace(execute=lambda: None)
    monkeypatch.setattr(database, 'supabase',SimpleNamespace(table=lambda name: SimpleNamespace(insert=insert)))
    assert log_chat_to_supabase('s','q','a') is True
    assert captured == {'session_id':'s','user_message':'q','bot_response':'a'}
    def fail(name): raise RuntimeError('offline')
    monkeypatch.setattr(database, 'supabase',SimpleNamespace(table=fail))
    assert log_chat_to_supabase('s','q','a') is False


def test_hf_path_rejects_invented_policy_even_with_context(monkeypatch, query_client):
    from types import SimpleNamespace
    client, mod = query_client
    monkeypatch.setenv('HF_TOKEN','test')
    hit=SimpleNamespace(payload={'doc_id':'p','text':'Sugar export quota was 100 tonnes in 2023.','title':'Dated policy'})
    monkeypatch.setattr(mod,'get_sparse_index',lambda:SimpleNamespace(search=lambda *a,**k:[hit]))
    monkeypatch.setattr(mod.requests,'post',lambda *a,**k:SimpleNamespace(status_code=200,raise_for_status=lambda:None,json=lambda:{'choices':[{'message':{'content':'Sugar exports have no restrictions.'}}]}))
    r=client.post('/api/query/',json={'question':'Invent sugar export restrictions?'})
    assert r.status_code==200 and r.json()['grounded'] is False
    assert 'no restrictions' not in r.json()['answer']
    assert r.json()['history_saved'] is False
