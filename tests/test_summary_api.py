from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import api
from astronomy import ArchiveError


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.delenv("JWT_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("REQUIRE_API_KEY", raising=False)
    api.app.state.client = object()
    with TestClient(api.app, raise_server_exceptions=True) as client:
        yield client


def test_system_summary_route(client, monkeypatch):
    result = {"systems": [{"name": "Example", "planet_count": 2}]}
    service = AsyncMock(return_value=result)
    monkeypatch.setattr(api, "summarize_system", service)
    response = client.post("/api/v1/summaries/system", json={"name": "Example"})
    assert response.status_code == 200
    assert response.json() == result
    assert service.call_args.args[1] == "Example"


@pytest.mark.parametrize("error,code", [(LookupError("missing"), 404), (ArchiveError("upstream failed"), 502), (ValueError("bad name"), 422)])
def test_system_errors(client, monkeypatch, error, code):
    monkeypatch.setattr(api, "summarize_system", AsyncMock(side_effect=error))
    assert client.post("/api/v1/summaries/system", json={"name": "Example"}).status_code == code


def test_summary_validation(client):
    assert client.post("/api/v1/summaries/system", json={"name": ""}).status_code == 422
    assert client.post("/api/v1/summaries/system", json={"name": "Example", "sql": "anything"}).status_code == 422


def test_object_summary_route(client, monkeypatch):
    monkeypatch.setattr(api, "_search", AsyncMock(return_value={"target": {"ra": 1, "dec": 2}, "counterparts": {}, "failures": []}))
    response = client.post("/api/v1/summaries/object", json={"ra": 1, "dec": 2})
    assert response.status_code == 200
    assert "candidate counterparts" in response.json()["text"]


def test_summaries_use_existing_auth(client, monkeypatch):
    monkeypatch.setenv("API_KEYS", "offline-test-key")
    assert client.post("/api/v1/summaries/system", json={"name": "Example"}).status_code == 401
