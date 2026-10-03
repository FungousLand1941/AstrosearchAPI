"""Preserve the upstream astronomy endpoints when integrating the science routers."""

import pytest

import api


def test_upstream_routes_coexist_with_science_routes():
    paths = api.app.openapi()["paths"]
    for path in ("/api/v1/summaries/object", "/api/v1/summaries/system",
                 "/api/v1/signals/cross-reference", "/api/v1/search/stream",
                 "/api/v1/ai/query", "/api/v1/alerts/poll"):
        assert path in paths


@pytest.mark.asyncio
async def test_object_summary_preserves_new_search_failures(monkeypatch):
    async def search(request):
        assert request.ra == 12
        return {"target": {"ra": 12, "dec": 3}, "counterparts": {},
                "catalogs_queried": 1, "failures": [{"catalog": "gaia_dr3"}]}

    monkeypatch.setattr(api, "_search", search)
    result = await api.object_summary_endpoint(api.SearchRequest(ra=12, dec=3))
    assert result["status"] == "partial"
    assert result["failures"] == [{"catalog": "gaia_dr3"}]
