"""Integration tests for health endpoints."""
import pytest


@pytest.mark.integration
class TestHealthEndpoints:
    async def test_health_liveness(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    async def test_health_info(self, client):
        response = await client.get("/health/info")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "environment" in data
