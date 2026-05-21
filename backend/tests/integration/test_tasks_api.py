"""Integration tests for task management API."""
import pytest
from uuid import uuid4


@pytest.mark.integration
class TestTasksAPI:
    async def test_create_task(self, client, sample_task_data):
        response = await client.post("/api/v1/tasks", json=sample_task_data)
        assert response.status_code in (200, 201)
        data = response.json()
        assert data["title"] == sample_task_data["title"]
        assert "id" in data

    async def test_list_tasks_empty(self, client):
        response = await client.get("/api/v1/tasks")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, (list, dict))

    async def test_get_nonexistent_task(self, client):
        response = await client.get(f"/api/v1/tasks/{uuid4()}")
        assert response.status_code == 404
