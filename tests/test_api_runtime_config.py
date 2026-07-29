from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from services.api.main import app


@pytest.mark.parametrize(
    "origin",
    ["http://localhost:3000", "http://127.0.0.1:3000"],
)
def test_default_development_cors_accepts_loopback_frontend(origin: str) -> None:
    response = TestClient(app).options(
        "/health",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
