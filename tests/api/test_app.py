"""Unit/integration tests for the FastAPI app factory (app.py)."""

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sign_language.api.app import create_app
from sign_language.core.settings import settings

_LOAD_ALL = "sign_language.api.app.load_all"


# ---------------------------------------------------------------------------
# App factory — no lifespan triggered, no TestClient needed
# ---------------------------------------------------------------------------


class TestCreateApp:
    def test_returns_fastapi_instance(self, mock_models):
        """Verify create app returns fastapi instance."""
        with patch(_LOAD_ALL, return_value=mock_models):
            app = create_app()
        assert isinstance(app, FastAPI)

    def test_title_comes_from_settings(self, mock_models):
        """Verify create app title comes from settings."""
        with patch(_LOAD_ALL, return_value=mock_models):
            app = create_app()
        assert app.title == settings.app_name

    def test_version_comes_from_settings(self, mock_models):
        """Verify create app version comes from settings."""
        with patch(_LOAD_ALL, return_value=mock_models):
            app = create_app()
        assert app.version == settings.version


# ---------------------------------------------------------------------------
# Lifespan — TestClient INSIDE the patch so the mock is active when
# lifespan fires load_all()
# ---------------------------------------------------------------------------


class TestLifespan:
    def test_load_all_called_on_startup(self, mock_models):
        """Verify lifespan load all called on startup."""
        with patch(_LOAD_ALL, return_value=mock_models) as mock_load:
            app = create_app()
            with TestClient(app):
                mock_load.assert_called_once()

    def test_app_state_is_set_after_startup(self, mock_models):
        """Verify lifespan app state is set after startup."""
        with patch(_LOAD_ALL, return_value=mock_models):
            app = create_app()
            with TestClient(app):
                assert hasattr(app.state, "app_state")

    def test_app_state_holds_correct_models(self, mock_models):
        """Verify lifespan app state holds correct models."""
        with patch(_LOAD_ALL, return_value=mock_models):
            app = create_app()
            with TestClient(app):
                assert app.state.app_state.models is mock_models


# ---------------------------------------------------------------------------
# CORS — TestClient INSIDE the patch
# ---------------------------------------------------------------------------


class TestCORS:
    def test_cors_allows_configured_origin(self, mock_models):
        """Verify c o r s cors allows configured origin."""
        origin = "http://localhost:5173"
        with (
            patch(_LOAD_ALL, return_value=mock_models),
            patch.object(settings, "cors_origins", [origin]),
        ):
            app = create_app()
            with TestClient(app) as c:
                r = c.get("/api/health", headers={"Origin": origin})
        assert r.status_code == 200
        assert r.headers.get("access-control-allow-origin") == origin

    def test_cors_wildcard_when_no_origins_configured(self, mock_models):
        """Verify c o r s cors wildcard when no origins configured."""
        origin = "http://anything.example"
        with (
            patch(_LOAD_ALL, return_value=mock_models),
            patch.object(settings, "cors_origins", []),
        ):
            app = create_app()
            with TestClient(app) as c:
                r = c.get("/api/health", headers={"Origin": origin})
        assert r.status_code == 200
        assert r.headers.get("access-control-allow-origin") == "*"


# ---------------------------------------------------------------------------
# Router prefix — TestClient INSIDE the patch
# ---------------------------------------------------------------------------


class TestRouterPrefix:
    def test_health_reachable_at_api_prefix(self, mock_models):
        """Verify router prefix health reachable at api prefix."""
        with patch(_LOAD_ALL, return_value=mock_models):
            app = create_app()
            with TestClient(app) as c:
                assert c.get("/api/health").status_code == 200

    def test_unprefixed_health_not_found(self, mock_models):
        """Verify router prefix unprefixed health not found."""
        with patch(_LOAD_ALL, return_value=mock_models):
            app = create_app()
            with TestClient(app, raise_server_exceptions=False) as c:
                r = c.get("/health")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Static file mount — TestClient INSIDE the patch
# ---------------------------------------------------------------------------


class TestStaticFileMount:
    def test_no_mount_when_dir_missing(self, mock_models, tmp_path):
        """Verify static file mount no mount when dir missing."""
        nonexistent = tmp_path / "dist_missing"
        with (
            patch(_LOAD_ALL, return_value=mock_models),
            patch.object(settings, "frontend_build_dir", nonexistent),
        ):
            app = create_app()
            with TestClient(app) as c:
                assert c.get("/api/health").status_code == 200

    def test_index_html_served_when_dir_exists(self, mock_models, tmp_path):
        """Verify static file mount index html served when dir exists."""
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<html><body>Vue app</body></html>")

        with (
            patch(_LOAD_ALL, return_value=mock_models),
            patch.object(settings, "frontend_build_dir", dist),
        ):
            app = create_app()
            with TestClient(app) as c:
                r = c.get("/")
        assert r.status_code == 200
        assert "Vue app" in r.text

    def test_api_routes_win_over_static_mount(self, mock_models, tmp_path):
        """Verify static file mount api routes win over static mount."""
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<html/>")

        with (
            patch(_LOAD_ALL, return_value=mock_models),
            patch.object(settings, "frontend_build_dir", dist),
        ):
            app = create_app()
            with TestClient(app) as c:
                r = c.get("/api/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}
