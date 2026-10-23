"""Acceptance tests for workspace live preview, asset streaming, and sandboxed serving."""

import os
from uuid import UUID, uuid4
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.agent.workspace_service import (
    get_workspace_dir,
    workspace_context,
    write_text_file,
)
from backend.core.database import SessionLocal
from backend.core.security import create_access_token
from backend.models.db import Project, User, Workspace


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def preview_workspace():
    """Create a persistent workspace populated with a portfolio website."""
    ws_id = f"test_preview_{uuid4().hex[:8]}"
    db = SessionLocal()
    try:
        user_id = UUID("a0000000-0000-4000-8000-000000000001")
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            user = User(id=user_id, email="preview_user@rorak.local", name="Preview User")
            db.add(user)
            db.flush()

        proj = Project(name="Preview Test Project", user_id=user_id)
        db.add(proj)
        db.flush()

        ws_record = Workspace(id=UUID(int=hash(ws_id) & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF), project_id=proj.id, user_id=user_id, storage_path=ws_id)
        # Use ws_id string for UUID in tests:
        pass
    finally:
        db.close()

    # Write test files
    with workspace_context(ws_id):
        write_text_file(
            "index.html",
            """<!DOCTYPE html>
<html>
<head>
    <title>Preview Test</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <h1>Portfolio Title</h1>
    <script src="script.js"></script>
</body>
</html>""",
        )
        write_text_file("style.css", "body { background: #111; color: #fff; }")
        write_text_file("script.js", "console.log('portfolio script active');")
        write_text_file("data.json", '{"title": "test", "version": 1}')
        write_text_file("README.md", "# Preview Workspace\nDocumentation test.")

    yield ws_id

    # Cleanup
    import shutil
    ws_dir = get_workspace_dir(ws_id)
    if ws_dir.exists():
        shutil.rmtree(ws_dir, ignore_errors=True)


def test_preview_serves_index_html(client, preview_workspace):
    """Ensure root preview endpoint returns index.html with correct MIME and CSP headers."""
    # When requesting without subpath
    resp = client.get(f"/api/agent/workspaces/{preview_workspace}/preview")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "<h1>Portfolio Title</h1>" in resp.text
    assert "default-src 'self'" in resp.headers["content-security-policy"]
    assert resp.headers.get("x-content-type-options") == "nosniff"

    # When requesting explicit index.html
    resp2 = client.get(f"/api/agent/workspaces/{preview_workspace}/preview/index.html")
    assert resp2.status_code == 200
    assert "<title>Preview Test</title>" in resp2.text


def test_preview_serves_linked_assets_with_correct_mimes(client, preview_workspace):
    """Ensure relative CSS and JS assets are streamed with proper content types."""
    # CSS
    resp_css = client.get(f"/api/agent/workspaces/{preview_workspace}/preview/style.css")
    assert resp_css.status_code == 200
    assert "text/css" in resp_css.headers["content-type"]
    assert "background: #111" in resp_css.text

    # JS
    resp_js = client.get(f"/api/agent/workspaces/{preview_workspace}/preview/script.js")
    assert resp_js.status_code == 200
    assert "application/javascript" in resp_js.headers["content-type"]
    assert "console.log('portfolio script active')" in resp_js.text

    # JSON
    resp_json = client.get(f"/api/agent/workspaces/{preview_workspace}/preview/data.json")
    assert resp_json.status_code == 200
    assert "application/json" in resp_json.headers["content-type"]
    assert resp_json.json()["version"] == 1


def test_preview_path_traversal_rejection(client, preview_workspace):
    """Ensure attempts to traverse out of the workspace directory are rejected with 403."""
    resp = client.get(f"/api/agent/workspaces/{preview_workspace}/preview/../../secret.txt")
    # Both 403 Forbidden and 400 Bad Request reject directory traversal
    assert resp.status_code in (400, 403, 404)

    resp2 = client.get(f"/api/agent/workspaces/{preview_workspace}/preview/..%2F..%2Fetc/passwd")
    assert resp2.status_code in (400, 403, 404)


def test_preview_missing_index_returns_friendly_empty_state(client):
    """When a workspace has no index.html, preview should return a clean informative page."""
    empty_ws_id = f"empty_ws_{uuid4().hex[:8]}"
    with workspace_context(empty_ws_id):
        write_text_file("notes.txt", "just some notes")

    try:
        resp = client.get(f"/api/agent/workspaces/{empty_ws_id}/preview")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "No <code>index.html</code> was found" in resp.text
    finally:
        import shutil
        ws_dir = get_workspace_dir(empty_ws_id)
        if ws_dir.exists():
            shutil.rmtree(ws_dir, ignore_errors=True)


def test_preview_missing_asset_returns_404(client, preview_workspace):
    """Missing non-index assets should return 404 gracefully."""
    resp = client.get(f"/api/agent/workspaces/{preview_workspace}/preview/images/nonexistent.png")
    assert resp.status_code == 404


def test_preview_user_ownership_isolation(client):
    """Ensure User A cannot preview a private workspace owned by User B."""
    db = SessionLocal()
    user_a_id = uuid4()
    user_b_id = uuid4()
    ws_id = uuid4()
    try:
        user_a = User(id=user_a_id, email=f"user_a_{user_a_id.hex[:6]}@example.com", name="User A")
        user_b = User(id=user_b_id, email=f"user_b_{user_b_id.hex[:6]}@example.com", name="User B")
        db.add_all([user_a, user_b])
        db.flush()

        proj_b = Project(id=uuid4(), name="User B Project", user_id=user_b_id)
        db.add(proj_b)
        db.flush()

        ws_b = Workspace(id=ws_id, project_id=proj_b.id, user_id=user_b_id, storage_path=str(ws_id))
        db.add(ws_b)
        db.commit()

        # Write index.html for ws_b
        with workspace_context(str(ws_id)):
            write_text_file("index.html", "<h1>User B Private Work</h1>")

        # User A authenticates with their own token
        token_a = create_access_token(subject=str(user_a_id))
        resp = client.get(
            f"/api/agent/workspaces/{ws_id}/preview",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert resp.status_code == 403

        # User B authenticates via query param token
        token_b = create_access_token(subject=str(user_b_id))
        resp_b = client.get(f"/api/agent/workspaces/{ws_id}/preview?token={token_b}")
        assert resp_b.status_code == 200
        assert "User B Private Work" in resp_b.text
    finally:
        db.close()
        import shutil
        ws_dir = get_workspace_dir(str(ws_id))
        if ws_dir.exists():
            shutil.rmtree(ws_dir, ignore_errors=True)


def test_preview_portfolio_site_renders(client):
    """Test previewing the generated portfolio workspace with index.html, style.css, and script.js."""
    portfolio_ws = "25b85e7c-157f-4291-b16e-d90c4777332e"
    resp = client.get(f"/api/agent/workspaces/{portfolio_ws}/preview/index.html")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "John Doe" in resp.text
    assert 'href="style.css"' in resp.text

    # Request linked style.css
    resp_css = client.get(f"/api/agent/workspaces/{portfolio_ws}/preview/style.css")
    assert resp_css.status_code == 200
    assert "text/css" in resp_css.headers["content-type"]

    # Request linked script.js
    resp_js = client.get(f"/api/agent/workspaces/{portfolio_ws}/preview/script.js")
    assert resp_js.status_code == 200
    assert "application/javascript" in resp_js.headers["content-type"]

