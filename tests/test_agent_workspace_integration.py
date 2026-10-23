"""End-to-End and Component Integration Tests for Agent Co in Rorak AI."""

import io
import os
import zipfile
import pytest
from uuid import uuid4
from fastapi.testclient import TestClient

from backend.main import app
from backend.core.security import create_access_token
from backend.core.database import get_db, Base, engine
from backend.models.db import User
from backend.agent.workspace_service import (
    write_text_file,
    read_text_file,
    list_tree,
    delete_file,
    create_zip_archive,
    resolve_safe_path,
    workspace_context,
)
from backend.agent.verification_engine import FactualVerifier, has_unresolved_placeholders
from backend.agent.state import GoalPlan, ActionStep, ActionExecutionRecord, SourceEvidence
from backend.agent.tools_registry import GLOBAL_TOOL_REGISTRY


@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def auth_client():
    client = TestClient(app)
    # Create persistent test user in DB
    gen = get_db()
    db = next(gen)
    try:
        user_id = uuid4()
        user = User(id=user_id, email=f"agent_test_{user_id.hex[:8]}@example.com")
        db.add(user)
        db.commit()

        token = create_access_token(data={"sub": str(user.id), "email": user.email})
        client.headers["Authorization"] = f"Bearer {token}"
        yield client
    finally:
        db.close()


def test_workspace_service_operations():
    ws_id = str(uuid4())
    with workspace_context(ws_id):
        # 1. Write file
        rel_path = write_text_file("docs/report.md", "# Test Report\n\nContent grounded in facts.")
        assert resolve_safe_path(rel_path).exists()

        # 2. Read file
        content = read_text_file("docs/report.md")
        assert "Content grounded in facts." in content

        # 3. List tree
        tree = list_tree()
        assert len(tree) >= 1
        assert tree[0]["name"] == "docs"
        assert tree[0]["type"] == "directory"
        assert len(tree[0]["children"]) == 1
        assert tree[0]["children"][0]["name"] == "report.md"

        # 4. Zip archive
        zip_bytes = create_zip_archive()
        assert len(zip_bytes) > 0
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            namelist = zf.namelist()
            assert "docs/report.md" in namelist

        # 5. Delete file
        delete_file("docs/report.md")
        assert not resolve_safe_path(rel_path).exists()


def test_factual_verifier_rejects_synthetic_tokens():
    bad_report = """# Framework Comparison
LangGraph version is {{step-1.output}} with stars 【step-1†source】.
License is spdx: [step-2$source].
*To be filled from search results*
"""
    findings = has_unresolved_placeholders(bad_report)
    assert len(findings) >= 3
    assert any("step-1" in f for f in findings)


def test_factual_verifier_accepts_clean_evidence_report():
    ws_id = str(uuid4())
    clean_report = """# Framework Comparison

## Executive Summary
Comprehensive comparison of autonomous agent frameworks.

## Comparison Criteria
Architecture, License, and Releases.

## Comparison Table
| Framework | Primary Architecture | License | Release | Source |
|---|---|---|---|---|
| LangGraph | Cyclical Graph | MIT | 0.2.0 | [Release](https://github.com/langchain-ai/langgraph) |

## Source Citations
- [LangGraph Repository](https://github.com/langchain-ai/langgraph)
"""
    with workspace_context(ws_id):
        write_text_file("framework_comparison.md", clean_report)

        goal_plan = GoalPlan(
            objective="Compare frameworks",
            task_type="research",
            deliverable_path="framework_comparison.md",
            steps=[
                ActionStep(id="step-1", title="Write report", description="Write final report", tool_name="write_file", tool_args={"path": "framework_comparison.md"})
            ]
        )
        execution_records = [
            ActionExecutionRecord(step_id="step-1", tool_name="write_file", tool_args={"path": "framework_comparison.md"}, tool_result={"bytes": len(clean_report)}, status="success", duration_ms=10)
        ]
        evidence = [
            SourceEvidence(url="https://github.com/langchain-ai/langgraph", title="LangGraph", status="retrieved", observation_date="2026-10-23", relevant_excerpt="Cyclical graph runtime", is_official_source=True)
        ]

        verifier = FactualVerifier(goal_plan=goal_plan, execution_records=execution_records, source_evidence=evidence)
        res = verifier.evaluate()

        assert res.objective_status == "SUCCESS"
        assert res.deliverable_status == "SUCCESS"
        assert res.source_verification_status == "SUCCESS"


def test_deterministic_data_analysis_calculator():
    csv_sample = """department,revenue,headcount
Sales,120.5,10
Engineering,250.0,25
Marketing,85.2,8
Support,45.0,15
Operations,95.0,12
"""
    tool_spec = GLOBAL_TOOL_REGISTRY.get("data_analysis_calc")
    assert tool_spec is not None

    result = tool_spec.handler(csv_content=csv_sample, target_columns=["revenue"])
    assert result["status"] == "success"
    assert result["dataset_summary"]["total_rows"] == 5
    rev_stats = result["descriptive_statistics"]["revenue"]
    assert rev_stats["mean"] == pytest.approx(119.14, 0.01)
    assert rev_stats["min"] == 45.0
    assert rev_stats["max"] == 250.0
    assert rev_stats["median"] == 95.0
    assert "missing_values_by_column" in result["dataset_summary"]


def test_agent_workspace_rest_api_lifecycle(auth_client):
    # 1. Create a project
    create_resp = auth_client.post(
        "/api/agent/projects",
        json={"name": "Integration Test Project", "description": "Testing full REST lifecycle"}
    )
    assert create_resp.status_code == 201
    proj_data = create_resp.json()
    assert "id" in proj_data
    assert "workspace_id" in proj_data
    ws_id = proj_data["workspace_id"]

    # 2. Get projects list
    list_resp = auth_client.get("/api/agent/projects")
    assert list_resp.status_code == 200
    res_data = list_resp.json()
    projects = res_data.get("projects", res_data)
    assert any(p["id"] == proj_data["id"] for p in projects)

    # 3. Create file in workspace
    file_resp = auth_client.post(
        f"/api/agent/workspaces/{ws_id}/files",
        json={"path": "src/main.py", "content": "print('Rorak Agent Co active')"}
    )
    assert file_resp.status_code == 200

    # 4. Read tree
    tree_resp = auth_client.get(f"/api/agent/workspaces/{ws_id}/tree")
    assert tree_resp.status_code == 200
    tree_data = tree_resp.json()
    tree = tree_data.get("tree", tree_data)
    assert any(node["name"] == "src" for node in tree)

    # 5. Read file
    read_resp = auth_client.get(f"/api/agent/workspaces/{ws_id}/files?path=src/main.py")
    assert read_resp.status_code == 200
    assert read_resp.json()["content"] == "print('Rorak Agent Co active')"

    # 6. Download ZIP
    zip_resp = auth_client.get(f"/api/agent/workspaces/{ws_id}/zip")
    assert zip_resp.status_code == 200
    assert zip_resp.headers["content-type"] == "application/zip"
    zip_content = io.BytesIO(zip_resp.content)
    with zipfile.ZipFile(zip_content, "r") as zf:
        assert "src/main.py" in zf.namelist()


def test_rorak_core_regression_routes(auth_client):
    health_resp = auth_client.get("/health/")
    assert health_resp.status_code == 200
    assert health_resp.json().get("status") in {"healthy", "ok"}
