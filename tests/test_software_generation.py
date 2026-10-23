"""Tests for software deliverable generation, placeholder detection, and tool resilience."""

import pytest
from backend.agent.tools_registry import GLOBAL_TOOL_REGISTRY, execute_list_files
from backend.agent.verification_engine import (
    has_unresolved_placeholders,
    validate_software_deliverable,
    FactualVerifier,
)
from backend.agent.state import ActionExecutionRecord, ActionStep, GoalPlan
from backend.agent.workspace_service import workspace_context, write_text_file


def test_placeholder_detection_css_and_comments():
    """Verify that comment placeholders and stub text are detected and rejected."""
    assert len(has_unresolved_placeholders("/* Portfolio CSS placeholder */")) > 0
    assert len(has_unresolved_placeholders("/* TODO: implement styling */")) > 0
    assert len(has_unresolved_placeholders("<!-- TODO: write content here -->")) > 0
    assert len(has_unresolved_placeholders("// placeholder function")) > 0
    assert len(has_unresolved_placeholders("# TODO: add script logic")) > 0
    assert len(has_unresolved_placeholders("{{step-1.output.stargazers_count}}")) > 0
    assert len(has_unresolved_placeholders("*To be filled from search results*")) > 0
    assert len(has_unresolved_placeholders("【step-1†source】")) > 0
    assert len(has_unresolved_placeholders("[step-1$source]")) > 0


def test_placeholder_detection_accepts_clean_code():
    """Verify valid code and markdown are not falsely flagged."""
    valid_css = """
    :root { --primary: #3b82f6; --bg: #0f172a; }
    body { margin: 0; font-family: sans-serif; background: var(--bg); color: #fff; }
    .hero { display: flex; align-items: center; justify-content: center; height: 80vh; }
    .card { padding: 20px; border-radius: 8px; border: 1px solid #334155; }
    @media (max-width: 768px) { .hero { flex-direction: column; } }
    """
    assert len(has_unresolved_placeholders(valid_css)) == 0


def test_validate_software_deliverables():
    """Verify CSS, HTML, and script structure checking and asset linkage."""
    # Defective CSS
    bad_css = "/* Portfolio CSS placeholder */"
    issues = validate_software_deliverable("style.css", bad_css, {"style.css": bad_css})
    assert len(issues) > 0

    # Valid CSS
    good_css = """
    body { background: #000; color: #fff; margin: 0; }
    .container { max-width: 1200px; margin: auto; padding: 20px; }
    .btn { background: #2563eb; color: #fff; border: none; border-radius: 4px; padding: 8px 16px; }
    """
    assert len(validate_software_deliverable("style.css", good_css, {"style.css": good_css})) == 0

    # HTML missing stylesheet
    broken_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head><link rel="stylesheet" href="missing.css"></head>
    <body><h1>Hello</h1></body>
    </html>
    """
    html_issues = validate_software_deliverable("index.html", broken_html, {"index.html": broken_html})
    assert any("missing.css" in iss for iss in html_issues)

    # HTML with valid linked stylesheet
    valid_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Portfolio</title>
        <link rel="stylesheet" href="style.css">
    </head>
    <body>
        <header><nav><a href="#">Home</a></nav></header>
        <main><h1>Welcome to My Portfolio</h1></main>
    </body>
    </html>
    """
    all_files = {"index.html": valid_html, "style.css": good_css}
    assert len(validate_software_deliverable("index.html", valid_html, all_files)) == 0


def test_tool_registry_list_files_flexibility(tmp_path):
    """Verify list_files tolerates path, directory, or extra kwargs."""
    ws_id = "test_flexibility_ws"
    with workspace_context(ws_id):
        write_text_file("index.html", "<!DOCTYPE html><html><body></body></html>")
        write_text_file("style.css", "body { color: red; }")

        # With directory
        r1 = GLOBAL_TOOL_REGISTRY.execute("list_files", {"directory": "."})
        assert r1["count"] >= 2

        # With path alias
        r2 = GLOBAL_TOOL_REGISTRY.execute("list_files", {"path": "."})
        assert r2["count"] >= 2

        # Empty dict
        r3 = GLOBAL_TOOL_REGISTRY.execute("list_files", {})
        assert r3["count"] >= 2

        # Extra kwargs from LLM
        r4 = GLOBAL_TOOL_REGISTRY.execute("list_files", {"path": ".", "ignored_argument": 42})
        assert r4["count"] >= 2


def test_factual_verifier_rejects_placeholder_deliverable():
    """Verify verifier returns FAILED if any deliverable has placeholder comments."""
    ws_id = "test_reject_placeholder_ws"
    with workspace_context(ws_id):
        write_text_file("index.html", "<!DOCTYPE html><html><body><h1>Portfolio</h1></body></html>")
        write_text_file("style.css", "/* Portfolio CSS placeholder */")

        plan = GoalPlan(
            objective="make a portfolio website",
            task_type="coding",
            expected_outcome="Portfolio website",
            steps=[
                ActionStep(id="step-1", title="HTML", tool_name="write_file", tool_args={"path": "index.html"}),
                ActionStep(id="step-2", title="CSS", tool_name="write_file", tool_args={"path": "style.css"}),
            ],
        )
        records = [
            ActionExecutionRecord(
                step_id="step-1",
                tool_name="write_file",
                tool_args={"path": "index.html"},
                started_at="2026-10-23T00:00:00Z",
                completed_at="2026-10-23T00:00:00Z",
                duration_ms=10,
                status="success",
            ),
            ActionExecutionRecord(
                step_id="step-2",
                tool_name="write_file",
                tool_args={"path": "style.css"},
                started_at="2026-10-23T00:00:00Z",
                completed_at="2026-10-23T00:00:00Z",
                duration_ms=10,
                status="success",
            ),
        ]

        verifier = FactualVerifier(goal_plan=plan, execution_records=records)
        res = verifier.evaluate()

        assert res.deliverable_status == "FAILED"
        assert res.objective_status == "FAILED"
        assert any("placeholder" in iss.lower() for iss in res.unverified_claims)


def test_factual_verifier_retried_steps_succeed():
    """Verify that a transiently failed step that retried successfully yields execution_status SUCCESS."""
    ws_id = "test_retry_success_ws"
    with workspace_context(ws_id):
        write_text_file(
            "index.html",
            "<!DOCTYPE html><html lang='en'><head><link rel='stylesheet' href='style.css'></head><body><h1>Hi</h1></body></html>"
        )
        write_text_file(
            "style.css",
            """
            body { background: #000; color: #fff; margin: 0; font-family: sans-serif; }
            .card { padding: 20px; border-radius: 8px; border: 1px solid #333; }
            .btn { background: #2563eb; color: #fff; padding: 8px 16px; border: none; }
            """
        )

        plan = GoalPlan(
            objective="make a portfolio website",
            task_type="coding",
            expected_outcome="Portfolio website",
            steps=[
                ActionStep(id="step-1", title="HTML", tool_name="write_file", tool_args={"path": "index.html"}, status="completed"),
                ActionStep(id="step-2", title="CSS", tool_name="write_file", tool_args={"path": "style.css"}, status="completed"),
            ],
        )
        # step-2 had a failed attempt first (e.g. rate limit), then succeeded
        records = [
            ActionExecutionRecord(
                step_id="step-1",
                tool_name="write_file",
                tool_args={"path": "index.html"},
                started_at="2026-10-23T00:00:00Z",
                completed_at="2026-10-23T00:00:00Z",
                duration_ms=10,
                status="success",
            ),
            ActionExecutionRecord(
                step_id="step-2",
                tool_name="write_file",
                tool_args={"path": "style.css"},
                started_at="2026-10-23T00:00:00Z",
                completed_at="2026-10-23T00:00:00Z",
                duration_ms=10,
                status="failed",
                error="Rate limit 429",
            ),
            ActionExecutionRecord(
                step_id="step-2",
                tool_name="write_file",
                tool_args={"path": "style.css"},
                started_at="2026-10-23T00:00:01Z",
                completed_at="2026-10-23T00:00:01Z",
                duration_ms=10,
                status="success",
            ),
        ]

        verifier = FactualVerifier(goal_plan=plan, execution_records=records)
        res = verifier.evaluate()

        assert res.execution_status == "SUCCESS"
        assert res.deliverable_status == "SUCCESS"
        assert res.objective_status == "SUCCESS"

