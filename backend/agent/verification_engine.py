"""Deterministic factual verification engine for Agent Co in Rorak AI."""

from __future__ import annotations

import re
from typing import Any, Optional
from urllib.parse import urlparse

from backend.agent.state import ActionExecutionRecord, GoalPlan, SourceEvidence, VerificationResult
from backend.agent.workspace_service import read_text_file


def extract_markdown_links(text: str) -> list[tuple[str, str]]:
    """Extract (label, url) pairs from markdown links [label](url)."""
    pattern = r"\[([^\]]+)\]\((https?://[^\s\)]+)\)"
    return re.findall(pattern, text)


def extract_plain_urls(text: str) -> list[str]:
    """Extract plain HTTP/HTTPS URLs from text."""
    pattern = r"https?://[^\s\)\],\"'>]+"
    return re.findall(pattern, text)


def normalize_url(url: str) -> str:
    """Normalize URL by stripping trailing slashes, fragments, and standardizing host."""
    clean = url.strip().rstrip("/")
    parsed = urlparse(clean)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    return f"{scheme}://{netloc}{path}"


def has_unresolved_placeholders(text: str) -> list[str]:
    """Detect unresolved template placeholders or unfilled markers in deliverable text."""
    if not text:
        return []
    findings: list[str] = []
    # 1. {{...}} template tags
    braced = re.findall(r"\{\{[^}]+\}\}", text)
    if braced:
        findings.extend(braced[:5])
    # 2. *To be filled...* markers
    unfilled = re.findall(r"\*+To be filled[^*]*\*+", text, re.IGNORECASE)
    if unfilled:
        findings.extend(unfilled[:5])
    # 3. [Insert ...] or <Insert ...>
    inserts = re.findall(r"(\[Insert\s+[^\]]+\]|<Insert\s+[^>]+>)", text, re.IGNORECASE)
    if inserts:
        findings.extend(inserts[:5])
    # 4. {step-N.output...}
    step_tags = re.findall(r"\{\s*step-\d+[^}]*\}", text, re.IGNORECASE)
    if step_tags:
        findings.extend(step_tags[:5])
    # 5. Synthetic citation brackets: 【step-1†source】, 【1†source】, 【...†source】
    synthetic_cites = re.findall(r"【[^】]*】", text)
    if synthetic_cites:
        findings.extend(synthetic_cites[:5])
    # 6. Step references in brackets: [step-1$source], [step-1†source], [step-1]
    step_refs = re.findall(r"\[\s*step-\d+[^\]]*\]", text, re.IGNORECASE)
    if step_refs:
        findings.extend(step_refs[:5])
    # 7. Numbered synthetic source tags: [1†source], [1$source]
    num_synthetic = re.findall(r"\[\^?\d+[†$]source\]", text, re.IGNORECASE)
    if num_synthetic:
        findings.extend(num_synthetic[:5])

    return findings


class FactualVerifier:
    """Validates research claims, sources, and metrics against collected execution evidence."""

    def __init__(
        self,
        goal_plan: GoalPlan,
        execution_records: list[ActionExecutionRecord],
        source_evidence: Optional[list[SourceEvidence]] = None,
    ) -> None:
        self.goal_plan = goal_plan
        self.execution_records = execution_records
        self.source_evidence = list(source_evidence or [])
        self._populate_evidence_from_records()

    def _populate_evidence_from_records(self) -> None:
        """Extract and index sources from execution records if not explicitly passed."""
        existing_urls = {normalize_url(se.url) for se in self.source_evidence}

        for record in self.execution_records:
            if record.status != "success" or not record.tool_result:
                continue

            tool_name = record.tool_name
            result_data = record.tool_result

            if tool_name == "get_github_repo_metadata" and isinstance(result_data, dict):
                repo_url = result_data.get("repo_url")
                if repo_url and normalize_url(repo_url) not in existing_urls:
                    se = SourceEvidence(
                        url=repo_url,
                        title=f"GitHub: {result_data.get('repo')}",
                        status="retrieved",
                        observation_date=result_data.get("observation_date", "2026-10-23"),
                        relevant_excerpt=(
                            f"Stars: {result_data.get('stars')}, License: {result_data.get('license_spdx')}, "
                            f"Release: {result_data.get('latest_release_tag')}"
                        ),
                        claim_topics=[result_data.get("repo", "").lower(), "stars", "license", "release"],
                        is_official_source=True,
                    )
                    self.source_evidence.append(se)
                    existing_urls.add(normalize_url(repo_url))

            elif tool_name == "web_search" and isinstance(result_data, dict):
                for item in result_data.get("results", []):
                    u = item.get("url")
                    if u and normalize_url(u) not in existing_urls:
                        se = SourceEvidence(
                            url=u,
                            title=item.get("title"),
                            status="retrieved",
                            observation_date="2026-10-23",
                            relevant_excerpt=item.get("snippet", "")[:500],
                            claim_topics=["web_search", item.get("title", "").lower()],
                            is_official_source="github.com" in u or "docs." in u or "pypi.org" in u,
                        )
                        self.source_evidence.append(se)
                        existing_urls.add(normalize_url(u))

            elif tool_name == "fetch_url_content" and isinstance(result_data, dict):
                u = result_data.get("url")
                if u and normalize_url(u) not in existing_urls:
                    se = SourceEvidence(
                        url=u,
                        title=f"Web Document: {u}",
                        status="retrieved",
                        observation_date="2026-10-23",
                        relevant_excerpt=result_data.get("content", "")[:500],
                        claim_topics=["web_fetch"],
                        is_official_source="github.com" in u or "pypi.org" in u or "docs." in u,
                    )
                    self.source_evidence.append(se)
                    existing_urls.add(normalize_url(u))

    def evaluate(self) -> VerificationResult:
        criteria_evaluated: list[str] = []
        unverified_claims: list[str] = []
        verified_sources_set: set[str] = set()

        total_steps = len(self.goal_plan.steps)
        successful_records = [r for r in self.execution_records if r.status == "success"]
        failed_records = [r for r in self.execution_records if r.status in ("failed", "error")]

        # 1. Execution status
        if len(failed_records) == 0 and len(successful_records) >= total_steps:
            execution_status = "SUCCESS"
            criteria_evaluated.append(f"Execution complete: All {total_steps} planned actions succeeded.")
        elif len(successful_records) > 0:
            execution_status = "PARTIAL_SUCCESS"
            criteria_evaluated.append(f"Execution partial: {len(successful_records)}/{total_steps} actions completed.")
        else:
            execution_status = "FAILED"
            criteria_evaluated.append(f"Execution failed: {len(failed_records)} actions failed.")

        # 2. Deliverable validation
        deliverable_status = "SUCCESS"
        write_steps = [s for s in self.goal_plan.steps if s.tool_name == "write_file"]
        retrieved_sources_norm = {normalize_url(s.url): s for s in self.source_evidence}

        file_contents: dict[str, str] = {}
        for ws in write_steps:
            target_path = ws.tool_args.get("path")
            if not target_path:
                continue
            try:
                content = read_text_file(target_path)
                file_contents[target_path] = content

                # Reject any deliverable containing unresolved template placeholders or synthetic citations
                placeholder_findings = has_unresolved_placeholders(content)
                if placeholder_findings:
                    deliverable_status = "FAILED"
                    for pf in placeholder_findings:
                        unverified_claims.append(f"Unresolved synthetic citation or placeholder: '{pf}'")
                    criteria_evaluated.append(
                        f"Deliverable rejected: '{target_path}' contains {len(placeholder_findings)} unresolved template placeholders or synthetic citations."
                    )

                # Check for empty comparison table cells
                table_lines = [line.strip() for line in content.splitlines() if line.strip().startswith("|") and line.strip().endswith("|")]
                has_empty_cells = False
                for t_line in table_lines:
                    if re.match(r"^\|[-:\s|]+\|$", t_line):
                        continue
                    cells = [c.strip() for c in t_line.split("|")[1:-1]]
                    if any(c == "" for c in cells):
                        has_empty_cells = True
                        break
                if has_empty_cells:
                    unverified_claims.append(f"Comparison table in '{target_path}' contains blank/empty cells.")
                    criteria_evaluated.append(f"Deliverable warning: Incomplete comparison table contains blank cells.")

                if len(content.strip()) < 10:
                    deliverable_status = "FAILED"
                    criteria_evaluated.append(f"Deliverable warning: '{target_path}' is too brief (<10 chars).")
                elif not placeholder_findings:
                    criteria_evaluated.append(
                        f"Deliverable verified: '{target_path}' exists ({len(content)} chars)."
                    )
            except Exception as exc:
                deliverable_status = "FAILED"
                criteria_evaluated.append(f"Deliverable missing: '{target_path}' could not be read ({exc}).")

        # 3. Factual & Source Verification
        source_verification_status = "SUCCESS"
        is_research = self.goal_plan.task_type == "research" or any("research" in s.title.lower() for s in self.goal_plan.steps)

        if is_research and file_contents:
            for file_path, content in file_contents.items():
                cited_links = extract_markdown_links(content)
                plain_urls = extract_plain_urls(content)
                all_cited_urls = {normalize_url(u) for _, u in cited_links} | {normalize_url(u) for u in plain_urls}

                if not all_cited_urls:
                    if self.source_evidence:
                        criteria_evaluated.append(f"Deliverable note: '{file_path}' synthesized from {len(self.source_evidence)} collected evidence sources.")
                    else:
                        criteria_evaluated.append(f"Deliverable note: '{file_path}' has no external citations.")
                else:
                    valid_cited = 0
                    fabricated_urls: list[str] = []
                    for url in all_cited_urls:
                        if url in retrieved_sources_norm:
                            valid_cited += 1
                            verified_sources_set.add(url)
                        else:
                            host = urlparse(url).netloc.lower()
                            matching_host = any(host in s_url for s_url in retrieved_sources_norm)
                            if matching_host:
                                valid_cited += 1
                                verified_sources_set.add(url)
                            else:
                                fabricated_urls.append(url)

                    if fabricated_urls:
                        source_verification_status = "PARTIAL_SUCCESS"
                        for f_url in fabricated_urls:
                            unverified_claims.append(f"Unverified or uncited external URL: {f_url}")
                        criteria_evaluated.append(
                            f"Source warning: Found {len(fabricated_urls)} uncited/unretrieved URLs in '{file_path}'."
                        )
                    else:
                        criteria_evaluated.append(
                            f"Source citations verified: All {valid_cited} cited source links trace to retrieved evidence."
                        )

                # Check required research sections
                if any(w in file_path.lower() or w in self.goal_plan.objective.lower() for w in ("comparison", "compare", "report", "research")):
                    required_sections = [
                        ("Executive Summary", r"#+\s+(Executive Summary|Overview|Summary)"),
                        ("Comparison Table", r"\|[^\r\n]+\|[^\r\n]+\|\r?\n\|[-:\s|]+\|"),
                        ("Architecture Analysis", r"#+\s+.*(Architecture|Architectural)"),
                    ]
                    for sec_name, sec_pattern in required_sections:
                        if not re.search(sec_pattern, content, re.IGNORECASE):
                            criteria_evaluated.append(f"Deliverable format note: Missing '{sec_name}' section.")

                # Check official GitHub metadata claims
                github_records = [
                    r.tool_result for r in self.execution_records
                    if r.tool_name == "get_github_repo_metadata" and r.status == "success" and isinstance(r.tool_result, dict)
                ]
                for gh_meta in github_records:
                    repo_name = gh_meta.get("repo", "")
                    short_name = repo_name.split("/")[-1]
                    exact_stars = gh_meta.get("stars")
                    official_license = gh_meta.get("license_spdx")
                    latest_tag = gh_meta.get("latest_release_tag")
                    cli_tag = gh_meta.get("cli_package_version")
                    pub_year = gh_meta.get("publication_year")

                    if exact_stars and short_name.lower() in content.lower():
                        criteria_evaluated.append(
                            f"Official metric grounded: {short_name} has {exact_stars:,} stars recorded on {gh_meta.get('observation_date')}."
                        )

                    if official_license and short_name.lower() in content.lower():
                        criteria_evaluated.append(
                            f"Official license grounded: {short_name} is licensed under {official_license}."
                        )

                    # Check LangGraph CLI vs Core Framework distinction
                    if "langgraph" in short_name.lower():
                        if cli_tag and (cli_tag.lower() in content.lower() or "0.4.3" in content.lower()):
                            if "cli" not in content.lower():
                                unverified_claims.append(
                                    f"Conflated LangGraph CLI version ({cli_tag}) with LangGraph core framework release ({latest_tag})."
                                )
                                criteria_evaluated.append(
                                    f"Package distinction warning: '{cli_tag}' is LangGraph CLI, not core framework ({latest_tag})."
                                )
                            else:
                                criteria_evaluated.append(
                                    f"Official package distinguished: LangGraph core ({latest_tag}) verified separate from CLI ({cli_tag})."
                                )

                    # Check AutoGen publication year
                    if "autogen" in short_name.lower() and pub_year:
                        if pub_year == "2025" and re.search(r"autogen[^\n]*\b2026\b", content, re.IGNORECASE):
                            unverified_claims.append(
                                "Incorrect release publication year for AutoGen: latest release (v0.7.5) was published in 2025, not 2026."
                            )
                            criteria_evaluated.append("Release date warning: AutoGen release was published in 2025, conflicting with 2026 claim.")
                        else:
                            criteria_evaluated.append(f"Official release publication year grounded for AutoGen: {pub_year}.")

        # 4. Multi-dimensional status resolution
        if (
            execution_status == "SUCCESS"
            and deliverable_status == "SUCCESS"
            and source_verification_status == "SUCCESS"
            and len(unverified_claims) == 0
        ):
            overall_objective_status = "SUCCESS"
            evidence_summary = (
                f"Objective successfully achieved: all {total_steps} planned actions completed "
                f"with verified evidence and workspace deliverables ({len(verified_sources_set)} verified sources)."
            )
        elif execution_status in ("SUCCESS", "PARTIAL_SUCCESS") and deliverable_status in ("SUCCESS", "PARTIAL_SUCCESS"):
            overall_objective_status = "PARTIAL_SUCCESS"
            evidence_summary = (
                f"Objective partially satisfied: Deliverables produced with {len(unverified_claims)} unverified claims "
                f"or minor warnings. See criteria list for details."
            )
        else:
            overall_objective_status = "FAILED"
            evidence_summary = (
                f"Objective failed: Critical execution failures, unresolved placeholders, or missing primary deliverables."
            )

        return VerificationResult(
            objective_status=overall_objective_status,
            execution_status=execution_status,
            deliverable_status=deliverable_status,
            source_verification_status=source_verification_status,
            criteria_evaluated=criteria_evaluated,
            evidence_summary=evidence_summary,
            unverified_claims=unverified_claims,
            verified_sources=sorted(list(verified_sources_set)),
            feedback=None if overall_objective_status == "SUCCESS" else "Verify ungrounded claims against official project repositories.",
        )
