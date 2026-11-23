"""Streamlit UI component for the Verified Alumni Finder feature."""

from __future__ import annotations

import asyncio
import csv
import io
import logging
from typing import Any

import streamlit as st

from services.alumni.config import AlumniConfig
from services.alumni.pipeline import run_alumni_pipeline
from services.alumni.schemas import AlumniPipelineReport, VerificationStatus

logger = logging.getLogger(__name__)


def _build_alumni_csv(report: AlumniPipelineReport) -> str:
    """Generate CSV string of verified alumni candidates."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Full Name",
        "Company",
        "Designation",
        "Profile URL",
        "Education Institution",
        "Education Degree",
        "Education Quote",
        "Education Source URL",
        "Employment Quote",
        "Employment Source URL",
        "Date Checked",
        "Status",
    ])

    for cr in report.company_results:
        for c in cr.verified_alumni:
            edu_inst = c.education.institution if c.education else ""
            edu_deg = c.education.degree or "" if c.education else ""
            edu_quote = c.education_evidence.quote if c.education_evidence else ""
            edu_url = str(c.education_evidence.source_url) if c.education_evidence else ""
            emp_quote = c.employment_evidence.quote if c.employment_evidence else ""
            emp_url = str(c.employment_evidence.source_url) if c.employment_evidence else ""
            dt_checked = str(c.education_evidence.date_checked) if c.education_evidence else ""

            writer.writerow([
                c.full_name,
                cr.company_group.canonical_name,
                c.designation or "",
                str(c.profile_url) if c.profile_url else "",
                edu_inst,
                edu_deg,
                edu_quote,
                edu_url,
                emp_quote,
                emp_url,
                dt_checked,
                "VERIFIED",
            ])

    return output.getvalue()


def render_alumni_tab(
    validation_report: Any,
    default_config: AlumniConfig | None = None,
) -> None:
    """Render the '🎓 GCT Alumni' tab inside the main Streamlit application."""
    st.subheader("🎓 Verified GCT Alumni Directory")
    st.markdown(
        "Find verified **Government College of Technology (GCT)** alumni working at each company "
        "discovered in your job search. Only candidates satisfying **both** independent education "
        "and current employment criteria are displayed."
    )

    cfg = default_config or AlumniConfig()

    # Sidebar / Configuration inspection
    search_key = cfg.serper_api_key
    if cfg.search_provider == "brave":
        search_key = cfg.brave_api_key
    elif cfg.search_provider == "tavily":
        search_key = cfg.tavily_api_key

    has_search_key = bool(search_key and search_key.strip())

    # Check job results
    active_jobs = []
    if validation_report is not None:
        if hasattr(validation_report, "active_results"):
            active_jobs = validation_report.active_results
        elif isinstance(validation_report, list):
            active_jobs = validation_report

    if not active_jobs:
        st.info(
            "👉 Please run the job search pipeline in **'⚙️ Search Criteria & Filters'** first. "
            "Alumni discovery runs on companies extracted from the active job results."
        )
        return

    # Warning for missing search key
    if not has_search_key:
        provider_name = cfg.search_provider.upper()
        st.error(
            f"⚠️ **Search API Key Missing:** No `{provider_name}_API_KEY` was found in `.env` or secrets. "
            f"Please configure your Search API key to enable verified alumni discovery."
        )

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        find_btn = st.button(
            "🔍 Find Verified Alumni",
            type="primary",
            disabled=not has_search_key or len(active_jobs) == 0,
            help="Run the multi-stage alumni extraction and verification pipeline on active companies.",
        )
    with col_info:
        st.caption(
            f"Target: **{cfg.college_name}** | Provider: **{cfg.search_provider}** | "
            f"Max queries/company: **{cfg.max_queries_per_company}**"
        )

    # Pipeline execution
    if find_btn:
        progress_bar = st.progress(0.0)
        status_box = st.empty()

        def _ui_status_callback(msg: str, progress: float) -> None:
            progress_bar.progress(min(max(progress, 0.0), 1.0))
            status_box.markdown(f"**Status:** {msg}")

        with st.spinner("Executing alumni verification pipeline..."):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                alumni_report = loop.run_until_complete(
                    run_alumni_pipeline(
                        job_results=active_jobs,
                        config=cfg,
                        status_callback=_ui_status_callback,
                    )
                )
                loop.close()
                st.session_state.alumni_report = alumni_report
                progress_bar.progress(1.0)
                status_box.success("✅ Alumni verification completed successfully!")
            except Exception as e:
                logger.exception("Alumni pipeline execution error: %s", e)
                status_box.error(f"❌ Error during alumni search pipeline: {e}")

    # Display results if available in session_state
    report: AlumniPipelineReport | None = st.session_state.get("alumni_report")
    if report is None:
        return

    st.markdown("---")

    # Summary Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🏢 Companies Evaluated", report.companies_evaluated)
    m2.metric("🎓 Verified Alumni Found", report.total_verified)
    m3.metric("⚪ Companies With No Match", report.companies_no_match)
    m4.metric("📡 Search API Calls Used", report.api_calls_used)

    # Download CSV button
    if report.total_verified > 0:
        csv_data = _build_alumni_csv(report)
        st.download_button(
            label="📥 Download verified alumni (CSV)",
            data=csv_data,
            file_name="verified_gct_alumni.csv",
            mime="text/csv",
        )

    st.markdown("### 🏢 Company Findings")

    for cr in report.company_results:
        canonical = cr.company_group.canonical_name
        verified_count = len(cr.verified_alumni)
        badge_text = f"({verified_count} verified)" if verified_count > 0 else "(0 verified)"

        with st.expander(f"🏢 {canonical} {badge_text}", expanded=(verified_count > 0)):
            if cr.status == "search_failed":
                st.warning(f"⚠️ Search failed for this company: {cr.message}")
            elif not cr.verified_alumni:
                st.info(
                    "No verified alumnus found — No candidate met the required education and "
                    "current-employment verification criteria."
                )
            else:
                for idx, a in enumerate(cr.verified_alumni, start=1):
                    col_lead, col_badge = st.columns([4, 1])
                    with col_lead:
                        st.markdown(f"#### {idx}. {a.full_name}")
                        if a.designation:
                            st.markdown(f"💼 **{a.designation}** @ **{canonical}**")
                        else:
                            st.markdown(f"💼 Employee @ **{canonical}**")
                    with col_badge:
                        st.markdown(
                            '<div style="text-align:right;"><span style="background-color:#10b981;'
                            'color:#ffffff;padding:4px 10px;border-radius:12px;font-size:0.85rem;'
                            'font-weight:600;">✅ Verified</span></div>',
                            unsafe_allow_html=True,
                        )

                    if a.profile_url:
                        st.markdown(f"🔗 [Public Profile Link]({a.profile_url})")

                    col_edu, col_emp = st.columns(2)
                    with col_edu:
                        st.markdown("**🎓 Education Verification:**")
                        if a.education and a.education.institution:
                            st.caption(f"Institution: {a.education.institution}")
                        if a.education and a.education.degree:
                            st.caption(f"Degree: {a.education.degree}")
                        if a.education_evidence:
                            st.markdown(
                                f'<blockquote style="font-size:0.85rem;border-left:3px solid #10b981;'
                                f'padding-left:8px;margin:4px 0;">"{a.education_evidence.quote}"</blockquote>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f'<span style="font-size:0.75rem;">Source: <a href="{a.education_evidence.source_url}">'
                                f'{a.education_evidence.source_url}</a></span>',
                                unsafe_allow_html=True,
                            )

                    with col_emp:
                        st.markdown("**💼 Current Employment Verification:**")
                        st.caption(f"Employer: {a.current_employer or canonical}")
                        if a.employment_evidence:
                            st.markdown(
                                f'<blockquote style="font-size:0.85rem;border-left:3px solid #10b981;'
                                f'padding-left:8px;margin:4px 0;">"{a.employment_evidence.quote}"</blockquote>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f'<span style="font-size:0.75rem;">Source: <a href="{a.employment_evidence.source_url}">'
                                f'{a.employment_evidence.source_url}</a></span>',
                                unsafe_allow_html=True,
                            )

                    if a.education_evidence and a.education_evidence.date_checked:
                        st.caption(f"📅 Date checked: {a.education_evidence.date_checked}")

                    st.markdown("---")

    # Optional debug expander for rejected candidates (off by default)
    all_rejected = []
    for cr in report.company_results:
        for r in cr.rejected_candidates:
            all_rejected.append((cr.company_group.canonical_name, r))

    if all_rejected:
        with st.expander(f"🛠️ Rejected candidates (debug) — {len(all_rejected)} candidates", expanded=False):
            st.caption(
                "These individuals were evaluated by the pipeline but failed education or current employment "
                "verification criteria. They are **NOT** considered alumni."
            )
            for comp_name, rej in all_rejected:
                reasons_str = ", ".join(rej.reject_reasons) if rej.reject_reasons else "unverified"
                st.markdown(f"- **{rej.full_name}** ({comp_name}) — `Reasons: {reasons_str}`")
