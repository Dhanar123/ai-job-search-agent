"""AI Job Search Agent — Streamlit User Interface.

Step 8: Complete user interface connecting the full backend pipeline:
1. Resume PDF upload & PyMuPDF candidate profile extraction.
2. Job search criteria: roles, locations, salary, experience, excluded keywords.
3. Multi-source scanning (Greenhouse, Lever, Ashby, RSS).
4. Deterministic filtering & deduplication (Step 4).
5. Live description enrichment & structured data parsing (Step 5).
6. Objective AI qualification matching and scoring (Step 6).
7. HTTPS URL validation, provider status checking, and ranking (Step 7).
8. Interactive filtering, sorting, and direct application links.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(__file__))

from providers.base import JobPosting
from services.candidate_profile import (
    CandidateProfile,
    Experience,
    Preferences,
    build_candidate_profile,
    get_llm_client,
)
from services.description_extractor import EnrichedJob, enrich_jobs_with_descriptions
from services.filtering import (
    EligibilityStatus,
    FilterCriteria,
    FilteredJob,
    filter_jobs,
)
from services.matching import AIEligibility, MatchResult, rank_jobs
from services.resume_parser import ResumeParseError, parse_resume
from services.scanner import ProviderConfig, ScanConfig, scan_all_providers
from services.validation import (
    ListingStatus,
    ValidatedJobResult,
    ValidationDecision,
    ValidationReport,
    get_apply_url,
    get_job_url,
    validate_and_rank_results,
)

logger = logging.getLogger(__name__)

# ── Page Configuration ──────────────────────────────────────────────

st.set_page_config(
    page_title="AI Job Search Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Modern Styling (CSS) ────────────────────────────────────────────

CUSTOM_CSS = """
<style>
/* Global aesthetics */
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}

/* Custom Card Container */
.job-card {
    background-color: #1e222d;
    border: 1px solid #2d3345;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15);
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.job-card:hover {
    border-color: #4f5b78;
    transform: translateY(-2px);
}

/* Badges */
.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 14px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 6px;
    margin-bottom: 6px;
}
.badge-source {
    background-color: #2b3040;
    color: #a0aec0;
    border: 1px solid #3c4257;
}
.badge-loc {
    background-color: #1a365d;
    color: #90cdf4;
}
.badge-eligible {
    background-color: #1c4532;
    color: #9ae6b4;
    border: 1px solid #276749;
}
.badge-potential {
    background-color: #5c3d10;
    color: #fbd38d;
    border: 1px solid #744210;
}
.badge-ineligible {
    background-color: #63171b;
    color: #feb2b2;
    border: 1px solid #742a2a;
}
.badge-open {
    background-color: #1c4532;
    color: #68d391;
    border: 1px solid #2f855a;
}
.badge-unverified {
    background-color: #2d3748;
    color: #cbd5e0;
    border: 1px solid #4a5568;
}

/* Score Pills */
.score-pill-high {
    background: linear-gradient(135deg, #276749, #2f855a);
    color: #ffffff;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: bold;
    font-size: 1.1rem;
    display: inline-block;
}
.score-pill-mid {
    background: linear-gradient(135deg, #744210, #b7791f);
    color: #ffffff;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: bold;
    font-size: 1.1rem;
    display: inline-block;
}
.score-pill-low {
    background: linear-gradient(135deg, #4a5568, #2d3748);
    color: #e2e8f0;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: bold;
    font-size: 1.1rem;
    display: inline-block;
}

/* Skills tags */
.skill-tag-match {
    background-color: #132e22;
    color: #68d391;
    border: 1px solid #22543d;
    padding: 2px 8px;
    border-radius: 8px;
    font-size: 0.8rem;
    margin: 2px 4px 2px 0;
    display: inline-block;
}
.skill-tag-missing {
    background-color: #382410;
    color: #f6ad55;
    border: 1px solid #5a3915;
    padding: 2px 8px;
    border-radius: 8px;
    font-size: 0.8rem;
    margin: 2px 4px 2px 0;
    display: inline-block;
}
.uncertainty-box {
    background-color: #262c3d;
    border-left: 3px solid #718096;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 0.85rem;
    color: #cbd5e0;
    margin-top: 10px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Session State Management ────────────────────────────────────────

def init_session_state() -> None:
    """Initialize persistent session state variables."""
    if "resume_text" not in st.session_state:
        st.session_state.resume_text = None
    if "profile" not in st.session_state:
        st.session_state.profile = None
    if "validation_report" not in st.session_state:
        st.session_state.validation_report = None
    if "scan_errors" not in st.session_state:
        st.session_state.scan_errors = []
    if "pipeline_stats" not in st.session_state:
        st.session_state.pipeline_stats = {}
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = 0


init_session_state()


# ── Helper: Candidate Fallback Profile ──────────────────────────────

def create_manual_profile(
    roles: list[str],
    locations: list[str],
    salary_str: str,
    experience_level: str,
    experience_years: int,
) -> CandidateProfile:
    """Create a minimal CandidateProfile from manual inputs if no resume is parsed."""
    return CandidateProfile(
        name="Candidate",
        summary=f"Experienced {experience_level} professional seeking {', '.join(roles)}.",
        skills=roles,
        total_experience_years=experience_years,
        experience=[
            Experience(
                title=roles[0] if roles else "Software Engineer",
                company="Recent Company",
                highlights=[f"Hands-on experience in {', '.join(roles)}"],
            )
        ],
        preferences=Preferences(
            preferred_roles=roles,
            preferred_locations=locations,
            salary_expectation=salary_str,
            experience_level=experience_level,
        ),
    )


# ── Pipeline Orchestrator ───────────────────────────────────────────

async def execute_full_pipeline(
    profile: CandidateProfile,
    criteria: FilterCriteria,
    sources: list[ProviderConfig],
    model: str,
    max_eval_jobs: int = 15,
    status_callback: Any = None,
) -> tuple[ValidationReport, dict[str, Any], list[dict[str, str]]]:
    """Execute the end-to-end job discovery, filtering, matching, and validation pipeline."""
    stats: dict[str, Any] = {}

    def _notify(msg: str) -> None:
        if status_callback and callable(status_callback):
            try:
                status_callback(msg)
            except Exception:
                pass

    # Stage 1: Discovery across sources
    _notify(f"📡 1. Scanning {len(sources)} job boards concurrently…")
    scan_config = ScanConfig(sources=sources)
    discovered_jobs, scan_errors = await scan_all_providers(scan_config)
    stats["discovered"] = len(discovered_jobs)

    if not discovered_jobs:
        return ValidationReport(), stats, scan_errors

    # Stage 2: Filtering and deduplication
    _notify(f"🧹 2. Filtering {len(discovered_jobs)} raw postings by role, salary, and location…")
    filtered_jobs = filter_jobs(discovered_jobs, criteria)
    eligible_filtered = [j for j in filtered_jobs if j.is_eligible]
    stats["filtered"] = len(eligible_filtered)

    if not eligible_filtered:
        return ValidationReport(), stats, scan_errors

    # Limit to top candidates for description enrichment and LLM scoring
    shortlist = eligible_filtered[:max_eval_jobs]
    stats["shortlisted"] = len(shortlist)

    # Stage 3: Enriching full descriptions
    _notify(f"📄 3. Fetching descriptions for top {len(shortlist)} matching candidates…")
    enriched_jobs = await enrich_jobs_with_descriptions(shortlist, max_concurrency=6)
    stats["enriched"] = len(enriched_jobs)

    # Stage 4: AI Qualification Matching
    _notify(f"🤖 4. Evaluating qualifications with AI ({len(enriched_jobs)} jobs)…")
    scored_results = await rank_jobs(
        profile,
        enriched_jobs,
        max_concurrency=2,
        model=model,
    )
    stats["scored"] = len(scored_results)

    # Stage 5: URL validation, provider status verification & ranking
    _notify(f"✅ 5. Checking vacancy status and validating links…")
    report = await validate_and_rank_results(
        scored_results,
        max_concurrency=6,
    )
    stats["active_matches"] = len(report.active_results)
    stats["excluded"] = len(report.excluded_results)

    return report, stats, scan_errors


# ── Sidebar: AI & Resume Setup ──────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Setup & Sources")
    st.markdown("---")

    # 1. AI Provider Status (supports .env and Streamlit Community Cloud secrets)
    def _get_secret(key: str, default: str = "") -> str:
        val = os.getenv(key, "")
        if not val:
            try:
                # Safely attempt to read from st.secrets if available
                val = str(st.secrets.get(key, ""))
            except Exception:
                val = ""
        return val or default

    has_groq = bool(_get_secret("GROQ_API_KEY"))
    has_openai = bool(_get_secret("OPENAI_API_KEY"))
    selected_model = _get_secret("GROQ_MODEL", "qwen/qwen3.8-27b")

    if has_groq:
        st.success("⚡ **Groq AI Active**")
    elif has_openai:
        st.info("🟢 **OpenAI Active**")
        selected_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    else:
        st.warning("⚠️ No API Key in `.env`")
        groq_input = st.text_input("Enter Groq API Key:", type="password")
        if groq_input:
            os.environ["GROQ_API_KEY"] = groq_input.strip()
            st.rerun()

    st.markdown("---")

    # 2. Resume PDF Upload
    st.subheader("📄 Resume Upload")
    uploaded_file = st.file_uploader(
        "Upload PDF Resume",
        type=["pdf"],
        help="Extracts candidate qualifications, skills, and experience automatically.",
    )

    if uploaded_file is not None and st.session_state.profile is None:
        file_bytes = uploaded_file.getvalue()
        if len(file_bytes) > 0:
            with st.spinner("Parsing resume with PyMuPDF & AI…"):
                try:
                    resume_text = parse_resume(file_bytes, uploaded_file.name)
                    st.session_state.resume_text = resume_text
                    profile = build_candidate_profile(resume_text, model=selected_model)
                    st.session_state.profile = profile
                    st.toast("✅ Resume successfully parsed!", icon="🎉")
                except ResumeParseError as exc:
                    st.error(f"⚠️ **Resume Parsing Warning:** {exc}")
                    st.info("💡 **Tip:** If this is a scanned PDF or image, please upload a text-selectable PDF, or configure your search preferences manually in Tab 1.")
                except Exception as exc:
                    st.error(f"Failed to parse resume: {exc}")

    # Display resume status if parsed
    if st.session_state.profile is not None:
        p: CandidateProfile = st.session_state.profile
        st.success(f"👤 **{p.name or 'Candidate'}**")
        st.caption(f"💼 {p.total_experience_years or 0} yrs exp · 🛠️ {len(p.skills)} skills identified")
        with st.expander("View Candidate Summary", expanded=False):
            if p.summary:
                st.write(f"*{p.summary}*")
            if p.skills:
                st.write("**Skills:** " + ", ".join(p.skills[:10]))
            if st.button("Reset Resume", key="reset_resume_btn"):
                st.session_state.profile = None
                st.session_state.resume_text = None
                st.rerun()

    st.markdown("---")

    # 3. Job Board Sources Configuration
    st.subheader("🌐 Discovery Sources")
    include_gh = st.checkbox("Greenhouse", value=True, help="Boards like GitLab")
    gh_board = st.text_input("Greenhouse Board", value="gitlab", disabled=not include_gh)

    include_lever = st.checkbox("Lever", value=True, help="Boards like Spotify")
    lever_company = st.text_input("Lever Slug", value="spotify", disabled=not include_lever)

    include_ashby = st.checkbox("Ashby", value=True, help="Boards like Ashby")
    ashby_board = st.text_input("Ashby Board", value="Ashby", disabled=not include_ashby)

    include_rss = st.checkbox("RSS Job Feed", value=True)
    rss_feed = st.text_input(
        "RSS URL",
        value="https://weworkremotely.com/categories/remote-programming-jobs.rss",
        disabled=not include_rss,
    )


# Build list of active provider configs
active_sources: list[ProviderConfig] = []
if include_gh and gh_board.strip():
    active_sources.append(ProviderConfig(provider="greenhouse", identifier=gh_board.strip()))
if include_lever and lever_company.strip():
    active_sources.append(ProviderConfig(provider="lever", identifier=lever_company.strip()))
if include_ashby and ashby_board.strip():
    active_sources.append(ProviderConfig(provider="ashby", identifier=ashby_board.strip()))
if include_rss and rss_feed.strip():
    active_sources.append(ProviderConfig(provider="rss", identifier=rss_feed.strip()))


# ── Main Header ─────────────────────────────────────────────────────

st.title("🔍 AI Job Search Agent")
st.caption(
    "Automated qualification matching pipeline: Real multi-board discovery, "
    "strict URL verification, and evidence-based AI qualification evaluation."
)

# ── Navigation Tabs (Tab 1: Search Criteria, Tab 2: Matching Jobs) ───

tab_criteria, tab_results = st.tabs(["⚙️ Search Criteria & Filters", "🎯 Matching Jobs & Rankings"])


# ═════════════════════════════════════════════════════════════════════
# TAB 1: Search Criteria & Filters
# ═════════════════════════════════════════════════════════════════════

with tab_criteria:
    st.subheader("🎯 Configure Target Role & Preferences")
    st.markdown(
        "Customize target roles, location constraints, and salary expectations. "
        "Default values are pre-populated from your uploaded resume when available."
    )

    # Defaults pre-populated from parsed profile if present
    pref = st.session_state.profile.preferences if st.session_state.profile else None
    default_roles = ", ".join(pref.preferred_roles) if pref and pref.preferred_roles else "Software Engineer, Backend, Python"
    default_locations = ", ".join(pref.preferred_locations) if pref and pref.preferred_locations else "Remote, United States"
    default_exp_level = pref.experience_level.capitalize() if pref and pref.experience_level else "Senior"
    default_years = st.session_state.profile.total_experience_years if st.session_state.profile else 5

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        roles_input = st.text_input(
            "Job Roles & Keywords (comma-separated)",
            value=default_roles,
            help="Keywords to match against job titles and core requirements.",
        )
        locations_input = st.text_input(
            "Preferred Locations (comma-separated)",
            value=default_locations,
            help="Cities, countries, or regions.",
        )
        remote_only_input = st.checkbox("Remote Only", value=True)

    with col_r2:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            min_salary_input = st.number_input(
                "Minimum Salary (INR/yr)",
                min_value=0,
                max_value=10_000_000,
                value=800_000,
                step=50_000,
                help="Only filters when salary is stated. Postings with unstated salary are preserved (supports ₹, LPA, INR).",
            )
        with col_s2:
            exp_levels = ["Any", "Entry", "Mid", "Senior", "Lead"]
            default_idx = exp_levels.index(default_exp_level) if default_exp_level in exp_levels else 3
            exp_level_input = st.selectbox(
                "Experience Level",
                options=exp_levels,
                index=default_idx,
            )

        col_y1, col_y2 = st.columns(2)
        with col_y1:
            exp_years_input = st.number_input(
                "Years of Experience",
                min_value=0,
                max_value=40,
                value=default_years or 4,
            )
        with col_y2:
            max_eval_input = st.number_input(
                "Max Jobs to Score (AI)",
                min_value=3,
                max_value=30,
                value=10,
                help="Limits concurrent LLM matching requests to keep execution fast.",
            )

        excluded_keywords_input = st.text_input(
            "Excluded Keywords (comma-separated)",
            value="intern, internship, unpaid",
            help="Postings with these terms in title or description will be filtered out.",
        )

    st.markdown("---")

    # Search Execution Button
    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        search_pressed = st.button(
            "🔍 Search & Match Jobs",
            type="primary",
            use_container_width=True,
        )

    with col_info:
        if not st.session_state.profile:
            st.info("💡 *No resume uploaded yet. Search will use criteria to construct a candidate profile.*")
        else:
            st.success(f"⚡ Ready to match against candidate profile: **{st.session_state.profile.name}**")

    # ── Trigger Pipeline Execution ──────────────────────────────────
    if search_pressed:
        if not active_sources:
            st.error("Please enable at least one job source in the sidebar.")
        else:
            # Parse inputs into FilterCriteria
            parsed_roles = [r.strip() for r in roles_input.split(",") if r.strip()]
            parsed_locs = [l.strip() for l in locations_input.split(",") if l.strip()]
            parsed_excluded = [e.strip() for e in excluded_keywords_input.split(",") if e.strip()]

            criteria = FilterCriteria(
                preferred_roles=parsed_roles,
                keywords=parsed_roles,
                excluded_keywords=parsed_excluded,
                locations=parsed_locs,
                remote_only=remote_only_input,
                min_salary=float(min_salary_input) if min_salary_input > 0 else None,
                salary_currency="INR",
                experience_level="" if exp_level_input == "Any" else exp_level_input.lower(),
                max_experience_years=exp_years_input + 4,
                min_experience_years=max(0, exp_years_input - 2),
            )

            # Ensure candidate profile exists
            active_profile = st.session_state.profile
            if active_profile is None:
                active_profile = create_manual_profile(
                    roles=parsed_roles,
                    locations=parsed_locs,
                    salary_str=f"₹{min_salary_input:,.0f}/yr" if min_salary_input > 0 else "Competitive",
                    experience_level=exp_level_input,
                    experience_years=exp_years_input,
                )
                st.session_state.profile = active_profile

            # Run with live status indicator
            with st.status("🚀 Running End-to-End Job Search Pipeline…", expanded=True) as status_box:
                st.write(f"📡 1. Scanning **{len(active_sources)}** active job board providers concurrently…")

                try:
                    report, stats, scan_errors = asyncio.run(
                        execute_full_pipeline(
                            profile=active_profile,
                            criteria=criteria,
                            sources=active_sources,
                            model=selected_model,
                            max_eval_jobs=int(max_eval_input),
                            status_callback=status_box.write,
                        )
                    )

                    st.session_state.validation_report = report
                    st.session_state.pipeline_stats = stats
                    st.session_state.scan_errors = scan_errors

                    st.write(f"🧹 2. Discovered **{stats.get('discovered', 0)}** total postings across sources.")
                    st.write(f"🎯 3. Filtered down to **{stats.get('filtered', 0)}** matching postings.")
                    st.write(f"📝 4. Extracted descriptions & evaluated **{stats.get('scored', 0)}** postings with {selected_model}.")
                    st.write(f"✅ 5. Validated links and live vacancy status: **{stats.get('active_matches', 0)}** active matching jobs ready!")

                    status_box.update(label="🎉 Pipeline Complete! Check 'Matching Jobs' tab.", state="complete")
                    st.toast("Matching jobs ready in Tab 2!", icon="🎯")
                except Exception as exc:
                    logger.exception("Pipeline failed: %s", exc)
                    st.error(f"Pipeline error: {exc}")
                    status_box.update(label="❌ Pipeline execution failed", state="error")


# ═════════════════════════════════════════════════════════════════════
# TAB 2: Matching Jobs & Rankings
# ═════════════════════════════════════════════════════════════════════

with tab_results:
    report: ValidationReport | None = st.session_state.validation_report

    if report is None:
        st.info("👋 No search run yet. Configure your preferences and click **'🔍 Search & Match Jobs'** in Tab 1.")
    else:
        stats = st.session_state.pipeline_stats

        # ── Pipeline Metric Strip ───────────────────────────────────
        m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
        with m_c1:
            st.metric("Discovered", stats.get("discovered", 0), help="Total raw postings fetched from providers")
        with m_c2:
            st.metric("Filtered", stats.get("filtered", 0), help="Postings matching roles, locations, and salary")
        with m_c3:
            st.metric("Scored by AI", stats.get("scored", 0), help="Postings evaluated for qualification alignment")
        with m_c4:
            st.metric("Active Matches", len(report.active_results), help="Validated open or unverified job matches")
        with m_c5:
            st.metric("Excluded", len(report.excluded_results), help="Closed/404 postings, duplicate listings, or invalid links")

        st.markdown("---")

        # ── Filter & Sorting Toolbar ────────────────────────────────
        st.subheader("🔍 Refine & Sort Results")
        f_c1, f_c2, f_c3, f_c4 = st.columns(4)

        with f_c1:
            eligibility_filter = st.selectbox(
                "Filter by Eligibility",
                options=["All", "Eligible", "Potentially Eligible", "Ineligible"],
                index=0,
            )

        with f_c2:
            status_filter = st.selectbox(
                "Filter by Vacancy Status",
                options=["All", "Confirmed Open", "Unverified"],
                index=0,
            )

        with f_c3:
            min_score_filter = st.slider(
                "Minimum Match Score",
                min_value=0,
                max_value=100,
                value=0,
                step=5,
            )

        with f_c4:
            sort_by = st.selectbox(
                "Sort Results By",
                options=[
                    "Match Score: High to Low",
                    "Match Score: Low to High",
                    "Eligibility Priority",
                    "Company (A to Z)",
                ],
                index=0,
            )

        # ── Filter Application ──────────────────────────────────────
        filtered_results: list[ValidatedJobResult] = []
        for r in report.active_results:
            # Score filter
            if r.score < min_score_filter:
                continue

            # Eligibility filter
            r_elig = str(r.eligibility).lower().strip()
            if eligibility_filter == "Eligible" and r_elig != "eligible":
                continue
            if eligibility_filter == "Potentially Eligible" and r_elig != "potentially_eligible":
                continue
            if eligibility_filter == "Ineligible" and r_elig != "ineligible":
                continue

            # Status filter
            if status_filter == "Confirmed Open" and r.listing_status != ListingStatus.OPEN:
                continue
            if status_filter == "Unverified" and r.listing_status != ListingStatus.UNVERIFIED:
                continue

            filtered_results.append(r)

        # ── Sorting Application ─────────────────────────────────────
        if sort_by == "Match Score: High to Low":
            filtered_results.sort(key=lambda x: x.score, reverse=True)
        elif sort_by == "Match Score: Low to High":
            filtered_results.sort(key=lambda x: x.score, reverse=False)
        elif sort_by == "Eligibility Priority":
            from services.validation import _eligibility_sort_priority
            filtered_results.sort(key=lambda x: (_eligibility_sort_priority(x.eligibility), -x.score))
        elif sort_by == "Company (A to Z)":
            filtered_results.sort(key=lambda x: str(x.job.get("company", "")).lower())

        st.markdown(f"**Showing {len(filtered_results)} of {len(report.active_results)} active job matches**")

        # ── Matching Job Cards ──────────────────────────────────────
        if not report.active_results:
            st.info("ℹ️ **No active job listings matched your search criteria across the selected job sources.**")
            st.markdown(
                """
                **Suggestions to find more jobs:**
                - Broaden your **Job Roles & Keywords** (e.g., try `Engineer` or `Developer`).
                - Turn off the **Remote Only** toggle to include on-site or hybrid opportunities.
                - Ensure at least one active job board source is checked in the sidebar.
                - Adjust or lower your **Minimum Salary** filter.
                """
            )
        elif not filtered_results:
            st.warning("⚠️ No job matches meet the current toolbar filter criteria. Try lowering the minimum match score or changing eligibility filters to 'All'.")
        else:
            for idx, res in enumerate(filtered_results):
                job = res.job
                title = job.get("title", "Untitled Role")
                company = job.get("company", "Company")
                location = job.get("location") or "Unspecified Location"
                source = job.get("source", "web").upper()
                apply_link = get_apply_url(job) or get_job_url(job)
                score = res.score
                eligibility_str = str(res.eligibility).lower()

                # Card HTML layout
                st.markdown('<div class="job-card">', unsafe_allow_html=True)

                c_top1, c_top2 = st.columns([3, 1])
                with c_top1:
                    st.markdown(f"### {title}")
                    st.markdown(
                        f"**🏢 {company}** &nbsp;·&nbsp; "
                        f'<span class="badge badge-loc">📍 {location}</span> '
                        f'<span class="badge badge-source">🌐 {source}</span>',
                        unsafe_allow_html=True,
                    )

                with c_top2:
                    # Match score pill
                    pill_class = "score-pill-high" if score >= 75 else ("score-pill-mid" if score >= 45 else "score-pill-low")
                    st.markdown(
                        f'<div style="text-align:right;">'
                        f'<span class="{pill_class}">Match Score: {score:.0f}/100</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption("Alignment evidence (not hiring probability)")

                # Badges row: Eligibility & Vacancy Status
                b_col1, b_col2 = st.columns([2, 1])
                with b_col1:
                    elig_badge = (
                        '<span class="badge badge-eligible">🟢 Eligible</span>'
                        if eligibility_str == "eligible"
                        else (
                            '<span class="badge badge-potential">🟡 Potentially Eligible</span>'
                            if eligibility_str == "potentially_eligible"
                            else '<span class="badge badge-ineligible">🔴 Ineligible</span>'
                        )
                    )

                    status_badge = (
                        '<span class="badge badge-open">✓ Verified Open</span>'
                        if res.listing_status == ListingStatus.OPEN
                        else '<span class="badge badge-unverified">? Unverified Vacancy</span>'
                    )

                    st.markdown(f"{elig_badge} {status_badge}", unsafe_allow_html=True)

                with b_col2:
                    # Clickable Apply button (opens in new browser tab, never auto-submits)
                    if apply_link and apply_link.startswith("https://"):
                        st.link_button(f"Apply on {company} ↗", url=apply_link, type="primary", use_container_width=True)
                        st.caption("🔒 *Opens original listing in new tab. Never auto-submits.*")
                    else:
                        st.caption("No HTTPS apply link")

                # Expandable Job Description Review
                desc_text = job.get("description") or job.get("description_snippet") or ""
                raw_status = job.get("description_status") or job.get("retrieval_status", "unavailable")
                retrieval_status = raw_status.value if hasattr(raw_status, "value") else str(raw_status)
                ret_label = "✅ Full Description" if retrieval_status == "complete" else ("⚠️ Partial Description" if retrieval_status == "partial" else "ℹ️ Summary Only")
                with st.expander(f"📄 View Job Description ({ret_label})", expanded=False):
                    if desc_text and desc_text.strip():
                        st.markdown(desc_text[:4000] + ("\n\n*…[truncated for display]*" if len(desc_text) > 4000 else ""))
                    else:
                        st.info("Full description text was unavailable from the provider endpoint. Evaluation used title and metadata.")

                # Skills overlap section
                match_obj = res.match
                matching_skills = match_obj.matching_skills if match_obj else []
                missing_skills = match_obj.missing_or_unverified_skills if match_obj else []

                col_sk1, col_sk2 = st.columns(2)
                with col_sk1:
                    st.markdown("**✅ Matching Skills / Qualifications:**")
                    if matching_skills:
                        tags_html = " ".join(f'<span class="skill-tag-match">✓ {s}</span>' for s in matching_skills)
                        st.markdown(tags_html, unsafe_allow_html=True)
                    else:
                        st.caption("None explicitly evidenced")

                with col_sk2:
                    st.markdown("**⚠️ Unverified / Missing Qualifications:**")
                    if missing_skills:
                        tags_html = " ".join(f'<span class="skill-tag-missing">? {s}</span>' for s in missing_skills)
                        st.markdown(tags_html, unsafe_allow_html=True)
                    else:
                        st.caption("All stated requirements evidenced")

                # AI Match Explanation
                if match_obj and match_obj.explanation:
                    st.markdown(f"**💡 Match Rationale:** *{match_obj.explanation}*")

                # Uncertainty notes
                if res.uncertainty_notes:
                    notes_str = "<br>• ".join(res.uncertainty_notes)
                    st.markdown(
                        f'<div class="uncertainty-box">'
                        f'<strong>ℹ️ Uncertainty Notes:</strong><br>• {notes_str}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("</div>", unsafe_allow_html=True)

        # ── Excluded Postings Expander ──────────────────────────────
        if report.excluded_results:
            with st.expander(f"🚫 View {len(report.excluded_results)} Excluded Postings (Expired, Duplicate, or Invalid Links)", expanded=False):
                st.caption("These listings were excluded by the Step 7 verification engine to protect data accuracy.")
                for ex in report.excluded_results:
                    ex_title = ex.job.get("title", "Job")
                    ex_company = ex.job.get("company", "Company")
                    ex_reason = ex.exclusion_reason or ex.status_reason or "Excluded"
                    ex_url = get_job_url(ex.job)
                    st.markdown(f"- **{ex_title}** @ {ex_company} — `{ex_reason}` (`{ex_url}`)")

        # ── Provider Scan Diagnostic Errors ─────────────────────────
        if st.session_state.scan_errors:
            with st.expander("⚠️ Provider Diagnostic Warnings", expanded=False):
                for err in st.session_state.scan_errors:
                    st.warning(f"**Provider {err.get('provider')} [{err.get('identifier')}]:** {err.get('error')}")
