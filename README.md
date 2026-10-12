# 🔍 AI Job Search Agent

An autonomous, end-to-end job search and matching application built with Python and Streamlit. The agent ingests candidate resumes, discovers live postings across multiple public ATS providers (Greenhouse, Lever, Ashby, and RSS), performs deterministic rule filtering, extracts full job descriptions, evaluates qualification alignment with an LLM, verifies HTTPS link validity and vacancy status, and presents verified jobs with transparent rationales.

---

## 🌟 Key Features

1. **Multi-Source Discovery (Step 1–3)**:
   - Greenhouse public job boards (e.g., GitLab).
   - Lever public postings (e.g., Spotify).
   - Ashby job board endpoints (e.g., Ashby).
   - RSS job feeds (e.g., We Work Remotely).

2. **Resume PDF Parsing & Candidate Profile (Step 2)**:
   - Multi-strategy PyMuPDF text extraction.
   - LLM structured candidate profile extraction (skills, experience, preferences).
   - Friendly guidance for scanned / image-only documents.

3. **Deterministic Filtering & Deduplication (Step 4)**:
   - Title and keyword matching.
   - Exclusion keywords.
   - Location and remote-only constraints.
   - Multi-currency salary filtering (INR / LPA, Lakhs, USD, EUR) with automatic conversion.
   - Canonical URL and provider ID cross-source deduplication.

4. **Description Extraction & Structured Data (Step 5)**:
   - Full description retrieval from provider APIs or public web pages (JSON-LD and HTML).
   - Retrieval status tracking: `complete`, `partial`, `unavailable`.
   - Anti-fabrication guarantees: original job and application URLs are preserved.

5. **Evidence-Based AI Matching & Qualification Scoring (Step 6)**:
   - Objective match score (0 to 100) based strictly on evidenced qualifications.
   - Matching skills vs. unverified/missing qualifications breakdown.
   - Separate rule-based eligibility vs. AI match score.
   - Single high-limit LLM model (`qwen/qwen3.8-27b`) with automatic retry and rate-limit backoff.

6. **Link Validation & Vacancy Verification (Step 7)**:
   - Strict HTTPS validation.
   - Live provider vacancy checking (Greenhouse, Lever, Ashby).
   - Anti-assumption rule: URL responsiveness alone never claims a job is open.
   - Exclusion of confirmed expired or closed listings (404/410).

7. **Reviewable Pre-Application UI (Step 8 & 9)**:
   - Expandable **"View Job Description"** in each job card.
   - Match rationale and transparent uncertainty notes.
   - Direct clickable **"Apply on [Company] ↗"** link opening the original posting in a new browser tab.
   - Zero automated application submissions.

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone <repo-url>
cd ai-job-search-agent
python -m venv .venv
# Activate virtual environment
# Windows: .venv\Scripts\activate | Unix: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file from `.env.example`:
```ini
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=qwen/qwen3.8-27b
```

### 3. Run Locally
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Testing

The test suite covers the entire pipeline and 9 critical end-to-end failure/edge scenarios:

```bash
# Run the complete Step 9 End-to-End Suite (9 scenarios)
python tests/test_step9_e2e.py

# Run unit and integration test suites
python tests/test_filtering.py
python tests/test_validation.py
python tests/test_ui_pipeline.py
```

---

## ☁️ Deployment

For deployment to **Streamlit Community Cloud**, see [DEPLOYMENT.md](DEPLOYMENT.md).
