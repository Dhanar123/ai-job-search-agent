# Deployment Guide: Streamlit Community Cloud

This guide explains how to deploy the **AI Job Search Agent** to **Streamlit Community Cloud** (free hosting for Streamlit applications) and how to run a clean installation locally.

---

## 1. Prerequisites

1. A **GitHub account** with this repository pushed to your personal account or organization.
2. A free **Groq API Key** from [console.groq.com](https://console.groq.com/keys).
   *(Optional: An OpenAI API key from [platform.openai.com](https://platform.openai.com) as an alternative).*

---

## 2. Required Environment Variables & Secrets

Streamlit Community Cloud manages environment variables and sensitive keys through its **Secrets Management** console (`st.secrets`).

| Variable Name | Required | Default Value | Description |
| :--- | :---: | :---: | :--- |
| `GROQ_API_KEY` | **Yes** | *None* | Your Groq API key (starts with `gsk_...`) |
| `GROQ_MODEL` | No | `qwen/qwen3.8-27b` | High-limit LLM model used for resume extraction and qualification evaluation |
| `GROQ_BASE_URL` | No | `https://api.groq.com/openai/v1` | Groq OpenAI-compatible endpoint |
| `OPENAI_API_KEY` | No | *None* | Optional fallback if using OpenAI |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Optional model name for OpenAI |

---

## 3. Step-by-Step Deployment Instructions

### Step 1: Push Project to GitHub
Ensure all code and configuration files are committed and pushed to your GitHub repository:
```bash
git add .
git commit -m "Deploy: Prepare AI Job Search Agent for Streamlit Cloud"
git push origin main
```

### Step 2: Sign In to Streamlit Community Cloud
1. Navigate to [share.streamlit.io](https://share.streamlit.io).
2. Sign in with your GitHub account.

### Step 3: Create a New App
1. Click **"New app"** (or **"Create app"**).
2. Choose **"I already have an app"**.
3. Fill in the deployment details:
   - **Repository**: `your-github-username/ai-job-search-agent`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: Choose a custom subdomain (e.g. `my-ai-job-agent.streamlit.app`) or use default.

### Step 4: Configure Secrets
1. Before or immediately after launching, click **"Advanced settings"** (or navigate to **App settings ⚙️ → Secrets**).
2. In the Secrets editor, paste your configuration (see `.streamlit/secrets.toml.example`):
   ```toml
   GROQ_API_KEY = "gsk_your_actual_groq_api_key_here"
   GROQ_MODEL = "qwen/qwen3.8-27b"
   ```
3. Click **"Save"**.

### Step 5: Deploy & Verify
1. Click **"Deploy!"**.
2. Streamlit Cloud will install dependencies from `requirements.txt`, launch `app.py`, and render the user interface.
3. Once running, verify:
   - Sidebar displays **"⚡ Groq AI Active"**.
   - Resume upload extracts text and builds candidate profile.
   - Job search scans live public providers (Greenhouse, Lever, Ashby, RSS).
   - "Apply" button opens the official job board application link in a new browser tab without auto-submitting.

---

## 4. Local Clean Installation Verification

To test the application from a clean local environment:

```bash
# 1. Create a fresh virtual environment
python -m venv .venv

# 2. Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Check for dependency conflicts
python -m pip check

# 5. Configure environment variables
# Copy .env.example to .env and set your GROQ_API_KEY
cp .env.example .env

# 6. Run automated end-to-end test suite (all 9 scenarios)
python tests/test_step9_e2e.py

# 7. Start the local Streamlit application
streamlit run app.py
```
