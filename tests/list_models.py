import os
import httpx
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GROQ_API_KEY")
resp = httpx.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=25.0)
if resp.status_code == 200:
    models = resp.json().get("data", [])
    models_sorted = sorted(models, key=lambda x: x.get("context_window", 0), reverse=True)
    for m in models_sorted:
        print(f"ID: {m.get('id')} | context_window: {m.get('context_window')}")
else:
    print(f"Error {resp.status_code}: {resp.text}")
