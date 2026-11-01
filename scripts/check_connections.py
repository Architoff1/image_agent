"""Check local configuration, Groq API authentication, and Kaggle GPU health."""
from __future__ import annotations

import os
import sys
from pathlib import Path
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=False)

from agent.llm import model_name

print("=" * 60)
print(" Image Agent — System Health & Connection Diagnostic ")
print("=" * 60)

has_error = False

# 1. Groq API Check
print("\n[1/2] Checking Groq API Configuration...")
key = os.getenv("GROQ_API_KEY", "").strip()
model = model_name()
print(f"  Configured Model: {model}")

if not key or key.lower().startswith("replace_") or key.startswith("gsk_Not"):
    print("  Status: NOT CONFIGURED / PLACEHOLDER KEY")
    print("  Action: Get a free API key from https://console.groq.com/ and set GROQ_API_KEY in .env")
    has_error = True
else:
    try:
        from groq import Groq
        client = Groq(api_key=key, timeout=10.0)
        # Fast lightweight ping to verify authentication
        client.models.list()
        print("  Status: [OK] Groq API key is valid and authenticated.")
    except Exception as exc:
        err_msg = str(exc)
        if "401" in err_msg or "invalid_api_key" in err_msg.lower():
            print("  Status: [FAIL] Authentication failed (Invalid API Key).")
            print("  Action: Check your key at https://console.groq.com/keys and update .env.")
        else:
            print(f"  Status: [FAIL] Groq connection check error: {type(exc).__name__}: {err_msg[:120]}")
        has_error = True

# 2. Kaggle GPU Backend Check
print("\n[2/2] Checking Kaggle GPU Backend...")
base = os.getenv("GPU_BACKEND_URL", "").strip().rstrip("/")
if not base or "YOUR_" in base or "your-" in base.lower():
    print("  Status: NOT CONFIGURED")
    print("  Action: Start kaggle/flux-fill-gpu-backend.ipynb in Kaggle and paste the ngrok URL into .env.")
    has_error = True
else:
    if base.endswith("/v1/edit"):
        base = base[:-len("/v1/edit")].rstrip("/")
    token = os.getenv("GPU_BACKEND_TOKEN", "").strip()
    headers = {"X-Backend-Token": token} if token else {}
    try:
        r = requests.get(base + "/health", headers=headers, timeout=(6, 15))
        ngrok_err = r.headers.get("ngrok-error-code", "")
        if ngrok_err == "ERR_NGROK_3200" or (r.status_code == 404 and "ngrok" in r.text.lower()):
            print("  Status: [FAIL] Ngrok tunnel is offline / expired (ERR_NGROK_3200).")
            print("  Action: Run all cells in kaggle/flux-fill-gpu-backend.ipynb on Kaggle to generate a new active URL.")
            has_error = True
        elif r.status_code == 401:
            print("  Status: [FAIL] Authentication rejected by backend (HTTP 401).")
            print("  Action: Verify GPU_BACKEND_TOKEN in .env matches the GPU_BACKEND_TOKEN in Kaggle Secrets.")
            has_error = True
        elif not r.ok:
            print(f"  Status: [FAIL] Backend returned HTTP {r.status_code}")
            has_error = True
        else:
            info = r.json()
            print(f"  Status: [OK] Backend HTTP {r.status_code}")
            print(f"  Backend service:    {info.get('service', 'gpu-backend')}")
            print(f"  Active model:       {info.get('model', 'not reported')}")
            print(f"  Model loaded:       {info.get('model_loaded', 'not reported')}")
            print(f"  Automatic masking:  {info.get('automatic_masking', 'not reported')}")
    except requests.RequestException as exc:
        print(f"  Status: [FAIL] Kaggle backend unreachable ({type(exc).__name__}).")
        print("  Action: Ensure the Kaggle notebook session is actively running.")
        has_error = True

print("\n" + "=" * 60)
if has_error:
    print(" Diagnostic completed with warnings/actions required above.")
    sys.exit(1)
else:
    print(" All connections verified successfully! Ready to run.")
    sys.exit(0)
