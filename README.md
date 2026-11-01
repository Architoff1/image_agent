# Image Agent — Groq-controlled orchestration + Kaggle image editing

This package consolidates the local application and the supplied Part 1 Kaggle inference notebook. It removes Ollama/Mistral from the local agent and uses Groq for planning, tool calling, and bounded supervision.

## Architecture

```text
Streamlit UI (localhost:8501)
        |
        v
Local FastAPI + SQLite (localhost:8000)
        |
        v
LangGraph planner/router
        |
        +--> Groq API: plan request and choose registered tools
        |
        v
Bounded tool-calling controller
  1. inspect_current_image
  2. check_gpu_backend
  3. execute_image_edit (at most 2 attempts)
  4. verify_image_output
        |
        v
Kaggle HTTPS service
  SDXL inpainting + Canny ControlNet + configured LoRA
  Grounding DINO + SAM automatic mask (unless manual mask supplied)
```

Groq chooses among registered tools and can adjust only validated inference parameters. The Python executor owns file access, credentials, HTTP requests, allowed operations, resource limits, and retry limits. The model never receives raw filesystem paths or API secrets.

## What this version does

- Uses Groq; no Ollama/Mistral package or runtime is required.
- Corrects frontend routing: Streamlit talks to `LOCAL_AGENT_API_URL`, never directly to the Kaggle inference endpoint.
- Uses the exact form fields exposed by the supplied Kaggle notebook: `num_inference_steps`, `guidance_scale`, `controlnet_conditioning_scale`, `strength`, and `seed`.
- Gives Groq a bounded tool-calling loop over registered image inspection, backend health, edit execution, and output verification tools.
- Validates planner output and clamps inference parameters to backend-supported ranges.
- Limits tool rounds and image-edit attempts; verifies image integrity and output dimensions, and checks exact outside-mask preservation for manual masks.
- Retains conversation history, image versions, optional manual masks, automatic masking, and download/restore features.
- Includes the supplied Kaggle notebook with cell outputs removed, plus scripts for launching and checking connections.

## Important capability boundaries

- The Kaggle notebook's checkpoint is currently named `lora_step_0020.safetensors` and its own metadata labels it a **diagnostic checkpoint that has not been production-quality validated**. A successful HTTP response or smoke test does not establish photorealistic/semantic quality. We have not silently substituted a different checkpoint.
- The notebook's SDXL + ControlNet + LoRA inference is used as it exists in the supplied notebook. Do not retrain or restart the currently running Kaggle session just to use this local package.
- Automatic masking can choose the wrong object when several similar objects appear. Use a manual white-on-black mask when target selection is ambiguous. White = editable; black = preserve.
- Verification checks file integrity/dimensions and manual-mask preservation. It does **not** prove that the requested object looks semantically correct; visually inspect outputs.
- Reference images are stored in chat history but are not passed as conditioning to the current Kaggle endpoint.
- New-image generation is not wired because the current Kaggle service exposes masked editing, not a generation-only endpoint. The app returns a clear unsupported-capability message rather than pretending otherwise.
- The controller has bounded retries and cannot guarantee a perfect edit. If a retry does not help, it reports failure instead of claiming success.

## 1. Setup on Ubuntu

Extract this ZIP into a new folder. Keep your old project and its `data/` folder as a backup.

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip unzip
cd /path/to/Image_Agent_Automated
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .example.env .env
nano .env
```

Configure these values in `.env`:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TIMEOUT=120
GROQ_MAX_TOOL_ROUNDS=8
GROQ_MAX_EDIT_ATTEMPTS=2

GPU_BACKEND_URL=https://your-current-kaggle-ngrok-url
GPU_BACKEND_TOKEN=the-same-token-configured-in-kaggle-secrets
GPU_BACKEND_TIMEOUT=900

LOCAL_AGENT_API_URL=http://127.0.0.1:8000
API_REQUEST_TIMEOUT=1200
```

Choose a Groq model currently available to your account that supports tool calling. If the configured model has been retired or does not support tools, update `GROQ_MODEL` to an available supported model. Get the API key from the official Groq Console: https://console.groq.com/ . Never put secrets into source files or share them in chat.

`GPU_BACKEND_URL` is the **HTTPS base URL** printed by your running Kaggle notebook; do not append `/v1/edit`. `GPU_BACKEND_TOKEN` must match the Kaggle Secret. Credentials stay in the local API process and are not passed to Streamlit or Groq.

## 2. Verify configuration

With `.venv` active:

```bash
python scripts/check_connections.py
```

This checks whether Groq is configured and makes an authenticated request to the Kaggle `/health` endpoint. It does not run a generation/edit or spend inference compute.

## 3. Start the application

Terminal 1 — API:

```bash
source .venv/bin/activate
./scripts/run_api.sh
```

Check it in another terminal:

```bash
curl http://127.0.0.1:8000/health
```

Terminal 2 — UI:

```bash
source .venv/bin/activate
./scripts/run_ui.sh
```

Open the Streamlit URL printed by the command (usually `http://localhost:8501`). Keep the Kaggle notebook session and ngrok tunnel alive while editing.

## 4. End-to-end editing workflow

1. Run `scripts/check_connections.py` and ensure Kaggle health is successful.
2. Open Streamlit, start a chat, upload your source image.
3. Describe the change (e.g. add an accessory, modify color, replace an object).
4. The Groq agent orchestrates the plan, verifies the source, executes the edit via Kaggle SDXL/ControlNet, and verifies output dimensions and preservation.
5. If automatic masking selects an ambiguous region, upload a manual mask: white marks the editable area, black marks the preserved area.
6. If the edit fails, copy the redacted API response/error and the tool trace; never send `.env` or tokens.

## API endpoints

Local API: `/health`, `/v1/conversations`, `/v1/conversations/{id}`, `/v1/conversations/{id}/messages`, `/v1/conversations/{id}/versions/{version}`, `/v1/files/{filename}`.

Kaggle backend: authenticated `/health`, `/v1/mask`, and `/v1/edit`.

## Safe upgrade

This ZIP intentionally excludes `.env`, Streamlit secrets, SQLite DB, uploads, outputs, `.git`, and Python caches. Copy your existing `.env` and `data/` into this folder only if you want to retain settings and history; do not overwrite the new source files with old copies. Keep the previous project folder until the first end-to-end edit succeeds.

## Dynamic orchestration engine — Phases 1–4

The local API now includes a capability-backed orchestrator in `agent/orchestration.py`:

1. **Phase 1 — planning:** `POST /v1/agent/plan` asks the configured Groq model for a Pydantic-validated task DAG with explicit dependencies and acceptance criteria. Cycles, duplicate IDs, unknown dependencies, unregistered capabilities, and oversized plans are rejected.
2. **Phase 2 — execution/delegation:** `POST /v1/agent/runs` plans and executes dependency-ready tasks concurrently up to `AGENT_MAX_WORKERS`. Capability implementations are Python-registered handlers; the model cannot execute arbitrary code. State-mutating image edits are serialized to avoid racing the same image.
3. **Phase 3 — evaluation/retry:** a separate Groq evaluator compares actual task outputs against the goal and final criteria. Failed tasks are not marked successful. Retries are bounded by both per-task budgets and `AGENT_MAX_RETRIES`; a run that does not pass is returned as `needs_review`.
4. **Phase 4 — API/observability:** `GET /v1/agent/capabilities` reports the actual capability registry; run responses include the validated plan, per-task evidence, evaluation, trace, retry details, and final status.

Example requests:

```bash
curl -s http://127.0.0.1:8000/v1/agent/capabilities
curl -s -X POST http://127.0.0.1:8000/v1/agent/plan \
  -H 'Content-Type: application/json' \
  -d '{"goal":"Inspect the uploaded image and report its dimensions","conversation_id":"YOUR_CONVERSATION_ID"}'
curl -s -X POST http://127.0.0.1:8000/v1/agent/runs \
  -H 'Content-Type: application/json' \
  -d '{"goal":"Inspect the uploaded image and report its dimensions","conversation_id":"YOUR_CONVERSATION_ID"}'
```

Configure optional limits in `.env`: `AGENT_MAX_TASKS=12`, `AGENT_MAX_WORKERS=4`, `AGENT_MAX_RETRIES=2`. The available executable capabilities at this stage are intentionally honest and finite: `image.inspect`, `image.edit` (the existing real GPU-backed controller), and `conversation.respond`. The planner cannot make browser access, arbitrary filesystem editing, research, or general code-writing appear available; those require separately implemented and tested capability handlers. This is a real orchestration core, not a claim that every domain tool has already been built.

The Kaggle notebook is intentionally unchanged in this update, per the staged plan to implement and validate the local orchestration first.
