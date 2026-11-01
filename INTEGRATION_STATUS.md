# Integration status — Groq-controlled Image Agent

## Implemented in this source package

- [x] Groq SDK replaces local Ollama/Mistral integration.
- [x] Groq JSON planner validates against a Pydantic schema (FLUX parameter defaults).
- [x] Groq tool-calling controller exposes a small registry of validated tools.
- [x] Controller checks source image and authenticated Kaggle health before editing.
- [x] Controller executes the real `/v1/edit` endpoint and may retry once within configured limits.
- [x] Local client sends the parameter names required by the Kaggle notebook `/v1/edit` contract.
- [x] Output integrity/dimension checks; exact outside-mask check when a manual mask is used.
- [x] Streamlit routes to local FastAPI, not to Kaggle directly.
- [x] Conversation/version storage preserved; tool trace is stored with the assistant plan.
- [x] Full operational pipelines: image analyzer, dual-engine web search, bounded edit controller.

## Kaggle GPU backend — `kaggle/flux-fill-gpu-backend.ipynb`

- [x] **Auto model selection**: tries **FLUX.1-Fill-dev** first (state-of-the-art, gated). On 403/GatedRepoError, automatically falls back to **SDXL Inpainting 1.0** (`diffusers/stable-diffusion-xl-1.0-inpainting-0.1` — MIT license, no HF token, loads directly in diffusers format).
- [x] **Grounding DINO + SAM** for zero-shot automatic masking on both models.
- [x] Identical REST API regardless of which model loads (`GET /health`, `POST /v1/mask`, `POST /v1/edit`).
- [x] `X-Model` response header reports the active model name.
- [x] `check_connections.py` reads and prints the active model name from `/health`.
- [x] Smoke test cell verifies the pipeline before the API is exposed.

## To unlock FLUX.1-Fill-dev (optional — highest quality)

1. Visit https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev and click **"Agree and access repository"**.
2. Get a read token at https://huggingface.co/settings/tokens.
3. Add `HF_TOKEN` as a Kaggle Secret. The notebook will automatically pick it up on next run.

## Not claimed as verified

- [ ] Live Groq API request: cannot be verified without the user's key and a network-enabled runtime.
- [ ] Live Kaggle HTTP request: cannot be verified from this build environment.
- [ ] End-to-end edit against the running GPU session: not run here.
- [ ] Semantic image quality/photorealism: requires visual acceptance testing.
