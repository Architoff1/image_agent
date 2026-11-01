# Urgent sunglasses edit fix

## What changed
- The Kaggle notebook now uses base SDXL inpainting + Canny ControlNet by default. The supplied LoRA was trained on unrelated object categories and was not validated for sunglasses, so it is disabled by default.
- Backend defaults for the edit endpoint are 35 steps, guidance 7.0, Canny ControlNet scale 0.35, strength 0.9.
- Streamlit includes an optional portrait-specific sunglasses mask preset for the exact 413×531 sample portrait. It is not a general face detector; upload a manually created mask for other images.
- The local Groq controller remains Groq-only; no Ollama/Mistral dependency is introduced.

## Required steps
1. In Kaggle, open `kaggle/model-set-up-server-finally.ipynb`, attach the same deployment-artifacts dataset and enable a GPU.
2. Ensure Kaggle Secrets contain `HF_TOKEN`, `NGROK_AUTHTOKEN`, and `GPU_BACKEND_TOKEN`.
3. Run notebook cells from the top in order. Wait for the final cell to print the new HTTPS URL. Keep the notebook session running.
4. On the local machine, copy `.example.env` to `.env`, set `GROQ_API_KEY`, `GPU_BACKEND_URL` to the printed HTTPS base URL (with or without `/v1/edit`), and `GPU_BACKEND_TOKEN` to the exact same token as Kaggle.
5. Install `python3 -m pip install -r requirements.txt` inside your venv. Start API with `bash scripts/run_api.sh` and UI with `bash scripts/run_ui.sh`.
6. Upload the original 413×531 portrait. Expand attachment options and tick **Use sunglasses mask preset for the supplied portrait**. Prompt: `Add realistic black sunglasses across both eyes, connected at the nose bridge, photorealistic, preserve identity, hair, beard, clothing, background and all other details.`
7. Send the request. If the image differs in dimensions from the sample portrait, do not use the preset; upload a matching mask instead.

## Honest status
Local code and notebook changes can be syntax-checked here, but the full remote inference cannot be claimed verified without running the updated notebook in your Kaggle account, a live ngrok URL, valid Kaggle secrets, and a valid Groq key. Base SDXL is substantially more appropriate for this accessory task than the unvalidated adapter, but diffusion output is not guaranteed on every seed.


## Pure Diffusion Pipeline
All accessory edits (such as sunglasses) are rendered purely by the diffusion inpainting model (FLUX.1-Fill or SDXL Inpainting) without any artificial 2D OpenCV or PIL vector overlays.
