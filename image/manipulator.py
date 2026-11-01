"""Client for the authenticated Kaggle GPU image-editing backend."""
import os
import requests
from pathlib import Path
from agent.state import AgentState


def _build_inpaint_prompt(task: dict) -> str:
    operation = task.get("operation", "modify")
    description = (task.get("description") or "Edit the image as requested.").strip()
    target = (task.get("target") or "object").strip()
    full_text = f"{operation} {target} {description}".lower()

    # Specialized high-signal diffusion prompts for common accessories
    if any(k in full_text for k in ("sunglass", "glasses", "shades", "spectacles")):
        return "a person wearing stylish black sunglasses with dark polarized lenses and black frames, sharp focus, natural lighting, photorealistic"

    if "helmet" in full_text:
        return "a person wearing a realistic black motorcycle helmet with a closed tinted visor, high detail, sharp focus, natural lighting, photorealistic"

    if any(k in full_text for k in ("hat", "cap", "beanie")):
        return f"a person wearing a stylish hat, highly detailed, photorealistic, sharp focus, natural lighting"

    if operation == "remove":
        instruction = f"Remove the {target} from the image and naturally reconstruct the background where it was."
    elif operation == "add":
        instruction = f"Add {description or target} naturally to the image."
    elif operation == "replace":
        instruction = f"Replace the {target} with {description}."
    else:
        instruction = f"Modify the {target}: {description}."
    parts = [instruction]
    constraints = task.get("constraints") or []
    if constraints:
        parts.append("Preserve: " + "; ".join(str(x) for x in constraints))
    parts.append("Photorealistic, natural lighting, sharp focus, high quality.")
    return " ".join(parts)


def execute_image_edit(state: AgentState, options: dict | None = None):
    task = dict(state.get("current_task") or {})
    if options:
        task.update(options)
    image_path = state.get("current_image")
    mask_path = state.get("mask_image")

    if not image_path or not Path(image_path).is_file():
        return {
            "response": "Please upload an image before requesting an edit.",
            "error": "missing_image",
        }
    operation = task.get("operation")
    if operation not in {"add", "remove", "modify", "replace"}:
        return {
            "response": f"Unsupported image-edit operation: {operation}",
            "error": "unsupported_operation",
        }

    has_manual_mask = bool(mask_path and Path(mask_path).is_file())
    base_url = os.getenv("GPU_BACKEND_URL", "").strip().rstrip("/")
    if not base_url or "YOUR_" in base_url or "your-" in base_url.lower():
        return {
            "response": "GPU_BACKEND_URL is not configured with the active Kaggle HTTPS base URL. Set it in .env.",
            "error": "missing_gpu_backend_url",
        }
    if base_url.endswith("/v1/edit"):
        base_url = base_url[:-len("/v1/edit")].rstrip("/")
    endpoint = f"{base_url}/v1/edit"
    backend_token = os.getenv("GPU_BACKEND_TOKEN", "").strip()
    if base_url.startswith("https://") and (not backend_token or backend_token.lower().startswith("the_same_value") or backend_token.lower().startswith("replace_")):
        return {
            "response": "GPU_BACKEND_TOKEN is missing from the local environment. Configure the same token used by Kaggle Secrets.",
            "error": "missing_gpu_backend_token",
        }

    headers = {"X-Backend-Token": backend_token} if backend_token else {}
    full_prompt = _build_inpaint_prompt(task)
    timeout = max(30, min(int(os.getenv("GPU_BACKEND_TIMEOUT", "900")), 1800))

    # Targeted mask generation for accessories when no manual mask is provided:
    # Restricts inpainting strictly to the required zone, completely protecting
    # unaffected areas (e.g., face, beard, clothing) outside the mask.
    generated_mask_bytes = None
    desc_text = (str(task.get("description", "")) + " " + str(task.get("target", "")) + " " + str(task.get("prompt", ""))).lower()
    is_glasses = any(k in desc_text for k in ("sunglass", "glasses", "shades", "spectacles"))
    is_helmet = "helmet" in desc_text

    if not has_manual_mask:
        if is_glasses:
            try:
                import io
                from PIL import Image, ImageDraw
                with Image.open(image_path) as src_im:
                    W, H = src_im.size
                eye_mask = Image.new("L", (W, H), 0)
                d = ImageDraw.Draw(eye_mask)
                x1, x2 = int(W * 0.24), int(W * 0.76)
                y1, y2 = int(H * 0.29), int(H * 0.47)
                d.rounded_rectangle((x1, y1, x2, y2), radius=max(4, int(H * 0.03)), fill=255)
                buf = io.BytesIO()
                eye_mask.save(buf, format="PNG")
                generated_mask_bytes = buf.getvalue()
            except Exception:
                generated_mask_bytes = None
        elif is_helmet:
            try:
                import io
                from PIL import Image, ImageDraw
                with Image.open(image_path) as src_im:
                    W, H = src_im.size
                helmet_mask = Image.new("L", (W, H), 0)
                d = ImageDraw.Draw(helmet_mask)
                x1, x2 = int(W * 0.17), int(W * 0.83)
                y1, y2 = 0, int(H * 0.58)
                d.rounded_rectangle((x1, y1, x2, y2), radius=max(8, int(H * 0.06)), fill=255)
                buf = io.BytesIO()
                helmet_mask.save(buf, format="PNG")
                generated_mask_bytes = buf.getvalue()
            except Exception:
                generated_mask_bytes = None

    import time
    from agent.telemetry import telemetry

    start_time = time.perf_counter()
    conv_id = state.get("conversation_id")
    request_meta = {
        "operation": operation,
        "target": task.get("target") or "object",
        "prompt": full_prompt,
        "num_inference_steps": max(1, min(int(task.get("num_inference_steps", 28)), 50)),
        "guidance_scale": max(1.0, min(float(task.get("guidance_scale", 30.0)), 50.0)),
        "controlnet_conditioning_scale": max(0.0, min(float(task.get("controlnet_conditioning_scale", 0.0)), 2.0)),
        "strength": 0.98 if (is_glasses or is_helmet) else max(0.0, min(float(task.get("strength", 1.0)), 1.0)),
        "seed": max(0, min(int(task.get("seed", 42)), 2147483647)),
        "has_manual_mask": has_manual_mask,
        "preset_mask_generated": bool(generated_mask_bytes),
        "source_image_name": Path(image_path).name,
    }

    try:
        with open(image_path, "rb") as image_file:
            files = {"image": (Path(image_path).name, image_file, "application/octet-stream")}
            mask_file = open(mask_path, "rb") if has_manual_mask else None
            try:
                if mask_file is not None:
                    files["mask"] = (Path(mask_path).name, mask_file, "application/octet-stream")
                elif generated_mask_bytes is not None:
                    mask_name = "helmet_mask.png" if is_helmet else "sunglasses_eye_band_mask.png"
                    files["mask"] = (mask_name, generated_mask_bytes, "image/png")

                use_auto_mask = "false" if (has_manual_mask or generated_mask_bytes is not None) else "true"
                request_meta["auto_mask"] = use_auto_mask

                response = requests.post(
                    endpoint,
                    headers=headers,
                    files=files,
                    data={
                        "prompt": full_prompt,
                        "operation": operation,
                        "target": task.get("target") or "object",
                        "auto_mask": use_auto_mask,
                        # These field names match the actual Kaggle notebook's /v1/edit contract.
                        "num_inference_steps": str(request_meta["num_inference_steps"]),
                        "guidance_scale": str(request_meta["guidance_scale"]),
                        "controlnet_conditioning_scale": str(request_meta["controlnet_conditioning_scale"]),
                        "strength": str(request_meta["strength"]),
                        "seed": str(request_meta["seed"]),
                    },
                    timeout=(20, timeout),
                )
            finally:
                if mask_file is not None:
                    mask_file.close()

        duration = time.perf_counter() - start_time
        response_headers = {k: v for k, v in response.headers.items()} if hasattr(response, "headers") else {}

        if not response.ok:
            try:
                body = response.json()
                detail = body.get("detail") or body.get("error") or response.text
            except (ValueError, AttributeError):
                detail = response.text
            detail = str(detail).strip()[:700] or "No error details returned."
            err_code = f"gpu_http_{response.status_code}"
            telemetry.log_kaggle_call(
                conversation_id=conv_id,
                endpoint=endpoint,
                request_meta=request_meta,
                response_meta={"headers": response_headers, "detail": detail},
                duration_s=duration,
                status_code=response.status_code,
                error=f"HTTP {response.status_code}: {detail}",
            )
            return {
                "response": f"GPU backend returned HTTP {response.status_code}: {detail}",
                "error": err_code,
                "kaggle_telemetry": {
                    "endpoint": endpoint,
                    "duration_seconds": round(duration, 3),
                    "status_code": response.status_code,
                    "request": request_meta,
                    "error": detail,
                },
            }

        content_type = response.headers.get("content-type", "").lower()
        if not content_type.startswith("image/"):
            detail = response.text[:500] if hasattr(response, "text") else "No response body."
            err_code = "gpu_unexpected_content_type"
            telemetry.log_kaggle_call(
                conversation_id=conv_id,
                endpoint=endpoint,
                request_meta=request_meta,
                response_meta={"headers": response_headers, "body": detail},
                duration_s=duration,
                status_code=response.status_code,
                error=f"Unexpected content-type: {content_type}",
            )
            return {
                "response": f"GPU backend returned an unexpected response type ({content_type or 'unknown'}). {detail}",
                "error": err_code,
                "kaggle_telemetry": {
                    "endpoint": endpoint,
                    "duration_seconds": round(duration, 3),
                    "status_code": response.status_code,
                    "content_type": content_type,
                    "error": detail,
                },
            }

        project_root = Path(__file__).resolve().parents[1]
        configured_output = Path(os.getenv("OUTPUT_DIR", str(project_root / "data" / "outputs"))).expanduser()
        output_dir = configured_output if configured_output.is_absolute() else (project_root / configured_output).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"edited_{os.urandom(8).hex()}.png"
        output_file.write_bytes(response.content)
        history = list(state.get("image_history", [])) + [str(image_path), str(output_file)]
        mask_mode = response.headers.get("X-Mask-Mode", "manual" if has_manual_mask else "automatic")
        model_used = response.headers.get("X-Model", "FLUX.1-Fill-dev")
        mask_target = response.headers.get("X-Mask-Target", task.get("target") or "object")
        resp_seed = response.headers.get("X-Seed", str(request_meta["seed"]))

        kaggle_telemetry = {
            "endpoint": endpoint,
            "duration_seconds": round(duration, 3),
            "status_code": response.status_code,
            "model_used": model_used,
            "mask_mode": mask_mode,
            "mask_target": mask_target,
            "seed": resp_seed,
            "request_parameters": request_meta,
            "output_size_bytes": len(response.content),
            "output_file": output_file.name,
        }

        telemetry.log_kaggle_call(
            conversation_id=conv_id,
            endpoint=endpoint,
            request_meta=request_meta,
            response_meta=kaggle_telemetry,
            duration_s=duration,
            status_code=response.status_code,
        )

        return {
            "current_image": str(output_file),
            "image_history": history,
            "response": f"Image edit completed successfully in {round(duration, 1)}s (model: {model_used}, mask mode: {mask_mode}).",
            "error": None,
            "kaggle_telemetry": kaggle_telemetry,
        }

    except requests.Timeout:
        duration = time.perf_counter() - start_time
        err_msg = f"The GPU edit timed out after {timeout} seconds. Check that the Kaggle notebook and ngrok tunnel are still running; a cold model startup may take longer."
        telemetry.log_kaggle_call(
            conversation_id=conv_id,
            endpoint=endpoint,
            request_meta=request_meta,
            response_meta={},
            duration_s=duration,
            status_code=None,
            error="Timeout",
        )
        return {
            "response": err_msg,
            "error": "gpu_timeout",
            "kaggle_telemetry": {
                "endpoint": endpoint,
                "duration_seconds": round(duration, 3),
                "error": "Request timed out",
                "timeout_limit": timeout,
            },
        }
    except requests.RequestException as exc:
        duration = time.perf_counter() - start_time
        err_msg = f"Could not reach the GPU backend. Check GPU_BACKEND_URL and confirm the Kaggle service and HTTPS tunnel are running. Details: {exc}"
        telemetry.log_kaggle_call(
            conversation_id=conv_id,
            endpoint=endpoint,
            request_meta=request_meta,
            response_meta={},
            duration_s=duration,
            status_code=None,
            error=str(exc),
        )
        return {
            "response": err_msg,
            "error": "gpu_connection_error",
            "kaggle_telemetry": {
                "endpoint": endpoint,
                "duration_seconds": round(duration, 3),
                "error": str(exc),
            },
        }
    except OSError as exc:
        duration = time.perf_counter() - start_time
        err_msg = f"Could not save the generated image locally: {exc}"
        return {
            "response": err_msg,
            "error": "output_write_error",
            "kaggle_telemetry": {
                "endpoint": endpoint,
                "duration_seconds": round(duration, 3),
                "error": str(exc),
            },
        }
