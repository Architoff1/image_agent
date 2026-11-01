"""Bounded Groq tool-calling controller for the image-editing workflow.

Groq chooses and orders registered tools. The Python side validates every argument,
limits edit attempts, and keeps file paths/secrets out of model-visible arguments.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageChops

from agent.llm import groq_chat, model_name
from agent.state import AgentState
from image.manipulator import execute_image_edit

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_current_image",
            "description": "Inspect the currently selected source image's dimensions and format before editing.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_gpu_backend",
            "description": "Check whether the authenticated Kaggle GPU inference backend is reachable and ready.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_image_edit",
            "description": "Run one real edit using the Kaggle GPU inpainting service. Uses FLUX.1-Fill-dev (preferred, state-of-the-art) or Juggernaut XL v9 (automatic fallback, fully open). Use only after inspecting the image and checking the backend. The operation must match the approved plan. Always provide FLUX-style parameters; the backend remaps them if Juggernaut is active.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["add", "remove", "modify", "replace"]},
                    "target": {"type": "string", "minLength": 1, "maxLength": 160},
                    "description": {"type": "string", "minLength": 1, "maxLength": 1200},
                    "num_inference_steps": {"type": "integer", "minimum": 1, "maximum": 50},
                    "guidance_scale": {"type": "number", "minimum": 1, "maximum": 50},
                    "controlnet_conditioning_scale": {"type": "number", "minimum": 0, "maximum": 2},
                    "strength": {"type": "number", "minimum": 0, "maximum": 1},
                    "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647},
                },
                "required": ["operation", "target", "description", "num_inference_steps", "guidance_scale", "controlnet_conditioning_scale", "strength", "seed"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_image_output",
            "description": "Verify that the last edit produced a readable image with the original dimensions. When a manual mask was supplied, verify exact pixel preservation outside it.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def _inspect_image(state: AgentState) -> dict[str, Any]:
    path = state.get("current_image")
    if not path or not Path(path).is_file():
        return {"ok": False, "error": "No source image is available. Ask the user to upload an image."}
    try:
        from image.analyzer import analyze_image
        res = analyze_image(path)
        if res.get("ok"):
            return res
        return {"ok": False, "error": res.get("error", "Source image is unreadable.")}
    except Exception as exc:
        return {"ok": False, "error": f"Source image is unreadable ({type(exc).__name__})."}


def _check_backend() -> dict[str, Any]:
    base = os.getenv("GPU_BACKEND_URL", "").strip().rstrip("/")
    token = os.getenv("GPU_BACKEND_TOKEN", "").strip()
    if not base or "YOUR_" in base or "your-" in base.lower():
        return {"ok": False, "error": "GPU_BACKEND_URL is not configured with the active Kaggle HTTPS URL."}
    if base.endswith("/v1/edit"):
        base = base[:-len("/v1/edit")].rstrip("/")
    if not token or token.lower().startswith("the_same_value") or token.lower().startswith("replace_"):
        return {"ok": False, "error": "GPU_BACKEND_TOKEN is not configured with the same secret used by Kaggle."}
    headers = {"X-Backend-Token": token}
    try:
        response = requests.get(base + "/health", headers=headers, timeout=(8, 20))
        if response.status_code != 200:
            return {"ok": False, "http_status": response.status_code, "error": "GPU backend health check failed. Check the Kaggle session, tunnel URL, and backend token."}
        data = response.json()
        ready = data.get("status") in {"ok", "ready"} and data.get("model_loaded", True)
        return {"ok": bool(ready), "service": data.get("service", "gpu-backend"), "model_loaded": data.get("model_loaded", True), "automatic_masking": data.get("automatic_masking"), "adapter": data.get("adapter"), "error": None if ready else "GPU service is reachable but not ready."}
    except requests.RequestException as exc:
        return {"ok": False, "error": f"GPU backend unreachable ({type(exc).__name__}). Verify the active Kaggle tunnel."}
    except ValueError:
        return {"ok": False, "error": "GPU backend health endpoint returned invalid JSON."}


def _verify_output(context: dict[str, Any], state: AgentState) -> dict[str, Any]:
    result = context.get("last_edit_result") or {}
    output_path = result.get("current_image")
    source_path = state.get("current_image")
    if result.get("error") or not output_path or not Path(output_path).is_file():
        return {"ok": False, "error": result.get("response", "No output image exists to verify.")}
    try:
        with Image.open(output_path) as out:
            out.verify()
        with Image.open(output_path) as out:
            out = out.convert("RGB")
            output_size = out.size
        with Image.open(source_path) as src:
            source_size = src.size
        if output_size != source_size:
            return {"ok": False, "error": f"Output dimensions {output_size} do not match source dimensions {source_size}."}
        manual_mask = state.get("mask_image")
        if manual_mask and Path(manual_mask).is_file():
            with Image.open(source_path) as im:
                original = im.convert("RGB")
            with Image.open(output_path) as im:
                edited = im.convert("RGB")
            with Image.open(manual_mask) as im:
                mask = im.convert("L").resize(source_size, Image.Resampling.NEAREST).point(lambda p: 255 if p >= 128 else 0)
            difference = ImageChops.difference(original, edited).convert("L")
            outside = Image.eval(mask, lambda p: 255 - p)
            outside_difference = ImageChops.multiply(difference, outside)
            if outside_difference.getbbox():
                return {"ok": False, "error": "Output changed pixels outside the manual editable mask."}
            return {"ok": True, "width": output_size[0], "height": output_size[1], "outside_mask_preserved": True, "note": "This validates file integrity and preservation, not semantic realism."}
        return {"ok": True, "width": output_size[0], "height": output_size[1], "outside_mask_preserved": None, "note": "File integrity and dimensions passed. Automatic-mask semantic quality still requires visual review."}
    except Exception as exc:
        return {"ok": False, "error": f"Output verification failed ({type(exc).__name__})."}


def run_edit_controller(state: AgentState) -> dict[str, Any]:
    task = dict(state.get("current_task") or {})
    if task.get("operation") not in {"add", "remove", "modify", "replace"}:
        return {"response": "The current plan does not describe a supported image-editing operation.", "error": "unsupported_operation", "tool_trace": []}
    if not state.get("current_image") or not Path(state["current_image"]).is_file():
        return {"response": "Please upload an image before requesting an edit.", "error": "missing_image", "tool_trace": []}

    max_rounds = max(2, min(int(os.getenv("GROQ_MAX_TOOL_ROUNDS", "8")), 12))
    max_edits = max(1, min(int(os.getenv("GROQ_MAX_EDIT_ATTEMPTS", "2")), 2))
    context: dict[str, Any] = {"edit_attempts": 0, "last_edit_result": None, "backend_check": None, "verification_passed": False, "last_verification": None}
    trace: list[dict[str, Any]] = []
    task_json = _json(task)
    user_request = next((m.get("content", "") for m in reversed(state.get("messages", [])) if m.get("role") == "user"), task.get("description", ""))
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": (
            "You are the execution supervisor for a real image-editing agent. You must use the registered tools; "
            "do not claim success without a verified output. Workflow: (1) call inspect_current_image, (2) call check_gpu_backend, "
            "(3) if both succeed, call execute_image_edit using the approved plan, (4) call verify_image_output. "
            "If an edit fails, you may make at most one retry with adjusted valid parameters. Do not change the planned operation. "
            "Do not ask for or expose secrets or filesystem paths. Do not invent capabilities. "
            "The backend runs the best available diffusion model: FLUX.1-Fill-dev (preferred, state-of-the-art flow-matching inpainting) "
            "or Juggernaut XL v9 (automatic fallback, fully open, photorealistic SDXL). Both use Grounding DINO/SAM for automatic masking. "
            "Always use FLUX-style parameters; the backend remaps them to the active model automatically. "
            "Default FLUX parameters: num_inference_steps=28, guidance_scale=30.0, strength=1.0, seed=42. "
            "Use guidance_scale 28–35 for most edits; lower (15–25) for subtle changes. "
            "When the tool workflow is complete, summarize what happened accurately and report which model was used."
        )},
        {"role": "user", "content": f"User request: {user_request}\nApproved plan: {task_json}\nA source image is attached to the execution context; use tools to operate on it."},
    ]

    def dispatch(name: str, raw_args: str) -> dict[str, Any]:
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            return {"ok": False, "error": "Tool arguments were not valid JSON."}
        if not isinstance(args, dict):
            return {"ok": False, "error": "Tool arguments must be a JSON object."}
        if name == "inspect_current_image":
            result = _inspect_image(state)
        elif name == "check_gpu_backend":
            result = _check_backend()
            context["backend_check"] = result
        elif name == "execute_image_edit":
            if context["edit_attempts"] >= max_edits:
                return {"ok": False, "error": f"Edit attempt limit reached ({max_edits})."}
            if context.get("last_edit_result") and not context["last_edit_result"].get("error") and context.get("last_verification") is None:
                return {"ok": False, "error": "A previous edit already produced an image. Verify it before requesting another edit."}
            if context.get("verification_passed"):
                return {"ok": False, "error": "The previous output passed verification; do not run another edit in this request."}
            if not context.get("backend_check", {}).get("ok"):
                return {"ok": False, "error": "Run a successful check_gpu_backend tool before editing."}
            requested_operation = args.get("operation", task.get("operation"))
            if requested_operation != task.get("operation"):
                return {"ok": False, "error": "Operation must match the approved plan."}
            try:
                steps = max(1, min(int(args.get("num_inference_steps", task.get("num_inference_steps", 28))), 50))
                guidance = max(1.0, min(float(args.get("guidance_scale", task.get("guidance_scale", 30.0))), 50.0))
                control = max(0.0, min(float(args.get("controlnet_conditioning_scale", task.get("controlnet_conditioning_scale", 0.0))), 2.0))
                strength = max(0.0, min(float(args.get("strength", task.get("strength", 1.0))), 1.0))
                seed = max(0, min(int(args.get("seed", task.get("seed", 42))), 2147483647))
            except (TypeError, ValueError):
                return {"ok": False, "error": "Inference parameters must be numeric and within their allowed ranges."}
            edited_task = dict(task)
            edited_task.update({
                "operation": requested_operation,
                "target": str(args.get("target", task.get("target", "object")))[:160],
                "description": str(args.get("description", task.get("description", "")))[:1200],
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "controlnet_conditioning_scale": control,
                "strength": strength,
                "seed": seed,
            })
            context["edit_attempts"] += 1
            context["last_verification"] = None
            context["verification_passed"] = False
            result = execute_image_edit({**state, "current_task": edited_task}, options=edited_task)
            context["last_edit_result"] = result
            if result.get("error"):
                result = {"ok": False, "attempt": context["edit_attempts"], "error": result.get("response", result.get("error")), "error_code": result.get("error")}
            else:
                result = {"ok": True, "attempt": context["edit_attempts"], "status": result.get("response"), "parameters": {"num_inference_steps": steps, "guidance_scale": guidance, "controlnet_conditioning_scale": control, "strength": strength, "seed": seed}}
        elif name == "verify_image_output":
            result = _verify_output(context, state)
            context["verification_passed"] = bool(result.get("ok"))
            context["last_verification"] = result
        else:
            result = {"ok": False, "error": f"Unknown tool '{name}'."}
        trace.append({"tool": name, "result": result})
        return result

    try:
        for _ in range(max_rounds):
            response = groq_chat(messages, tools=TOOL_DEFINITIONS, tool_choice="auto", max_tokens=1200)
            message = response.choices[0].message
            calls = getattr(message, "tool_calls", None) or []
            if not calls:
                final_text = (message.content or "").strip()
                # Safety fallback: a plan must not be left unexecuted because the model stopped early.
                if not context.get("last_edit_result"):
                    if context.get("backend_check") is None:
                        check = _check_backend(); context["backend_check"] = check
                        trace.append({"tool": "check_gpu_backend", "result": check, "fallback": True})
                    if context.get("backend_check", {}).get("ok"):
                        fallback_args = _json({
                            "operation": task["operation"], "target": task.get("target", "object"),
                            "description": task.get("description", user_request),
                            "num_inference_steps": task.get("num_inference_steps", 28),
                            "guidance_scale": task.get("guidance_scale", 30.0),
                            "controlnet_conditioning_scale": task.get("controlnet_conditioning_scale", 0.0),
                            "strength": task.get("strength", 1.0), "seed": task.get("seed", 42),
                        })
                        fallback = dispatch("execute_image_edit", fallback_args)
                        trace.append({"tool": "execute_image_edit", "result": fallback, "fallback": True})
                        if fallback.get("ok"):
                            verified = dispatch("verify_image_output", "{}")
                            if not verified.get("ok"):
                                return {"response": "An image was generated, but output verification did not pass: " + str(verified.get("error")), "error": "verification_failed", "tool_trace": trace}
                            trace.append({"tool": "verify_image_output", "result": verified, "fallback": True})
                            current = context["last_edit_result"]
                            return {"current_image": current.get("current_image"), "image_history": current.get("image_history", []), "response": "Image edit completed and output checks passed. " + str(verified.get("note", "")), "error": None, "tool_trace": trace}
                    if not context.get("last_edit_result"):
                        if not context.get("backend_check", {}).get("ok"):
                            return {"response": "No image edit was run because the Kaggle GPU backend did not pass its health check. Verify the current tunnel URL, token, and running notebook, then retry.", "error": "gpu_backend_unavailable", "tool_trace": trace}
                        return {"response": "The controller did not execute the planned edit. Check whether the configured Groq model supports tool calling, then retry.", "error": "edit_not_executed", "tool_trace": trace}
                result = context.get("last_edit_result") or {}
                if result.get("error"):
                    return {"response": result.get("response", final_text or "Image edit failed."), "error": result.get("error"), "tool_trace": trace}
                verification = _verify_output(context, state)
                trace.append({"tool": "verify_image_output", "result": verification, "automatic": True})
                if not verification.get("ok"):
                    return {"response": "Image generation returned a file, but verification failed: " + str(verification.get("error")), "error": "verification_failed", "tool_trace": trace}
                return {"current_image": result.get("current_image"), "image_history": result.get("image_history", []), "response": final_text or "Image edit completed and output checks passed.", "error": None, "tool_trace": trace}

            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [call.model_dump(exclude_none=True) for call in calls],
            })
            for call in calls:
                name = call.function.name
                outcome = dispatch(name, call.function.arguments)
                messages.append({"role": "tool", "tool_call_id": call.id, "name": name, "content": _json(outcome)})
                if name == "execute_image_edit" and not outcome.get("ok") and context["edit_attempts"] >= max_edits:
                    # Continue once so Groq can give an accurate explanation; no more inference calls are permitted.
                    continue
        result = context.get("last_edit_result") or {}
        if result and not result.get("error"):
            verified = _verify_output(context, state)
            trace.append({"tool": "verify_image_output", "result": verified, "automatic": True})
            if verified.get("ok"):
                return {"current_image": result.get("current_image"), "image_history": result.get("image_history", []), "response": "Image edit completed and output checks passed.", "error": None, "tool_trace": trace}
        return {"response": "The controller reached its bounded tool-call limit. No unverified success was reported. Review the tool trace and try again if needed.", "error": "tool_round_limit", "tool_trace": trace}
    except Exception as exc:
        return {"response": f"Groq tool controller failed ({type(exc).__name__}): {str(exc)[:240]}. Check GROQ_API_KEY, GROQ_MODEL, and whether that model supports tool calling.", "error": "groq_controller_error", "tool_trace": trace}
