"""LangGraph nodes: plan, route, and delegate execution to registered capabilities."""
from __future__ import annotations

import json
from agent.state import AgentState
from agent.llm import groq_chat, conversation_response
from agent.schemas import PlannerOutput
from agent.controller import run_edit_controller

PLANNER_PROMPT = """You are the planner for a general image-editing agent. Return one JSON object matching the provided schema.
Classify the user's latest request as exactly one of:
- conversation/none: ordinary conversation
- image_generation/generate: create a completely new image (not currently supported by the deployed Kaggle service)
- image_edit/add: add something to an existing image
- image_edit/remove: remove an existing element
- image_edit/modify: change an existing element
- image_edit/replace: replace an existing element
- image_analysis/analyze: inspect or describe an image
Rules:
- Do not use object-specific hardcoded placement rules. Infer the existing target/placement region from the request and context.
- For remove/modify/replace, target the existing object to segment. For add, target the existing placement region, not the new object.
- Keep the edit description actionable and preserve all explicit user constraints.
- The backend runs the best available diffusion model: FLUX.1-Fill-dev (preferred, state-of-the-art, needs HF license) or
  Juggernaut XL v9 (automatic fallback, fully open, excellent photorealism). Both serve identical API routes.
  Always emit FLUX-style parameters; the backend automatically remaps them if Juggernaut is active.
- Default inference parameters: num_inference_steps=28, guidance_scale=30.0, controlnet_conditioning_scale=0.0, strength=1.0, seed=42.
- guidance_scale 28–35 for strong edits, 15–25 for subtle changes.
- Do not claim reference-image conditioning is available: the current API accepts source image, optional mask, prompt, target and inference parameters only.
- If no source image is available, still classify the request correctly; the executor will ask for an upload when required.
"""


def _latest_user(messages):
    return next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")


def planner_node(state: AgentState):
    messages = state.get("messages", [])
    latest = _latest_user(messages).strip()
    if not latest:
        return {"error": "No user request was supplied.", "current_task": None}
    payload = {
        "latest_request": latest,
        "recent_context": messages[-10:],
        "source_image_available": bool(state.get("current_image")),
        "manual_mask_available": bool(state.get("mask_image")),
        "reference_image_attached": bool(state.get("reference_images")),
        "schema": PlannerOutput.model_json_schema(),
    }
    try:
        response = groq_chat([
            {"role": "system", "content": PLANNER_PROMPT + " Return JSON only; all keys must match the schema."},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ], response_format={"type": "json_object"}, max_tokens=1400)
        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        validated = PlannerOutput.model_validate(parsed)
        task = validated.model_dump()
        # For the explicitly unsupported operations, do not route into a fake executor.
        return {"current_task": task, "error": None}
    except Exception as exc:
        return {"error": f"Groq planner failed ({type(exc).__name__}): {str(exc)[:240]}", "current_task": None}


def router_node(state: AgentState):
    if state.get("error") or not state.get("current_task"):
        return {"route": "conversation"}
    intent = state["current_task"].get("intent")
    routes = {"conversation": "conversation", "image_generation": "image_generation", "image_edit": "image_edit", "image_analysis": "image_analysis"}
    return {"route": routes.get(intent, "conversation")}


def conversation_node(state: AgentState):
    if state.get("error"):
        return {"response": state["error"], "error": state["error"]}
    try:
        answer = conversation_response(state.get("messages", []))
        return {"response": answer, "error": None}
    except Exception as exc:
        return {"response": f"Groq conversation request failed ({type(exc).__name__}): {str(exc)[:240]}", "error": "groq_conversation_error"}


def image_generation_node(state: AgentState):
    return {"response": "New-image generation is not exposed by the current Kaggle deployment. This workflow currently supports editing an uploaded image; generation needs a separate registered backend capability.", "error": "generation_not_supported"}


def image_edit_node(state: AgentState):
    result = run_edit_controller(state)
    return result


def image_analysis_node(state: AgentState):
    path = state.get("current_image")
    if not path:
        return {"response": "Upload an image first, then ask me to inspect it.", "error": "missing_image"}
    try:
        from image.analyzer import analyze_image
        details = analyze_image(path)
        if not details.get("ok"):
            return {"response": details.get("error", "Image inspection failed."), "error": "image_analysis_error"}

        user_req = _latest_user(state.get("messages", []))
        summary = details.get("summary", "Image metadata read successfully.")

        try:
            prompt = (
                "You are an assistant with grounded local image analysis tools. "
                "Answer the user request accurately using this verified image metadata. "
                "Do not invent invisible details. If semantic scene understanding is requested, "
                "state clearly what is known from the measurements and note that open-domain scene understanding "
                "relies on these structural measurements.\n"
                f"Verified Measurements: {json.dumps(details, default=str)}\n"
                f"User request: {user_req}"
            )
            response = groq_chat([{"role": "user", "content": prompt}], max_tokens=400)
            text = (response.choices[0].message.content or "").strip()
            return {"response": text or summary, "error": None, "analysis": details}
        except Exception:
            # Deterministic fallback when Groq is unavailable or unconfigured
            return {"response": f"Image analysis results:\n{summary}", "error": None, "analysis": details}
    except Exception as exc:
        return {"response": f"Image inspection failed ({type(exc).__name__}): {str(exc)[:200]}", "error": "image_analysis_error"}

