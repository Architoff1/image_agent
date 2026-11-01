"""Telemetry, diagnostic logging, and AI report generation for Image Agent.

Provides thread-safe structured logging of system events, Groq calls,
and Kaggle GPU backend interactions, with automatic secret sanitization
and one-click export for sharing with AI models and agents.
"""
from __future__ import annotations

import collections
import datetime
import json
import os
import platform
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("APP_DATA_DIR", str(PROJECT_ROOT / "data"))).expanduser().resolve()
LOGS_DIR = DATA_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

JSONL_LOG_PATH = LOGS_DIR / "diagnostic.jsonl"
TEXT_LOG_PATH = LOGS_DIR / "diagnostic.log"

# Patterns for sensitive keys and tokens to redact
SENSITIVE_KEY_PATTERNS = [
    re.compile(r"api[-_]?key", re.IGNORECASE),
    re.compile(r"token", re.IGNORECASE),
    re.compile(r"secret", re.IGNORECASE),
    re.compile(r"password", re.IGNORECASE),
    re.compile(r"authorization", re.IGNORECASE),
]

SENSITIVE_VALUE_PREFIXES = [
    "gsk_",
    "hf_",
    "sk-",
]


def sanitize_value(val: Any) -> Any:
    """Recursively sanitize credentials and sensitive tokens from objects."""
    if isinstance(val, str):
        # Check known prefixes
        for prefix in SENSITIVE_VALUE_PREFIXES:
            if val.startswith(prefix) and len(val) > len(prefix) + 6:
                return f"{prefix}***[len={len(val)}]"
        if "replace_" in val.lower() or "your_" in val.lower():
            return val
        return val
    elif isinstance(val, dict):
        cleaned = {}
        for k, v in val.items():
            k_str = str(k)
            is_sensitive = any(pat.search(k_str) for pat in SENSITIVE_KEY_PATTERNS)
            if is_sensitive and isinstance(v, str) and v:
                if len(v) > 8:
                    cleaned[k] = f"{v[:3]}***[len={len(v)}]"
                else:
                    cleaned[k] = "***[redacted]"
            else:
                cleaned[k] = sanitize_value(v)
        return cleaned
    elif isinstance(val, (list, tuple, set)):
        return [sanitize_value(x) for x in val]
    return val


class DiagnosticLogger:
    """Thread-safe singleton logger for system, Groq, and Kaggle telemetry."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DiagnosticLogger, cls).__new__(cls)
                cls._instance._init_logger()
            return cls._instance

    def _init_logger(self):
        self._buffer: collections.deque = collections.deque(maxlen=300)
        self._file_lock = threading.Lock()
        self._kaggle_history: collections.deque = collections.deque(maxlen=100)

    def log(
        self,
        category: str,
        action: str,
        conversation_id: Optional[str] = None,
        level: str = "INFO",
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record a structured event into buffer and log files."""
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        record = {
            "timestamp": now_utc.isoformat(),
            "level": level.upper(),
            "category": category,
            "action": action,
            "conversation_id": conversation_id,
            "error": error,
            "details": sanitize_value(details or {}),
        }

        with self._file_lock:
            self._buffer.append(record)

            # Persist to jsonl
            try:
                with open(JSONL_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            except Exception:
                pass

            # Persist to formatted text log
            try:
                line = (
                    f"[{record['timestamp'][:19]}] [{record['level']}] [{record['category']}] "
                    f"[{record['conversation_id'] or 'global'}] {record['action']}"
                )
                if record["error"]:
                    line += f" | ERROR: {record['error']}"
                if record["details"]:
                    details_summary = json.dumps(record["details"], default=str)
                    if len(details_summary) > 300:
                        details_summary = details_summary[:297] + "..."
                    line += f" | {details_summary}"
                with open(TEXT_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass

        return record

    def log_kaggle_call(
        self,
        conversation_id: Optional[str],
        endpoint: str,
        request_meta: Dict[str, Any],
        response_meta: Dict[str, Any],
        duration_s: float,
        status_code: Optional[int] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record full telemetry for a Kaggle GPU backend request."""
        details = {
            "endpoint": endpoint,
            "duration_seconds": round(duration_s, 3),
            "status_code": status_code,
            "request": request_meta,
            "response": response_meta,
        }
        level = "ERROR" if error or (status_code and status_code >= 400) else "INFO"
        record = self.log(
            category="kaggle_gpu",
            action="execute_inference" if "/edit" in endpoint else "health_or_mask",
            conversation_id=conversation_id,
            level=level,
            details=details,
            error=error,
        )
        with self._file_lock:
            self._kaggle_history.append({
                "timestamp": record["timestamp"],
                "conversation_id": conversation_id,
                "endpoint": endpoint,
                "duration_seconds": round(duration_s, 3),
                "status_code": status_code,
                "model_used": response_meta.get("model") or response_meta.get("X-Model"),
                "mask_mode": response_meta.get("mask_mode") or response_meta.get("X-Mask-Mode"),
                "target": request_meta.get("target"),
                "prompt": request_meta.get("prompt"),
                "error": error,
            })
        return record

    def log_planner(
        self,
        conversation_id: Optional[str],
        user_request: str,
        model_name: str,
        parsed_plan: Optional[Dict[str, Any]],
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record planner outcome."""
        return self.log(
            category="groq_planner",
            action="generate_plan",
            conversation_id=conversation_id,
            level="ERROR" if error else "INFO",
            details={
                "model": model_name,
                "user_request": user_request[:300],
                "plan": parsed_plan,
            },
            error=error,
        )

    def log_tool_call(
        self,
        conversation_id: Optional[str],
        tool_name: str,
        arguments: Any,
        result: Any,
        duration_s: float = 0.0,
    ) -> Dict[str, Any]:
        """Record tool call dispatch."""
        is_err = isinstance(result, dict) and (result.get("ok") is False or result.get("error"))
        return self.log(
            category="controller_tool",
            action=f"call_{tool_name}",
            conversation_id=conversation_id,
            level="ERROR" if is_err else "INFO",
            details={
                "tool": tool_name,
                "arguments": arguments,
                "result": result,
                "duration_seconds": round(duration_s, 3),
            },
            error=str(result.get("error")) if is_err else None,
        )

    def get_recent_logs(
        self,
        limit: int = 100,
        conversation_id: Optional[str] = None,
        category: Optional[str] = None,
        min_level: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Filter and return recent in-memory logs."""
        with self._file_lock:
            logs = list(self._buffer)

        filtered = []
        for r in reversed(logs):
            if conversation_id and r.get("conversation_id") and r.get("conversation_id") != conversation_id:
                continue
            if category and r.get("category") != category:
                continue
            if min_level:
                levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
                if levels.index(r.get("level", "INFO")) < levels.index(min_level.upper()):
                    continue
            filtered.append(r)
            if len(filtered) >= limit:
                break
        return filtered

    def get_kaggle_history(self, limit: int = 50, conversation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return history of Kaggle calls."""
        with self._file_lock:
            items = list(self._kaggle_history)
        if conversation_id:
            items = [x for x in items if x.get("conversation_id") == conversation_id]
        return list(reversed(items))[:limit]


# Global logger instance
telemetry = DiagnosticLogger()


def gather_system_diagnostic() -> Dict[str, Any]:
    """Gather non-sensitive environment and backend status."""
    import requests
    from agent.llm import model_name

    gpu_url = os.getenv("GPU_BACKEND_URL", "").strip().rstrip("/")
    if gpu_url.endswith("/v1/edit"):
        gpu_url = gpu_url[:-len("/v1/edit")].rstrip("/")
    has_gpu_url = bool(gpu_url and "YOUR_" not in gpu_url and "your-" not in gpu_url.lower())

    token = os.getenv("GPU_BACKEND_TOKEN", "").strip()
    has_token = bool(token and not token.lower().startswith("the_same") and not token.lower().startswith("replace_"))

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    has_groq_key = bool(groq_key and not groq_key.lower().startswith("replace_") and not groq_key.startswith("gsk_Not"))

    # Test Kaggle backend if configured
    kaggle_health = {"reachable": False, "status_code": None, "details": None, "error": None}
    if has_gpu_url:
        try:
            headers = {"X-Backend-Token": token} if token else {}
            r = requests.get(gpu_url + "/health", headers=headers, timeout=(4, 8))
            kaggle_health["reachable"] = r.ok
            kaggle_health["status_code"] = r.status_code
            kaggle_health["headers"] = dict(r.headers)
            if r.ok:
                kaggle_health["details"] = r.json()
            else:
                kaggle_health["error"] = r.text[:300]
        except Exception as exc:
            kaggle_health["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"

    # Test Groq authentication
    groq_status = {"authenticated": False, "model": model_name(), "error": None}
    if has_groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key, timeout=5.0)
            client.models.list()
            groq_status["authenticated"] = True
        except Exception as exc:
            groq_status["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"

    return {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "system_time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "groq": groq_status,
        "kaggle_backend": {
            "configured": has_gpu_url,
            "url_sanitized": gpu_url[:20] + "..." if len(gpu_url) > 20 else gpu_url,
            "token_configured": has_token,
            "health": kaggle_health,
        },
        "storage": {
            "data_dir": str(DATA_DIR),
            "logs_dir": str(LOGS_DIR),
        },
    }


def build_diagnostic_bundle(
    conversation_id: Optional[str] = None,
    db_conn=None,
) -> Dict[str, Any]:
    """Compile a complete, structured diagnostic dictionary for an AI agent or model."""
    import sqlite3

    system_info = gather_system_diagnostic()
    kaggle_history = telemetry.get_kaggle_history(limit=25, conversation_id=conversation_id)
    recent_logs = telemetry.get_recent_logs(limit=40, conversation_id=conversation_id)

    conv_data = None
    messages_data = []
    versions_data = []

    need_close = False
    if db_conn is None:
        db_path = DATA_DIR / "image_agent.sqlite3"
        if db_path.is_file():
            db_conn = sqlite3.connect(db_path)
            db_conn.row_factory = sqlite3.Row
            need_close = True

    if db_conn:
        try:
            # If no conversation_id, pick the most recently updated one
            if not conversation_id:
                row = db_conn.execute("SELECT id FROM conversations ORDER BY updated_at DESC LIMIT 1").fetchone()
                if row:
                    conversation_id = row["id"]

            if conversation_id:
                conv_row = db_conn.execute("SELECT * FROM conversations WHERE id=?", (conversation_id,)).fetchone()
                if conv_row:
                    conv_data = dict(conv_row)

                for m in db_conn.execute("SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at ASC", (conversation_id,)):
                    d = dict(m)
                    if d.get("task_json"):
                        try:
                            d["task_parsed"] = json.loads(d["task_json"])
                        except Exception:
                            d["task_parsed"] = d["task_json"]
                    messages_data.append(d)

                for v in db_conn.execute("SELECT * FROM image_versions WHERE conversation_id=? ORDER BY version ASC", (conversation_id,)):
                    v_dict = dict(v)
                    p = Path(v_dict["path"]) if v_dict.get("path") else None
                    if p and p.is_file():
                        v_dict["file_exists"] = True
                        v_dict["file_size_bytes"] = p.stat().st_size
                        try:
                            from PIL import Image
                            with Image.open(p) as im:
                                v_dict["dimensions"] = im.size
                        except Exception:
                            pass
                    else:
                        v_dict["file_exists"] = False
                    versions_data.append(v_dict)
        finally:
            if need_close:
                db_conn.close()

    bundle = {
        "diagnostic_version": "1.0.0",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "target_conversation_id": conversation_id,
        "system_info": system_info,
        "conversation": conv_data,
        "messages": messages_data,
        "versions": versions_data,
        "kaggle_interactions": kaggle_history,
        "recent_logs": recent_logs,
    }

    return sanitize_value(bundle)


def generate_markdown_report(bundle: Dict[str, Any]) -> str:
    """Format the diagnostic bundle as a comprehensive Markdown report for AI models.

    Includes explicit AI diagnosis prompts, formatted tables, tool traces,
    and Kaggle request/response details.
    """
    lines = []
    lines.append("# 🩺 Image Agent Diagnostic Report (AI & Debugging Bundle)")
    lines.append(f"> Generated at: `{bundle.get('generated_at')}` | Target Chat: `{bundle.get('target_conversation_id') or 'N/A'}`\n")
    lines.append("## 🤖 Context for AI Models & Diagnostic Agents")
    lines.append(
        "You are inspecting a diagnostic report from an autonomous Image Editing Agent.\n"
        "- **Stack:** Streamlit UI -> Local FastAPI server -> LangGraph planner & controller (Groq LLM) -> Kaggle GPU Inpainting Backend (FLUX.1-Fill-dev / SDXL Inpainting 1.0 + Grounding DINO / SAM).\n"
        "- **Goal:** Analyze the conversation history, planner decisions, tool execution trace, Kaggle GPU responses, and errors below to pinpoint any failure, unexpected behavior, or configuration defect.\n"
    )

    # 1. System & Connectivity Status
    sys_info = bundle.get("system_info", {})
    groq_info = sys_info.get("groq", {})
    kaggle_info = sys_info.get("kaggle_backend", {})
    health = kaggle_info.get("health", {}) or {}

    lines.append("## 1. ⚙️ System & Backend Connectivity")
    lines.append(f"- **OS & Python:** `{sys_info.get('platform')}` | Python `{sys_info.get('python_version')}`")
    lines.append(f"- **Groq Model:** `{groq_info.get('model')}` | **Authenticated:** `{groq_info.get('authenticated')}`")
    if groq_info.get("error"):
        lines.append(f"  - ⚠️ **Groq Error:** `{groq_info.get('error')}`")

    lines.append(f"- **Kaggle GPU Backend Configured:** `{kaggle_info.get('configured')}` | **URL:** `{kaggle_info.get('url_sanitized')}`")
    lines.append(f"- **Kaggle Reachable:** `{health.get('reachable')}` (HTTP `{health.get('status_code')}`)")
    if health.get("details"):
        d = health["details"]
        lines.append(f"  - **Backend Model:** `{d.get('model')}`")
        lines.append(f"  - **GPU Hardware:** `{d.get('gpu')}`")
        lines.append(f"  - **Model Loaded:** `{d.get('model_loaded')}`")
        lines.append(f"  - **Automatic Masking (DINO+SAM):** `{d.get('automatic_masking')}`")
    if health.get("error"):
        lines.append(f"  - ⚠️ **Kaggle Health Error:** `{health.get('error')}`")

    lines.append("")

    # 2. Conversation Metadata
    conv = bundle.get("conversation")
    lines.append("## 2. 💬 Active Conversation Overview")
    if conv:
        lines.append(f"- **Title:** `{conv.get('title')}`")
        lines.append(f"- **Conversation ID:** `{conv.get('id')}`")
        lines.append(f"- **Created:** `{conv.get('created_at')}` | **Updated:** `{conv.get('updated_at')}`")
        lines.append(f"- **Current Active Image Path:** `{conv.get('current_image') or 'None'}`")
    else:
        lines.append("*No conversation selected or found in database.*")
    lines.append("")

    # 3. Image Version Lineage
    versions = bundle.get("versions", [])
    lines.append(f"## 3. 🖼️ Image Version Lineage ({len(versions)} versions)")
    if versions:
        lines.append("| Version | Kind | Dimensions | Size | Parent | File Exists | Path |")
        lines.append("|---|---|---|---|---|---|---|")
        for v in versions:
            dim = f"{v.get('dimensions', [0, 0])[0]}×{v.get('dimensions', [0, 0])[1]}" if v.get("dimensions") else "N/A"
            sz = f"{v.get('file_size_bytes', 0) / 1024:.1f} KB" if v.get("file_size_bytes") else "N/A"
            exists = "✅" if v.get("file_exists") else "❌"
            p = Path(v.get("path", "")).name
            lines.append(f"| **V{v.get('version')}** | `{v.get('kind')}` | {dim} | {sz} | V{v.get('parent_version') or 'root'} | {exists} | `{p}` |")
    else:
        lines.append("*No image versions recorded.*")
    lines.append("")

    # 4. Chronological Chat & Tool Execution Trace
    messages = bundle.get("messages", [])
    lines.append(f"## 4. 📜 Detailed Chronological Chat & Execution Log ({len(messages)} messages)")
    for idx, m in enumerate(messages, start=1):
        role_icon = "👤 User" if m.get("role") == "user" else "🤖 Assistant"
        lines.append(f"### Message #{idx} · {role_icon} · `{m.get('created_at', '')[:19].replace('T', ' ')}`")
        if m.get("content"):
            lines.append(f"**Text:**\n> {m.get('content')}\n")
        if m.get("attachment_path"):
            lines.append(f"- 📎 **Source Attachment:** `{Path(m['attachment_path']).name}`")
        if m.get("reference_path"):
            lines.append(f"- 📎 **Reference Attachment:** `{Path(m['reference_path']).name}`")
        if m.get("image_version"):
            lines.append(f"- 🖼️ **Created Image Version:** `V{m['image_version']}`")
        if m.get("error"):
            lines.append(f"- ❌ **Error Recorded:** `{m['error']}`")

        # Task & tool trace details
        task = m.get("task_parsed")
        if isinstance(task, dict):
            tool_trace = task.get("_tool_trace") or []
            kaggle_meta = task.get("kaggle_telemetry")
            clean_task = {k: v for k, v in task.items() if k not in ("_tool_trace", "kaggle_telemetry")}
            if clean_task:
                lines.append("<details><summary>🔍 Edit Plan / Task Details</summary>\n")
                lines.append("```json")
                lines.append(json.dumps(clean_task, indent=2, ensure_ascii=False, default=str))
                lines.append("```\n</details>")

            if tool_trace:
                lines.append("<details><summary>🛠️ Groq Controller Tool Trace</summary>\n")
                lines.append("| Step | Tool Name | Outcome / Status | Note |")
                lines.append("|---|---|---|---|")
                for tidx, step in enumerate(tool_trace, start=1):
                    tname = step.get("tool")
                    tres = step.get("result", {})
                    tok = "✅ OK" if (isinstance(tres, dict) and tres.get("ok")) else "❌ FAILED"
                    note = tres.get("error") or tres.get("status") or tres.get("note") or ""
                    lines.append(f"| {tidx} | `{tname}` | {tok} | {str(note)[:80]} |")
                lines.append("```json")
                lines.append(json.dumps(tool_trace, indent=2, ensure_ascii=False, default=str))
                lines.append("```\n</details>")

            if kaggle_meta:
                lines.append("<details><summary>⚡ Kaggle Inpainting Telemetry for this Message</summary>\n")
                lines.append("```json")
                lines.append(json.dumps(kaggle_meta, indent=2, ensure_ascii=False, default=str))
                lines.append("```\n</details>")

        lines.append("---")

    # 5. Kaggle Interactions
    k_calls = bundle.get("kaggle_interactions", [])
    lines.append(f"## 5. ⚡ Kaggle GPU Backend Calls ({len(k_calls)} recorded)")
    if k_calls:
        lines.append("| Timestamp | Endpoint | Status | Latency | Model | Mask Mode | Target | Error |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for k in k_calls:
            ts = k.get("timestamp", "")[:19].replace("T", " ")
            ep = k.get("endpoint", "")
            st = k.get("status_code") or "N/A"
            lat = f"{k.get('duration_seconds')}s" if k.get("duration_seconds") is not None else "N/A"
            mod = k.get("model_used") or "N/A"
            mm = k.get("mask_mode") or "N/A"
            tgt = (k.get("target") or "")[:20]
            err = (k.get("error") or "")[:35]
            lines.append(f"| {ts} | `{ep}` | `{st}` | {lat} | {mod} | {mm} | {tgt} | {err} |")
    else:
        lines.append("*No direct Kaggle GPU backend calls logged in this session.*")
    lines.append("")

    # 6. Recent Logs
    r_logs = bundle.get("recent_logs", [])
    lines.append(f"## 6. 📜 Recent Telemetry Logs ({len(r_logs)} events)")
    if r_logs:
        lines.append("```text")
        for log in r_logs:
            ts = log.get("timestamp", "")[:19]
            lvl = log.get("level", "INFO")
            cat = log.get("category", "")
            act = log.get("action", "")
            err = f" | ERROR: {log['error']}" if log.get("error") else ""
            det = f" | {json.dumps(log.get('details', {}), default=str)[:120]}" if log.get("details") else ""
            lines.append(f"[{ts}] [{lvl:5s}] [{cat:12s}] {act}{err}{det}")
        lines.append("```")
    else:
        lines.append("*No recent logs recorded.*")

    lines.append("\n## 7. 💡 Recommended Next Actions for AI")
    lines.append(
        "1. Check if Kaggle GPU Backend is connected and reporting `model_loaded=True`.\n"
        "2. If `Groq planner failed (NotFoundError)`: Verify that `GROQ_MODEL` in `.env` is set to an active model (e.g., `llama-3.3-70b-versatile` or `llama-3.1-8b-instant`).\n"
        "3. If Kaggle returned HTTP 401 or 404: Verify the `ngrok` tunnel is still active and `GPU_BACKEND_TOKEN` matches Kaggle Secrets.\n"
        "4. If inpainting result didn't change the desired area: Check the mask mode (`automatic` vs `provided`) and target prompt.\n"
    )

    return "\n".join(lines)
