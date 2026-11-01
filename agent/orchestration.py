"""Dynamic, capability-backed task orchestration (Phases 1-4).

The LLM proposes a task DAG; Python validates it against the live capability registry,
executes dependency-ready tasks concurrently, records evidence, and evaluates the final
result against the original request. No model-generated Python or shell commands execute.
"""
from __future__ import annotations
import json, os, re, uuid, hashlib, html, urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Callable
from pydantic import BaseModel, Field, ValidationError, model_validator
from agent.llm import groq_chat

MAX_TASKS = int(os.getenv("AGENT_MAX_TASKS", "12"))
MAX_WORKERS = int(os.getenv("AGENT_MAX_WORKERS", "4"))
MAX_RETRIES = int(os.getenv("AGENT_MAX_RETRIES", "2"))

class PlannedTask(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_-]{0,47}$")
    title: str = Field(min_length=1, max_length=120)
    objective: str = Field(min_length=1, max_length=1200)
    capability: str = Field(min_length=1, max_length=100)
    depends_on: list[str] = Field(default_factory=list, max_length=12)
    acceptance_criteria: list[str] = Field(min_length=1, max_length=8)
    inputs: dict[str, Any] = Field(default_factory=dict)
    max_retries: int = Field(default=1, ge=0, le=3)

class TaskPlan(BaseModel):
    goal: str = Field(min_length=1, max_length=4000)
    rationale: str = Field(default="", max_length=1500)
    tasks: list[PlannedTask] = Field(min_length=1, max_length=12)
    final_acceptance_criteria: list[str] = Field(min_length=1, max_length=10)
    @model_validator(mode="after")
    def validate_dag(self):
        ids = [t.id for t in self.tasks]
        if len(ids) != len(set(ids)): raise ValueError("Task IDs must be unique")
        known = set(ids)
        for t in self.tasks:
            if t.id in t.depends_on: raise ValueError(f"Task {t.id} cannot depend on itself")
            missing = set(t.depends_on) - known
            if missing: raise ValueError(f"Task {t.id} depends on unknown tasks: {sorted(missing)}")
        visited, active = set(), set()
        by_id = {t.id:t for t in self.tasks}
        def visit(n):
            if n in active: raise ValueError("Task dependencies contain a cycle")
            if n in visited: return
            active.add(n)
            for dep in by_id[n].depends_on: visit(dep)
            active.remove(n); visited.add(n)
        for n in ids: visit(n)
        return self

@dataclass(frozen=True)
class Capability:
    name: str
    description: str
    handler: Callable[[dict[str, Any]], dict[str, Any]]
    input_schema: dict[str, Any]
    mutates_state: bool = False

class CapabilityRegistry:
    def __init__(self): self._items: dict[str, Capability] = {}; self._lock = RLock()
    def register(self, capability: Capability):
        if not re.fullmatch(r"[a-z][a-z0-9_.-]{1,99}", capability.name): raise ValueError("Invalid capability name")
        with self._lock:
            if capability.name in self._items: raise ValueError(f"Capability already registered: {capability.name}")
            self._items[capability.name] = capability
    def get(self, name: str) -> Capability:
        with self._lock:
            if name not in self._items: raise ValueError(f"Unknown capability '{name}'. Available: {', '.join(sorted(self._items))}")
            return self._items[name]
    def describe(self):
        with self._lock: return [{"name": c.name, "description": c.description, "input_schema": c.input_schema, "mutates_state": c.mutates_state} for c in self._items.values()]

registry = CapabilityRegistry()

def _image_inspect(ctx):
    path = ctx.get("current_image")
    if not path or not Path(path).is_file(): return {"ok": False, "error": "No readable source image is available."}
    from image.analyzer import analyze_image
    result = analyze_image(path)
    if result.get("ok"):
        return result
    return {"ok": False, "error": result.get("error", "Image inspection failed.")}

def _image_edit(ctx):
    from agent.controller import run_edit_controller
    task = dict(ctx.get("task") or {})
    task.update(ctx.get("inputs") or {})
    task["intent"] = "image_edit"
    task.setdefault("operation", "modify")
    task.setdefault("description", ctx.get("objective", ""))
    state = {"messages": ctx.get("messages", []), "current_image": ctx.get("current_image"), "mask_image": ctx.get("mask_image"), "reference_images": ctx.get("reference_images", []), "current_task": task, "tool_trace": []}
    result = run_edit_controller(state)
    return {"ok": not bool(result.get("error")), "response": result.get("response"), "error": result.get("error"), "current_image": result.get("current_image", ctx.get("current_image")), "task": task, "tool_trace": result.get("tool_trace", [])}

def _conversation_respond(ctx):
    from agent.llm import conversation_response
    messages = list(ctx.get("messages", []))
    messages.append({"role":"user", "content": ctx.get("goal", ctx.get("objective", ""))})
    return {"ok": True, "response": conversation_response(messages)}

registry.register(Capability("image.inspect", "Validate an uploaded image and return actual image metadata.", _image_inspect, {"type":"object","properties":{},"additionalProperties":False}))
registry.register(Capability("image.edit", "Run the existing real SDXL/ControlNet image-edit controller through the configured GPU backend.", _image_edit, {"type":"object","properties":{"operation":{"type":"string","enum":["add","remove","modify","replace"]},"target":{"type":"string"},"description":{"type":"string"}},"additionalProperties":True}, True))
registry.register(Capability("conversation.respond", "Generate a response using the configured Groq chat model.", _conversation_respond, {"type":"object","properties":{},"additionalProperties":False}))


# Real, bounded capabilities beyond the original image workflow. These operate on actual
# network responses and filesystem bytes; they never report simulated success.
WORKSPACE_ROOT = Path(os.getenv("AGENT_WORKSPACE_DIR", str(Path(__file__).resolve().parents[1] / "workspace"))).expanduser().resolve()
MAX_WORKSPACE_FILE_BYTES = int(os.getenv("AGENT_MAX_FILE_BYTES", "2000000"))

def _ddg_html_search(query: str, max_results: int = 8) -> list[dict[str, Any]]:
    """Public web search via DuckDuckGo HTML search without requiring external API keys."""
    import subprocess
    from bs4 import BeautifulSoup
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    import requests
    html_content = ""
    try:
        resp = requests.post("https://html.duckduckgo.com/html/", data={"q": query}, headers=headers, timeout=(4, 10))
        if resp.status_code == 200:
            html_content = resp.text
    except Exception:
        pass

    if not html_content:
        try:
            cmd = [
                "curl", "-s", "-L",
                "--user-agent", headers["User-Agent"],
                "--data-urlencode", f"q={query}",
                "https://html.duckduckgo.com/html/",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=12)
            if proc.returncode == 0:
                html_content = proc.stdout
        except Exception:
            pass

    if not html_content:
        return []

    soup = BeautifulSoup(html_content, "html.parser")
    results = []
    for result in soup.find_all("div", class_="web-result"):
        title_tag = result.find("a", class_="result__a")
        snippet_tag = result.find("a", class_="result__snippet")
        if not title_tag:
            continue
        href = title_tag.get("href", "")
        if "uddg=" in href:
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
            href = qs.get("uddg", [href])[0]
        title = title_tag.get_text(strip=True)
        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
        if href.startswith("http") and "duckduckgo.com" not in href:
            results.append({"title": title, "url": href, "snippet": snippet})
            if len(results) >= max_results:
                break
    return results


def _wikipedia_search(query: str, max_results: int = 4) -> list[dict[str, Any]]:
    """Fallback encyclopedia lookup via Wikipedia OpenSearch API."""
    import requests
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "opensearch", "search": query, "limit": max_results, "namespace": 0, "format": "json"},
            headers={"User-Agent": "ImageAgent/1.1 (https://github.com/)"},
            timeout=(3, 8),
        )
        if resp.ok:
            data = resp.json()
            if len(data) >= 4:
                titles, snippets, urls = data[1], data[2], data[3]
                results = []
                for t, s, u in zip(titles, snippets, urls):
                    if u.startswith("http"):
                        results.append({"title": t, "url": u, "snippet": s or t})
                return results
    except Exception:
        pass
    return []


def _web_search(ctx):
    """Search the public web and return sourced result URLs/snippets.

    Uses Tavily API if TAVILY_API_KEY is configured; otherwise uses DuckDuckGo HTML
    search parsing (BeautifulSoup) with zero required API keys.
    """
    query = str(
        (ctx.get("inputs") or {}).get("query")
        or ctx.get("objective")
        or ctx.get("goal")
        or ""
    ).strip()
    if not query:
        return {"ok": False, "query": "", "results": [], "result_count": 0,
                "error": "A non-empty search query is required."}

    max_results = max(1, min(int(os.environ.get("SEARCH_MAX_RESULTS", "8")), 20))
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()

    # Strategy 1: Tavily if API key is provided
    if api_key and not api_key.lower().startswith("replace_"):
        import requests
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"query": query, "search_depth": os.environ.get("TAVILY_SEARCH_DEPTH", "basic"), "max_results": max_results},
                timeout=(5, 30),
            )
            if response.ok:
                payload = response.json()
                raw_results = payload.get("results", []) if isinstance(payload, dict) else []
                results = []
                for item in raw_results:
                    if isinstance(item, dict) and item.get("url"):
                        results.append({
                            "title": str(item.get("title") or "Untitled").strip(),
                            "url": str(item["url"]).strip(),
                            "snippet": str(item.get("content") or item.get("snippet") or "").strip(),
                            **({"score": item["score"]} if isinstance(item.get("score"), (int, float)) else {}),
                        })
                if results:
                    return {"ok": True, "query": query, "results": results, "result_count": len(results), "provider": "tavily"}
        except Exception:
            pass  # Fall through to DuckDuckGo

    # Strategy 2: DuckDuckGo HTML search parsing
    ddg_results = _ddg_html_search(query, max_results=max_results)
    if ddg_results:
        return {"ok": True, "query": query, "results": ddg_results, "result_count": len(ddg_results), "provider": "duckduckgo"}

    # Strategy 3: Wikipedia encyclopedic lookup
    wiki_results = _wikipedia_search(query, max_results=max_results)
    if wiki_results:
        return {"ok": True, "query": query, "results": wiki_results, "result_count": len(wiki_results), "provider": "wikipedia"}

    return {
        "ok": False,
        "query": query,
        "results": [],
        "result_count": 0,
        "error": "Search returned no usable results from upstream sources.",
        "provider": "none",
    }

def _safe_workspace_path(relative):
    rel=Path(str(relative or "").strip())
    if not str(relative).strip() or rel.is_absolute() or any(part in {"..", ""} for part in rel.parts):
        raise ValueError("Path must be a non-empty relative path without '..'.")
    target=(WORKSPACE_ROOT/rel).resolve()
    if target != WORKSPACE_ROOT and WORKSPACE_ROOT not in target.parents:
        raise ValueError("Path escapes the configured agent workspace.")
    return target

def _workspace_write(ctx):
    inputs=ctx.get("inputs") or {}
    rel=inputs.get("path")
    content=inputs.get("content")
    if not isinstance(content,str): return {"ok":False,"error":"'content' must be a string; no file was written."}
    encoded=content.encode("utf-8")
    if len(encoded)>MAX_WORKSPACE_FILE_BYTES: return {"ok":False,"error":f"File exceeds {MAX_WORKSPACE_FILE_BYTES} byte limit."}
    try: target=_safe_workspace_path(rel)
    except ValueError as exc: return {"ok":False,"error":str(exc)}
    target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_name(target.name+".tmp-"+uuid.uuid4().hex)
    try:
        temp.write_bytes(encoded)
        temp.replace(target)
    finally:
        if temp.exists(): temp.unlink()
    actual=target.read_bytes()
    return {"ok":actual==encoded,"path":str(target),"bytes_written":len(actual),"sha256":hashlib.sha256(actual).hexdigest(),"verified_by_readback":actual==encoded}

def _document_create(ctx):
    inputs=ctx.get("inputs") or {}
    title=str(inputs.get("title") or "").strip()
    content=inputs.get("content")
    if not title or not isinstance(content,str) or not content.strip():
        return {"ok":False,"error":"Document creation requires non-empty title and content."}
    safe_title=re.sub(r"[^a-zA-Z0-9._-]+","_",title).strip("._")[:100] or "document"
    rel=inputs.get("path") or f"documents/{safe_title}.md"
    body=f"# {title}\n\n{content.rstrip()}\n"
    nested=dict(ctx); nested["inputs"]={"path":rel,"content":body}
    result=_workspace_write(nested)
    result.update({"document_title":title,"format":"markdown"})
    return result

registry.register(Capability("web.search", "Search the public web and return sourced result URLs/snippets.", _web_search, {"type":"object","properties":{"query":{"type":"string","minLength":1}},"required":["query"],"additionalProperties":False}))
registry.register(Capability("workspace.write_file", "Write and read-back-verify a UTF-8 file inside the configured, path-confined agent workspace.", _workspace_write, {"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"],"additionalProperties":False}, True))
registry.register(Capability("document.create_markdown", "Create a Markdown document in the confined workspace and verify its bytes by readback.", _document_create, {"type":"object","properties":{"title":{"type":"string"},"content":{"type":"string"},"path":{"type":"string"}},"required":["title","content"],"additionalProperties":False}, True))

PLANNER_SYSTEM = """You are a task planner for a real agentic system. Return only JSON matching the supplied schema. Decompose the goal into the smallest useful executable tasks, with explicit dependencies and verifiable acceptance criteria. Use ONLY listed registered capabilities; do not invent capabilities, tools, fake completions, or unsupported work. Prefer independent tasks without dependencies when genuinely safe. Any task requiring an unsupported capability must be reported in the plan rationale and must not be represented as completed. For image edits use image.inspect before image.edit and make image.edit depend on inspection. For web research use web.search and include URLs in the task evidence. For creating a file use workspace.write_file with explicit path and content. For a Markdown document use document.create_markdown with explicit title and content. Never claim to run arbitrary code: no code execution capability is registered. Use conversation.respond only when a natural-language response is the actual requested work. Inputs must be JSON objects and must not contain code to execute. Final acceptance criteria must directly reflect the user's goal."""

def create_plan(goal: str, context: dict[str, Any] | None = None) -> TaskPlan:
    goal = goal.strip()
    if not goal: raise ValueError("Goal cannot be empty")
    if len(goal) > 4000: raise ValueError("Goal exceeds 4000 characters")
    payload = {"goal":goal,"context":context or {},"registered_capabilities":registry.describe(),"schema":TaskPlan.model_json_schema(),"limits":{"max_tasks":MAX_TASKS,"max_workers":MAX_WORKERS,"max_retries":MAX_RETRIES}}
    response = groq_chat([{"role":"system","content":PLANNER_SYSTEM},{"role":"user","content":json.dumps(payload, ensure_ascii=False, default=str)}], response_format={"type":"json_object"}, max_tokens=3500)
    raw = response.choices[0].message.content or "{}"
    plan = TaskPlan.model_validate_json(raw)
    if len(plan.tasks) > MAX_TASKS: raise ValueError(f"Plan has {len(plan.tasks)} tasks; limit is {MAX_TASKS}")
    for task in plan.tasks: registry.get(task.capability)
    return plan

EVALUATOR_SYSTEM = """You are a strict independent evaluator. Evaluate actual task outputs and evidence against the original user goal and final acceptance criteria. Never mark a criterion passed solely because an agent claims success; use concrete returned evidence, errors, and verification results. Return JSON: {\"passed\": boolean, \"score\": number from 0 to 1, \"criteria\": [{\"criterion\": string, \"passed\": boolean, \"evidence\": string}], \"summary\": string, \"retry_task_ids\": [string]}. Only request retry of a failed task whose capability can reasonably fix the failure. Do not say a task succeeded if its result has ok=false or an error."""

def evaluate(plan: TaskPlan, results: dict[str, Any]) -> dict[str, Any]:
    payload = {"goal":plan.goal,"criteria":plan.final_acceptance_criteria,"task_results":results}
    response = groq_chat([{"role":"system","content":EVALUATOR_SYSTEM},{"role":"user","content":json.dumps(payload, ensure_ascii=False, default=str)}], response_format={"type":"json_object"}, max_tokens=1800)
    data = json.loads(response.choices[0].message.content or "{}")
    if not isinstance(data.get("passed"), bool) or not isinstance(data.get("criteria"), list): raise ValueError("Evaluator returned invalid result schema")
    data["score"] = max(0.0, min(1.0, float(data.get("score", 0))))
    data["retry_task_ids"] = [x for x in data.get("retry_task_ids", []) if x in results]
    # Deterministic hard gate: the LLM cannot override failed/blocked/missing required tasks.
    required_ids = {task.id for task in plan.tasks}
    missing = sorted(required_ids - set(results))
    bad = sorted(tid for tid in required_ids if tid in results and results[tid].get("status") != "succeeded")
    if missing or bad:
        data["passed"] = False
        data["summary"] = (str(data.get("summary", "")) + f" Hard gate: missing tasks={missing}; unsuccessful tasks={bad}.").strip()
        data["retry_task_ids"] = list(dict.fromkeys(data["retry_task_ids"] + [tid for tid in bad if results[tid].get("status") == "failed"]))
    return data

def _run_one(task: PlannedTask, goal: str, context: dict[str, Any], prior: dict[str, Any], retry_budget: int | None = None) -> dict[str, Any]:
    capability = registry.get(task.capability)
    inputs = dict(task.inputs)
    inputs.update({k:v for k,v in context.items() if k in {"operation","target","description","num_inference_steps","guidance_scale","controlnet_conditioning_scale","strength","seed"} and k not in inputs})
    args = {**context, "goal":goal, "objective":task.objective, "inputs":inputs, "task":inputs, "dependency_results":prior}
    last = None
    budget = min(task.max_retries, MAX_RETRIES) if retry_budget is None else max(0, min(retry_budget, MAX_RETRIES))
    for attempt in range(budget + 1):
        try:
            out = capability.handler(args)
            if not isinstance(out, dict): out = {"ok":False,"error":"Capability returned a non-object result"}
            last = {"task_id":task.id,"capability":task.capability,"attempt":attempt+1,"status":"succeeded" if out.get("ok") is True else "failed","output":out,"finished_at":datetime.now(timezone.utc).isoformat()}
            if out.get("ok") is True: return last
        except Exception as exc:
            last = {"task_id":task.id,"capability":task.capability,"attempt":attempt+1,"status":"failed","output":{"ok":False,"error":f"{type(exc).__name__}: {str(exc)[:500]}"},"finished_at":datetime.now(timezone.utc).isoformat()}
    return last or {"task_id":task.id,"capability":task.capability,"status":"failed","output":{"ok":False,"error":"Task execution failed"}}

def execute_plan(plan: TaskPlan, context: dict[str, Any] | None = None, *, evaluator: Callable[[TaskPlan, dict[str, Any]], dict[str, Any]] = evaluate) -> dict[str, Any]:
    ctx = dict(context or {}); results: dict[str, Any] = {}; trace: list[dict[str, Any]] = []
    tasks = {t.id:t for t in plan.tasks}; pending = set(tasks)
    while pending:
        failed_deps = [tid for tid in pending if any(results.get(d,{}).get("status") != "succeeded" for d in tasks[tid].depends_on if d in results)]
        for tid in failed_deps:
            results[tid] = {"task_id":tid,"capability":tasks[tid].capability,"status":"blocked","output":{"ok":False,"error":"Dependency failed or was blocked"}}
            pending.remove(tid)
        ready = [tasks[tid] for tid in pending if all(d in results and results[d].get("status")=="succeeded" for d in tasks[tid].depends_on)]
        # Tasks that mutate the same shared image state must be serialized, not raced.
        mutating = [t for t in ready if registry.get(t.capability).mutates_state]
        if mutating: ready = [min(mutating, key=lambda t: list(tasks).index(t.id))]
        if not ready:
            if pending: raise RuntimeError(f"Scheduler deadlock; unresolved tasks: {sorted(pending)}")
            break
        dep_snapshot = {t.id:{d:results[d] for d in t.depends_on} for t in ready}
        with ThreadPoolExecutor(max_workers=max(1,min(MAX_WORKERS,len(ready)))) as pool:
            futures = {pool.submit(_run_one,t,plan.goal,ctx,dep_snapshot[t.id]):t for t in ready}
            batch = []
            for fut in as_completed(futures): batch.append((futures[fut],fut.result()))
        for task, result in batch:
            results[task.id]=result; trace.append({"event":"task_finished","task_id":task.id,"capability":task.capability,"status":result.get("status"),"attempt":result.get("attempt")}); pending.remove(task.id)
            out = result.get("output",{})
            if out.get("current_image"): ctx["current_image"] = out["current_image"]
    # Independent evaluation and bounded remediation cycle. A task is only rerun when it failed
    # or the evaluator explicitly selects it, and its own retry budget is not exceeded.
    evaluation = evaluator(plan, results)
    retries_used: dict[str,int] = {tid:0 for tid in tasks}
    for _ in range(MAX_RETRIES):
        if evaluation.get("passed"): break
        candidates = [tid for tid in evaluation.get("retry_task_ids",[]) if tid in tasks and retries_used[tid] < min(tasks[tid].max_retries,MAX_RETRIES)]
        if not candidates: break
        for tid in candidates:
            task=tasks[tid]; deps={d:results[d] for d in task.depends_on if d in results}
            result=_run_one(task,plan.goal,ctx,deps,retry_budget=0); retries_used[tid]+=1; results[tid]=result
            trace.append({"event":"task_retry","task_id":tid,"attempt":result.get("attempt"),"status":result.get("status")})
            if result.get("output",{}).get("current_image"): ctx["current_image"]=result["output"]["current_image"]
        evaluation=evaluator(plan,results)
    required_ok = all(t.id in results and results[t.id].get("status") == "succeeded" for t in plan.tasks)
    if not required_ok:
        evaluation["passed"] = False
        evaluation["summary"] = (str(evaluation.get("summary", "")) + " Deterministic gate: one or more required tasks did not succeed.").strip()
    return {"run_id":uuid.uuid4().hex,"goal":plan.goal,"plan":plan.model_dump(),"task_results":results,"evaluation":evaluation,"trace":trace,"status":"succeeded" if evaluation.get("passed") and required_ok else "needs_review","limits":{"max_workers":MAX_WORKERS,"max_retries":MAX_RETRIES}}
