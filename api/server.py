"""Local API with persistent conversations and immutable image versions."""
import os, uuid, sqlite3, json
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / '.env', override=False)
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from agent.graph import graph
from agent.orchestration import create_plan, execute_plan, registry, TaskPlan

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.getenv('APP_DATA_DIR', str(PROJECT_ROOT / 'data'))).expanduser().resolve()
ROOT.mkdir(parents=True, exist_ok=True)
UPLOADS = ROOT / 'uploads'; UPLOADS.mkdir(parents=True, exist_ok=True)
OUTPUTS = Path(os.getenv('OUTPUT_DIR', str(ROOT / 'outputs'))).expanduser()
OUTPUTS = OUTPUTS.resolve() if OUTPUTS.is_absolute() else (PROJECT_ROOT / OUTPUTS).resolve()
OUTPUTS.mkdir(parents=True, exist_ok=True)
DB_PATH = ROOT / 'image_agent.sqlite3'
app = FastAPI(title='Image Agent API', version='1.1.0')
ALLOWED = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

def now(): return datetime.now(timezone.utc).isoformat()
@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH, timeout=30); conn.row_factory = sqlite3.Row
    try:
        conn.execute('PRAGMA foreign_keys=ON')
        yield conn
        conn.commit()
    finally: conn.close()

def init_db():
    with db() as c:
        c.executescript('''
        CREATE TABLE IF NOT EXISTS conversations(id TEXT PRIMARY KEY,title TEXT NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,current_image TEXT);
        CREATE TABLE IF NOT EXISTS messages(id TEXT PRIMARY KEY,conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,role TEXT NOT NULL,content TEXT NOT NULL,created_at TEXT NOT NULL,task_json TEXT,image_version INTEGER,attachment_path TEXT,reference_path TEXT,error TEXT);
        CREATE TABLE IF NOT EXISTS image_versions(id TEXT PRIMARY KEY,conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,version INTEGER NOT NULL,path TEXT NOT NULL,kind TEXT NOT NULL,created_at TEXT NOT NULL,parent_version INTEGER,source_message_id TEXT,UNIQUE(conversation_id,version));
        ''')
init_db()

def safe_file(upload: UploadFile, prefix: str) -> Path:
    ext = Path(upload.filename or '').suffix.lower()
    if ext not in ALLOWED: raise HTTPException(400, f'Unsupported file extension: {ext or "none"}')
    path = UPLOADS / f'{prefix}_{uuid.uuid4().hex}{ext}'
    data = upload.file.read()
    if not data: raise HTTPException(400, 'Uploaded file is empty.')
    if len(data) > 25 * 1024 * 1024: raise HTTPException(413, 'Images must be 25 MB or smaller.')
    path.write_bytes(data)
    return path

def rowdict(row): return dict(row) if row else None

def add_message(c, conv_id, role, content, **kwargs):
    mid = uuid.uuid4().hex
    c.execute('INSERT INTO messages(id,conversation_id,role,content,created_at,task_json,image_version,attachment_path,reference_path,error) VALUES(?,?,?,?,?,?,?,?,?,?)',
      (mid,conv_id,role,content,now(),json.dumps(kwargs.get('task')) if kwargs.get('task') is not None else None,kwargs.get('image_version'),kwargs.get('attachment_path'),kwargs.get('reference_path'),kwargs.get('error')))
    return mid

from agent.llm import model_name

@app.get('/health')
def health():
    gpu_url = os.getenv('GPU_BACKEND_URL', '').strip()
    groq_key = os.getenv('GROQ_API_KEY', '').strip()
    is_groq_valid = bool(groq_key and not groq_key.lower().startswith('replace_') and not groq_key.startswith('gsk_Not'))
    is_gpu_valid = bool(gpu_url and 'YOUR_' not in gpu_url and 'your-' not in gpu_url.lower())
    return {
        'status': 'ok',
        'service': 'local-image-agent',
        'database': str(DB_PATH),
        'gpu_backend_configured': is_gpu_valid,
        'groq_configured': is_groq_valid,
        'groq_model': model_name(),
    }



class AgentRunRequest(BaseModel):
    goal: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None

@app.get('/v1/agent/capabilities')
def list_agent_capabilities():
    """List only capabilities actually registered and executable in this server."""
    return {"capabilities": registry.describe()}

@app.post('/v1/agent/plan')
def plan_agent_run(payload: AgentRunRequest):
    """Ask the configured Groq model to create and validate a dynamic task DAG."""
    context = {}
    if payload.conversation_id:
        with db() as c:
            conv = c.execute('SELECT current_image FROM conversations WHERE id=?', (payload.conversation_id,)).fetchone()
            if not conv: raise HTTPException(404, 'Conversation not found.')
            context['current_image_available'] = bool(conv['current_image'] and Path(conv['current_image']).is_file())
    try:
        plan = create_plan(payload.goal, context)
        return {"plan": plan.model_dump(), "available_capabilities": registry.describe()}
    except Exception as exc:
        raise HTTPException(422, f"Planning failed: {type(exc).__name__}: {str(exc)[:400]}")

@app.post('/v1/agent/runs')
def run_agent_plan(payload: AgentRunRequest):
    """Plan, execute, and independently evaluate a goal using registered capabilities."""
    context = {"messages": [{"role":"user", "content":payload.goal}]}
    if payload.conversation_id:
        with db() as c:
            conv = c.execute('SELECT current_image FROM conversations WHERE id=?', (payload.conversation_id,)).fetchone()
            if not conv: raise HTTPException(404, 'Conversation not found.')
            context['current_image'] = conv['current_image']
            rows = c.execute('SELECT role,content FROM messages WHERE conversation_id=? ORDER BY created_at DESC LIMIT 12', (payload.conversation_id,)).fetchall()
            context['messages'] = [{"role":r['role'],"content":r['content']} for r in reversed(rows)] + [{"role":"user","content":payload.goal}]
            context['image_history'] = []
    try:
        plan = create_plan(payload.goal, {"current_image_available": bool(context.get('current_image') and Path(context['current_image']).is_file()), "registered_capabilities": [x['name'] for x in registry.describe()]})
        result = execute_plan(plan, context)
        # If a registered worker produced a new image, only update a conversation after verifying
        # that the output path exists. The existing chat/version workflow remains unchanged.
        if payload.conversation_id:
            output_paths = [r.get('output',{}).get('current_image') for r in result['task_results'].values()]
            output_path = next((x for x in reversed(output_paths) if x and Path(x).is_file()), None)
            if output_path:
                with db() as c:
                    current = c.execute('SELECT current_image FROM conversations WHERE id=?', (payload.conversation_id,)).fetchone()
                    if current and str(Path(output_path).resolve()) != str(Path(current['current_image']).resolve() if current['current_image'] else ''):
                        version = (c.execute('SELECT MAX(version) AS v FROM image_versions WHERE conversation_id=?', (payload.conversation_id,)).fetchone()['v'] or 0) + 1
                        c.execute('INSERT INTO image_versions(id,conversation_id,version,path,kind,created_at,parent_version) VALUES(?,?,?,?,?,?,?)', (uuid.uuid4().hex,payload.conversation_id,version,str(Path(output_path).resolve()),'agent_run',now(),version-1 or None))
                        c.execute('UPDATE conversations SET current_image=?,updated_at=? WHERE id=?',(str(Path(output_path).resolve()),now(),payload.conversation_id))
        return result
    except HTTPException: raise
    except Exception as exc:
        raise HTTPException(500, f"Agent run failed: {type(exc).__name__}: {str(exc)[:400]}")

@app.get('/v1/conversations')
def list_conversations():
    with db() as c: return [dict(r) for r in c.execute('SELECT id,title,created_at,updated_at,current_image FROM conversations ORDER BY updated_at DESC')]

@app.post('/v1/conversations')
def create_conversation(title: str = Form('New chat')):
    cid=uuid.uuid4().hex; t=now(); title=title.strip()[:100] or 'New chat'
    with db() as c: c.execute('INSERT INTO conversations(id,title,created_at,updated_at) VALUES(?,?,?,?)',(cid,title,t,t))
    return {'id':cid,'title':title,'created_at':t,'updated_at':t,'current_image':None}

@app.patch('/v1/conversations/{conversation_id}')
def rename_conversation(conversation_id: str, title: str = Form(...)):
    title=title.strip()[:100]
    if not title: raise HTTPException(400,'Title cannot be empty.')
    with db() as c:
        cur=c.execute('UPDATE conversations SET title=?,updated_at=? WHERE id=?',(title,now(),conversation_id))
        if not cur.rowcount: raise HTTPException(404,'Conversation not found.')
    return {'id':conversation_id,'title':title}

@app.delete('/v1/conversations/{conversation_id}')
def delete_conversation(conversation_id: str):
    with db() as c:
        cur=c.execute('DELETE FROM conversations WHERE id=?',(conversation_id,))
        if not cur.rowcount: raise HTTPException(404,'Conversation not found.')
    return {'deleted':True}

@app.get('/v1/conversations/{conversation_id}')
def get_conversation(conversation_id: str):
    with db() as c:
        conv=c.execute('SELECT * FROM conversations WHERE id=?',(conversation_id,)).fetchone()
        if not conv: raise HTTPException(404,'Conversation not found.')
        messages=[]
        for r in c.execute('SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at',(conversation_id,)):
            d=dict(r); d['task']=json.loads(d.pop('task_json')) if d.get('task_json') else None
            if d.get('attachment_path'): d['attachment_url']='/v1/files/'+Path(d['attachment_path']).name
            if d.get('reference_path'): d['reference_url']='/v1/files/'+Path(d['reference_path']).name
            messages.append(d)
        versions=[dict(r) for r in c.execute('SELECT id,version,path,kind,created_at,parent_version,source_message_id FROM image_versions WHERE conversation_id=? ORDER BY version DESC',(conversation_id,))]
        current_version=next((v['version'] for v in versions if conv['current_image'] and str(Path(v['path']).resolve())==str(Path(conv['current_image']).resolve())),None)
    return {'conversation':dict(conv),'messages':messages,'versions':versions,'current_image_version':current_version,'current_image_url':('/v1/files/'+Path(conv['current_image']).name) if conv['current_image'] else None}

@app.post('/v1/conversations/{conversation_id}/messages')
def send_message(conversation_id: str, request: str = Form(...), image: UploadFile|None=File(None), mask: UploadFile|None=File(None), reference: UploadFile|None=File(None)):
    request=request.strip()
    if not request and not image: raise HTTPException(400,'Enter a message or upload an image.')
    with db() as c:
        conv=c.execute('SELECT * FROM conversations WHERE id=?',(conversation_id,)).fetchone()
        if not conv: raise HTTPException(404,'Conversation not found.')
        previous_image=conv['current_image']
        previous_version=c.execute('SELECT MAX(version) AS v FROM image_versions WHERE conversation_id=?',(conversation_id,)).fetchone()['v'] or 0
        prior_messages=[dict(r) for r in c.execute('SELECT role,content FROM messages WHERE conversation_id=? ORDER BY created_at DESC LIMIT 12',(conversation_id,))]
    image_path=safe_file(image,'source') if image else (Path(previous_image) if previous_image and Path(previous_image).is_file() else None)
    mask_path=safe_file(mask,'mask') if mask else None
    ref_path=safe_file(reference,'reference') if reference else None
    if image_path and not request: request='Please inspect the uploaded image and tell me what you can help edit.'
    display_title=conv['title']
    with db() as c:
        if display_title=='New chat' and request:
            display_title=request[:42]+('…' if len(request)>42 else '')
            c.execute('UPDATE conversations SET title=? WHERE id=?',(display_title,conversation_id))
        user_mid=add_message(c,conversation_id,'user',request,attachment_path=str(image_path) if image else None,reference_path=str(ref_path) if ref_path else None)
    # Persist each newly uploaded source as a version before inference starts.
    # This makes retries possible after detection/inpainting errors and ensures a
    # failed edit never creates a fake generated-image version.
    if image and image_path:
        source_path = str(Path(image_path).resolve())
        with db() as c:
            latest = c.execute('SELECT current_image FROM conversations WHERE id=?', (conversation_id,)).fetchone()
            latest_version = c.execute(
                'SELECT MAX(version) AS v FROM image_versions WHERE conversation_id=?',
                (conversation_id,),
            ).fetchone()['v'] or 0
            if latest and (not latest['current_image'] or str(Path(latest['current_image']).resolve()) != source_path):
                source_version = latest_version + 1
                source_kind = 'original' if latest_version == 0 else 'source'
                c.execute(
                    'INSERT INTO image_versions(id,conversation_id,version,path,kind,created_at,parent_version,source_message_id) VALUES(?,?,?,?,?,?,?,?)',
                    (uuid.uuid4().hex, conversation_id, source_version, source_path, source_kind, now(), latest_version or None, user_mid),
                )
                c.execute(
                    'UPDATE conversations SET current_image=?,updated_at=? WHERE id=?',
                    (source_path, now(), conversation_id),
                )
                previous_image = source_path
                previous_version = source_version
            else:
                previous_image = latest['current_image'] if latest else previous_image
                previous_version = latest_version

    # Include prior conversational context, while preserving the latest source image.
    messages=[{'role':r['role'],'content':r['content']} for r in reversed(prior_messages)]
    messages.append({'role':'user','content':request})
    initial={'messages':messages,'current_image':str(image_path) if image_path else None,'mask_image':str(mask_path) if mask_path else None,'reference_images':[str(ref_path)] if ref_path else [],'image_history':[],'current_task':None,'route':None,'response':None,'error':None}
    # The current backend has no reference-conditioning input yet. Preserve the reference as conversation context,
    # but avoid falsely claiming it was used to condition pixels in the current diffusion pipeline.
    try: result=graph.invoke(initial)
    except Exception as exc:
        msg=f'Agent execution failed: {type(exc).__name__}: {exc}'
        with db() as c: add_message(c,conversation_id,'assistant',msg,error=msg); c.execute('UPDATE conversations SET updated_at=? WHERE id=?',(now(),conversation_id))
        raise HTTPException(500,msg)
    response_text=result.get('response') or 'Request processed.'
    if ref_path and result.get('route')=='image_edit':
        response_text += '\n\nNote: the reference image is saved with this message, but reference-image conditioning is not yet wired into the current GPU model.'
    current=result.get('current_image') or previous_image
    is_new_version=bool(current and Path(current).is_file() and (not previous_image or str(Path(current).resolve())!=str(Path(previous_image).resolve())))
    version=None
    if is_new_version:
        version=previous_version+1
        # First user-provided image is Version 1; generated edits increment it.
        kind='original' if image_path and Path(current).resolve()==Path(image_path).resolve() else 'edit'
        with db() as c:
            c.execute('INSERT INTO image_versions(id,conversation_id,version,path,kind,created_at,parent_version,source_message_id) VALUES(?,?,?,?,?,?,?,?)',(uuid.uuid4().hex,conversation_id,version,str(Path(current).resolve()),kind,now(),previous_version or None,user_mid))
            c.execute('UPDATE conversations SET current_image=?,updated_at=? WHERE id=?',(str(Path(current).resolve()),now(),conversation_id))
    # Only mark a chat message as having created an image version when the
    # operation succeeded. A first uploaded source may be stored as the
    # conversation's original version even if the subsequent edit fails, but
    # that must never be presented as a successful generated edit.
    message_version = version if (version is not None and not result.get('error') and is_new_version and current and image_path and Path(current).resolve() != Path(image_path).resolve()) else None
    with db() as c:
        add_message(c,conversation_id,'assistant',response_text,task={**(result.get('current_task') or {}), '_tool_trace': result.get('tool_trace', [])},image_version=message_version,error=result.get('error'))
        c.execute('UPDATE conversations SET updated_at=? WHERE id=?',(now(),conversation_id))
    return {'conversation_id':conversation_id,'response':response_text,'task':result.get('current_task'),'route':result.get('route'),'error':result.get('error'),'image_version':message_version,'image_url':('/v1/files/'+Path(current).name) if message_version is not None and current else None,'current_image_url':('/v1/files/'+Path(current).name) if current and Path(current).is_file() else None,'tool_trace':result.get('tool_trace',[])}

@app.get('/v1/conversations/{conversation_id}/versions/{version}')
def get_version(conversation_id: str, version: int):
    with db() as c:
        row=c.execute('SELECT path FROM image_versions WHERE conversation_id=? AND version=?',(conversation_id,version)).fetchone()
        if not row: raise HTTPException(404,'Image version not found.')
    return FileResponse(row['path'],media_type='image/png',filename=f'image_v{version}.png')

@app.post('/v1/conversations/{conversation_id}/versions/{version}/restore')
def restore_version(conversation_id: str, version: int):
    with db() as c:
        row=c.execute('SELECT path FROM image_versions WHERE conversation_id=? AND version=?',(conversation_id,version)).fetchone()
        if not row: raise HTTPException(404,'Image version not found.')
        c.execute('UPDATE conversations SET current_image=?,updated_at=? WHERE id=?',(row['path'],now(),conversation_id))
        if not c.execute('SELECT 1 FROM conversations WHERE id=?',(conversation_id,)).fetchone(): raise HTTPException(404,'Conversation not found.')
    return {'restored':True,'version':version,'current_image_url':'/v1/files/'+Path(row['path']).name}

@app.get('/v1/files/{filename}')
def get_file(filename: str):
    """Serve a single image from trusted app storage, including legacy DB paths.

    Never accept a caller-supplied filesystem path here: only the basename is
    used, and candidate paths are resolved and constrained to app storage.
    """
    name = Path(filename).name
    if not name or name != filename or name in {'.', '..'}:
        raise HTTPException(400, 'Invalid filename.')
    roots = [UPLOADS.resolve(), OUTPUTS.resolve()]
    # Include the project-local defaults to support installations whose .env
    # changed APP_DATA_DIR after files were created.
    roots.extend([(PROJECT_ROOT / 'data' / 'uploads').resolve(),
                  (PROJECT_ROOT / 'data' / 'outputs').resolve()])
    seen = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        candidate = (root / name).resolve()
        if candidate.parent == root and candidate.is_file():
            ext = candidate.suffix.lower()
            media = {'.png':'image/png', '.jpg':'image/jpeg', '.jpeg':'image/jpeg',
                     '.webp':'image/webp', '.bmp':'image/bmp'}.get(ext)
            if not media:
                raise HTTPException(415, 'Unsupported file type.')
            return FileResponse(str(candidate), media_type=media, filename=candidate.name)
    raise HTTPException(404, 'File not found in app uploads or outputs.')
