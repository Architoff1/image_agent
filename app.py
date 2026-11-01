import os, time, threading, urllib.parse, requests, streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from PIL import Image, ImageDraw
from io import BytesIO

# 1. Load local .env if present (local development)
load_dotenv(dotenv_path=Path(__file__).resolve().parent/'.env', override=False)

# 2. Bridge Streamlit Cloud secrets (st.secrets) to os.environ so FastAPI & agent modules have access
def sync_secrets_to_env():
    try:
        if hasattr(st, "secrets"):
            def _extract(obj):
                items = {}
                try:
                    for k, v in obj.items():
                        if isinstance(v, dict) or hasattr(v, "items"):
                            items.update(_extract(v))
                        elif isinstance(v, (str, int, float, bool)):
                            items[k] = str(v)
                except Exception:
                    pass
                return items

            for k, v in _extract(st.secrets).items():
                os.environ.setdefault(k, v)
                os.environ.setdefault(k.upper(), v)
    except Exception:
        pass

    # Normalize aliases for LOCAL_AGENT_API_URL
    aliases = [
        "LOCAL_AGENT_API", "local_agent_api", "local_AGENT_API",
        "AGENT_API", "agent_api", "LOCAL_API_URL", "local_api_url", "LOCAL_AGENT_API_URL"
    ]
    for a in aliases:
        if a in os.environ:
            val = os.environ[a].strip()
            if val and val.lower() not in ("none", "null"):
                os.environ["LOCAL_AGENT_API_URL"] = val
                break

    if not os.getenv("LOCAL_AGENT_API_URL", "").strip():
        os.environ["LOCAL_AGENT_API_URL"] = "http://127.0.0.1:8000"

    # Normalize Groq and GPU environment aliases
    for groq_alias in ["groq_api_key", "GROQ_KEY", "groq_key"]:
        if groq_alias in os.environ and "GROQ_API_KEY" not in os.environ:
            os.environ["GROQ_API_KEY"] = os.environ[groq_alias]
    for gpu_url_alias in ["gpu_backend_url", "GPU_URL", "gpu_url"]:
        if gpu_url_alias in os.environ and "GPU_BACKEND_URL" not in os.environ:
            os.environ["GPU_BACKEND_URL"] = os.environ[gpu_url_alias]
    for gpu_token_alias in ["gpu_backend_token", "GPU_TOKEN", "gpu_token"]:
        if gpu_token_alias in os.environ and "GPU_BACKEND_TOKEN" not in os.environ:
            os.environ["GPU_BACKEND_TOKEN"] = os.environ[gpu_token_alias]

sync_secrets_to_env()

API = os.getenv("LOCAL_AGENT_API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = max(60, min(int(os.getenv("API_REQUEST_TIMEOUT", "1200")), 1800))

@st.cache_resource
def start_embedded_api_server(target_url: str):
    """If target_url is localhost/loopback and not responding, run FastAPI in a background daemon thread."""
    parsed = urllib.parse.urlparse(target_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000
    is_local = host in ("127.0.0.1", "localhost", "0.0.0.0")

    if not is_local:
        return True, "external"

    # Check if a server is already running and healthy
    try:
        r = requests.get(f"http://127.0.0.1:{port}/health", timeout=1.5)
        if r.status_code == 200:
            return True, "already_running"
    except Exception:
        pass

    # Start FastAPI with uvicorn in a daemon thread
    def _run_server():
        import uvicorn
        from api.server import app as fastapi_app
        uvicorn.run(fastapi_app, host="127.0.0.1", port=port, log_level="warning")

    server_thread = threading.Thread(target=_run_server, daemon=True, name="fastapi_backend")
    server_thread.start()

    # Wait up to 10 seconds for the server to become healthy
    start_time = time.time()
    while time.time() - start_time < 10:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=1)
            if r.status_code == 200:
                return True, "started"
        except Exception:
            time.sleep(0.4)

    return False, "failed_to_start"

backend_ready, backend_mode = start_embedded_api_server(API)

st.set_page_config(page_title='Image Agent',page_icon='🖼️',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
.block-container{padding-top:1.3rem;max-width:1500px}.stChatMessage{border-radius:14px}.current-chip{display:inline-block;border:1px solid #555;border-radius:10px;padding:5px 10px;font-size:.85rem}.sidebar-title{font-size:1.15rem;font-weight:700}
</style>''',unsafe_allow_html=True)

def get_auth_headers():
    return {}

def api_get(path, **kwargs):
    kwargs.setdefault("timeout", 20)
    response = requests.get(API + path, **kwargs)
    response.raise_for_status()
    return response

def api_post(path, **kwargs):
    kwargs.setdefault("timeout", TIMEOUT)
    response = requests.post(API + path, **kwargs)
    response.raise_for_status()
    return response

def refresh_chats():
    try: 
        st.session_state.chats = api_get('/v1/conversations').json()
    except requests.RequestException as e: 
        st.session_state.api_error = str(e)
        st.session_state.chats = []

def open_chat(cid):
    try:
        data = api_get(f'/v1/conversations/{cid}').json()
        st.session_state.chat_id = cid
        st.session_state.chat_data = data
    except requests.RequestException as e:
        st.session_state.api_error = str(e)

if 'chats' not in st.session_state: refresh_chats()
if 'chat_id' not in st.session_state: st.session_state.chat_id=None
if 'chat_data' not in st.session_state: st.session_state.chat_data=None
if 'show_preview' not in st.session_state: st.session_state.show_preview=False
if 'show_versions' not in st.session_state: st.session_state.show_versions=False

with st.sidebar:
    st.markdown('<div class="sidebar-title">🖼️ Image Agent</div>',unsafe_allow_html=True)
    if st.button('＋  New chat',width='stretch',type='primary'):
        try:
            r=api_post('/v1/conversations',data={'title':'New chat'}).json(); refresh_chats(); open_chat(r['id']); st.rerun()
        except requests.RequestException as e: st.error(f'Could not create chat: {e}')
    st.divider(); st.caption('YOUR CHATS')
    for chat in st.session_state.chats:
        label=('● ' if chat['id']==st.session_state.chat_id else '')+(chat.get('title') or 'New chat')
        if st.button(label[:48],key='chat_'+chat['id'],width='stretch'): open_chat(chat['id']); st.rerun()
    st.divider()
    
    # Backend & Configuration diagnostics
    if not backend_ready:
        st.error(f'⚠️ Backend API offline at {API}.')
    else:
        try:
            health = api_get('/health', timeout=3).json()
            if backend_mode in ('started', 'already_running'):
                st.success('🟢 Local API connected (embedded)')
            else:
                st.success('🟢 API connected (external)')
            
            if health.get('groq_configured'):
                st.caption(f"🤖 Groq model: `{health.get('groq_model')}`")
            else:
                st.warning('⚠️ `GROQ_API_KEY` missing. Add it in Streamlit Secrets.')

            if health.get('gpu_backend_configured'):
                st.caption('⚡ Kaggle GPU backend connected')
            else:
                st.warning('⚠️ `GPU_BACKEND_URL` missing. Set in Streamlit Secrets after starting Kaggle.')
        except requests.RequestException:
            st.error('Backend API offline. Check server logs.')
"""
    with st.expander('🔑 Streamlit Cloud Secrets Guide'):
        st.markdown('''
To configure secrets on **Streamlit Community Cloud**, go to your app settings → **Secrets** and paste:
```toml
GROQ_API_KEY = "gsk_..."
GROQ_MODEL = "llama-3.3-70b-versatile"
GPU_BACKEND_URL = "https://your-kaggle-ngrok-url.ngrok-free.dev"
GPU_BACKEND_TOKEN = "your-kaggle-secret-token"

# Optional: defaults to http://127.0.0.1:8000
# The FastAPI server runs automatically inside Streamlit!
LOCAL_AGENT_API_URL = "http://127.0.0.1:8000"
```
''')
"""
    if st.session_state.chat_id:
        with st.expander('Conversation settings'):
            new_title=st.text_input('Rename chat',value=(st.session_state.chat_data or {}).get('conversation',{}).get('title',''))
            if st.button('Save title'):
                try: requests.patch(API+f"/v1/conversations/{st.session_state.chat_id}",data={'title':new_title},headers=get_auth_headers(),timeout=10).raise_for_status(); refresh_chats(); open_chat(st.session_state.chat_id); st.rerun()
                except requests.RequestException as e: st.error(str(e))
            if st.button('Delete conversation',type='secondary'):
                try: api_delete=requests.delete(API+f"/v1/conversations/{st.session_state.chat_id}",headers=get_auth_headers(),timeout=10); api_delete.raise_for_status(); st.session_state.chat_id=None; st.session_state.chat_data=None; refresh_chats(); st.rerun()
                except requests.RequestException as e: st.error(str(e))

if st.session_state.chat_id is None and st.session_state.chats:
    try: open_chat(st.session_state.chats[0]['id'])
    except Exception: pass
if st.session_state.chat_id is None:
    st.title('Your image editor, in a conversation')
    st.write('Upload an image, describe the change, and keep refining it through a saved conversation.')
    if st.button('Start a new chat',type='primary'):
        try: r=api_post('/v1/conversations',data={'title':'New chat'}).json(); refresh_chats(); open_chat(r['id']); st.rerun()
        except requests.RequestException as e: st.error(str(e))
    st.stop()

try:
    if st.session_state.chat_id:
        open_chat(st.session_state.chat_id)
except requests.RequestException as e:
    st.error(f'Could not load this conversation: {e}'); st.stop()

data=st.session_state.chat_data
if not data or 'conversation' not in data:
    st.info('Please select or start a chat from the sidebar.')
    st.stop()
conv=data['conversation']; versions=data['versions']; current_url=data.get('current_image_url')
head_left,head_right=st.columns([5,1])
with head_left: st.subheader(conv.get('title') or 'New chat')
with head_right:
    if current_url:
        if st.button(f'🖼️ V{data.get("current_image_version") or max((v["version"] for v in versions),default=1)} · Current image',width='stretch'): st.session_state.show_preview=True; st.rerun()
    else: st.caption('No current image')
if st.session_state.show_preview and current_url:
    with st.container(border=True):
        x1,x2=st.columns([5,1])
        with x1: st.markdown('### Current image')
        with x2:
            if st.button('✕ Close',width='stretch'): st.session_state.show_preview=False; st.rerun()
        try:
            img=api_get(current_url).content; st.image(img,width='stretch')
            st.download_button('⬇ Download current image',data=img,file_name=f"current_v{data.get('current_image_version') or max((v['version'] for v in versions),default=1)}.png",mime='image/png',width='stretch')
        except requests.RequestException as e: st.error(f'Image preview failed: {e}')

for msg in data['messages']:
    with st.chat_message('user' if msg['role']=='user' else 'assistant'):
        st.markdown(msg['content'] or ('Uploaded an image.' if msg.get('attachment_path') else ''))
        if msg.get('attachment_url'):
            try: st.image(api_get(msg['attachment_url']).content,caption='Image attached to this message',width=360)
            except requests.RequestException: st.caption('Attachment unavailable')
        if msg.get('reference_url'):
            with st.expander('Reference image'):
                try: st.image(api_get(msg['reference_url']).content,width=300)
                except requests.RequestException: st.caption('Reference unavailable')
        if msg.get('image_version'):
            st.caption(f"🖼️ Created image version V{msg['image_version']}")
        if msg.get('task'):
            with st.expander('Edit plan'): st.json(msg['task'])

if versions:
    with st.expander(f'Image history · {len(versions)} version(s)'):
        for v in versions:
            vc1,vc2,vc3=st.columns([1,4,2])
            with vc1: st.markdown(f"**V{v['version']}**")
            with vc2: st.caption(f"{v['kind']} · {v['created_at'][:19].replace('T',' ')}")
            with vc3:
                try:
                    image_bytes=api_get(f"/v1/conversations/{st.session_state.chat_id}/versions/{v['version']}").content
                    st.download_button('Download',data=image_bytes,file_name=f"image_v{v['version']}.png",mime='image/png',key=f"dl_{v['id']}")
                except requests.RequestException: st.caption('Unavailable')
                if st.button('Restore',key=f"restore_{v['id']}"):
                    try: api_post(f"/v1/conversations/{st.session_state.chat_id}/versions/{v['version']}/restore"); open_chat(st.session_state.chat_id); st.success(f"Current image set to V{v['version']}"); st.rerun()
                    except requests.RequestException as e: st.error(str(e))

st.divider()
with st.expander('Attach images or optional mask',expanded=False):
    uploaded_image=st.file_uploader('Source image (optional after first upload)',type=['png','jpg','jpeg','webp','bmp'],key='source_upload')
    reference_image=st.file_uploader('Reference image (optional)',type=['png','jpg','jpeg','webp','bmp'],key='reference_upload')
    mask_file=st.file_uploader('Editable mask (optional; white = edit, black = preserve)',type=['png','jpg','jpeg','webp','bmp'],key='mask_upload')
    st.caption('Mask is optional: without one, the Kaggle GPU backend uses Grounding DINO + SAM. White pixels are editable; black pixels are preserved. Reference images are stored but are not used to condition diffusion.')
    sunglasses_quick_fix = st.checkbox('Use sunglasses mask preset for the supplied portrait (fast path)', value=False, help='For the exact 413×531 sample portrait shared in this conversation. For other photos, upload a suitable mask or use automatic masking.')
if uploaded_image: st.image(uploaded_image,caption='New source image',width=280)
if reference_image: st.image(reference_image,caption='Reference image',width=220)
if mask_file: st.image(mask_file,caption='Editable mask',width=220)
with st.form('chat_form',clear_on_submit=True):
    prompt=st.text_area('Message',placeholder='Describe the change… Example: Add realistic black sunglasses across both eyes. Preserve identity, hair, beard, clothes, background, and all pixels outside the mask.',height=90,label_visibility='collapsed')
    submitted=st.form_submit_button('Send ↗',type='primary',width='stretch')
if submitted:
    files={}
    if uploaded_image: files['image']=(uploaded_image.name,uploaded_image.getvalue(),uploaded_image.type or 'application/octet-stream')
    if reference_image: files['reference']=(reference_image.name,reference_image.getvalue(),reference_image.type or 'application/octet-stream')
    if mask_file: files['mask']=(mask_file.name,mask_file.getvalue(),mask_file.type or 'application/octet-stream')
    elif sunglasses_quick_fix:
        # Portrait-specific mask: deliberately editable around both eyes and nose bridge.
        # It is a convenience path for the sample image, not a general face detector.
        from PIL import ImageDraw
        preset = Image.new('L', (413, 531), 0)
        draw = ImageDraw.Draw(preset)
        draw.rounded_rectangle((108, 158, 305, 238), radius=20, fill=255)
        draw.polygon([(108, 181), (305, 181), (295, 225), (120, 225)], fill=255)
        buf = BytesIO(); preset.save(buf, format='PNG')
        files['mask'] = ('sunglasses_edit_mask.png', buf.getvalue(), 'image/png')
    try:
        with st.spinner('Thinking and processing the request…'):
            response=api_post(f"/v1/conversations/{st.session_state.chat_id}/messages",data={'request':prompt},files=files or None)
        result=response.json(); refresh_chats(); open_chat(st.session_state.chat_id)
        if result.get('error'): st.warning(result['response'])
        else: st.success(result['response'])
        st.rerun()
    except requests.HTTPError as e:
        try: st.error(f"Request failed ({e.response.status_code}): {e.response.text}")
        except Exception: st.error(str(e))
    except requests.RequestException as e: st.error(f'Could not reach the API: {e}')
