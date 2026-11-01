# Capability implementation and boundaries

The registry exposes handlers that perform real work, not simulated outputs:

- `image.inspect`: decodes, verifies, and analyzes the current image using Pillow (dimensions, exposure, contrast, color tone) and OpenCV (facial landmark detection).
- `image.edit`: invokes the bounded image-edit controller and configured remote SDXL/ControlNet GPU backend.
- `conversation.respond`: calls the configured Groq model.
- `web.search`: searches the public web with dual engine: Tavily API (when configured with TAVILY_API_KEY) with automatic fallback to DuckDuckGo HTML search (using BeautifulSoup) requiring zero external keys.
- `workspace.write_file`: writes UTF-8 bytes within `AGENT_WORKSPACE_DIR` (default `./workspace`), prevents path traversal, and verifies contents by readback. It does not execute code.
- `document.create_markdown`: creates a Markdown document inside that workspace and verifies the written bytes.

## Deliberate safety boundaries

There is no arbitrary shell/Python execution, unrestricted filesystem access, or authenticated browser session. These are not silently simulated. To add code execution safely, deploy a separately isolated runner (container/VM, no host mounts, network policy, resource/time limits, and explicit authorization) and register it as a distinct capability. Web search uses public results only; it is not a browser automation session.

The planner can only select registered capabilities. All task outputs are actual handler results. Overall success is blocked if any required task is missing, failed, or blocked, even if the LLM evaluator says otherwise. External live behavior still requires integration testing with valid credentials/network.
