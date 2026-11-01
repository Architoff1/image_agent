from typing import TypedDict, Optional, Any

class Message(TypedDict):
    role: str
    content: str

class Task(TypedDict, total=False):
    intent: str
    operation: str
    description: str
    target: str
    location: str
    reference_required: bool
    reference_indices: list[int]
    constraints: list[str]
    num_inference_steps: int
    guidance_scale: float
    controlnet_conditioning_scale: float
    strength: float
    seed: int

class AgentState(TypedDict, total=False):
    messages: list[Message]
    current_image: Optional[str]
    mask_image: Optional[str]
    reference_images: list[str]
    image_history: list[str]
    current_task: Optional[Task]
    route: Optional[str]
    response: Optional[str]
    error: Optional[str]
    tool_trace: list[dict[str, Any]]
