from typing import Literal
from pydantic import BaseModel, Field, model_validator


class PlannerOutput(BaseModel):
    intent: Literal["conversation", "image_generation", "image_edit", "image_analysis"]
    operation: Literal["none", "generate", "add", "remove", "modify", "replace", "analyze"]
    description: str = Field(min_length=1, max_length=1200)
    target: str = Field(default="object", max_length=160)
    location: str = Field(default="", max_length=240)
    reference_required: bool = False
    reference_indices: list[int] = Field(default_factory=list, max_length=8)
    constraints: list[str] = Field(default_factory=list, max_length=12)
    # Initial inference settings for FLUX.1-Fill-dev. Groq can revise these through bounded tool calls.
    # FLUX uses much higher guidance_scale (28-50) than SDXL (5-8). strength=1.0 is recommended for inpainting.
    num_inference_steps: int = Field(default=28, ge=1, le=50)
    guidance_scale: float = Field(default=30.0, ge=1.0, le=50.0)
    controlnet_conditioning_scale: float = Field(default=0.0, ge=0.0, le=2.0)  # Not used by FLUX; kept for API compat
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)

    @model_validator(mode="after")
    def validate_intent_operation(self):
        valid = {
            "conversation": {"none"},
            "image_generation": {"generate"},
            "image_edit": {"add", "remove", "modify", "replace"},
            "image_analysis": {"analyze"},
        }
        if self.operation not in valid[self.intent]:
            raise ValueError(f"Invalid combination: {self.intent}/{self.operation}")
        return self
