"""Compatibility wrapper for the remote Kaggle GPU editing backend.

The active edit path is ``image.manipulator.execute_image_edit``. This wrapper
keeps configuration consistent for any older code that still constructs it.
"""
import os


class ImagePipeline:
    def __init__(self, backend_url: str | None = None):
        value = (backend_url or os.getenv("GPU_BACKEND_URL", "")).strip().rstrip("/")
        if value.endswith("/v1/edit"):
            value = value[:-len("/v1/edit")].rstrip("/")
        self.backend_url = value

    @property
    def configured(self) -> bool:
        return bool(self.backend_url)
