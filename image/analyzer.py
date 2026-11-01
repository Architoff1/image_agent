"""Local image analyzer providing grounded dimensional, perceptual, and landmark analysis.

Uses Pillow for dimension/color analysis and OpenCV Haar cascades for facial
landmark detection. Does not hallucinate or simulate metadata.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from PIL import Image, ImageStat


def analyze_image(image_path: str | Path) -> dict[str, Any]:
    """Inspect an image and return concrete, measurable characteristics."""
    path = Path(image_path)
    if not path.is_file():
        return {"ok": False, "error": f"Image file not found: {path.name}"}

    try:
        file_size = path.stat().st_size
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            width, height = img.size
            img_format = img.format or "UNKNOWN"
            mode = img.mode
            aspect_ratio = round(width / max(1, height), 3)

            # Perceptual statistics
            rgb = img.convert("RGB")
            stat = ImageStat.Stat(rgb)
            mean_r, mean_g, mean_b = [round(x, 1) for x in stat.mean[:3]]
            # Perceived luminance (ITU-R BT.601)
            mean_luminance = round(0.299 * mean_r + 0.587 * mean_g + 0.114 * mean_b, 1)
            contrast_std = round(sum(stat.stddev[:3]) / 3.0, 1) if stat.stddev else 0.0

            if mean_luminance < 60:
                lighting = "dark / low-key"
            elif mean_luminance > 195:
                lighting = "bright / high-key"
            else:
                lighting = "balanced / normal exposure"

            # Dominant tone
            if mean_r > mean_b + 15:
                tone = "warm"
            elif mean_b > mean_r + 15:
                tone = "cool"
            else:
                tone = "neutral"

        # Landmark detection via OpenCV if available
        detected_faces = []
        try:
            import cv2
            import numpy as np
            gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            for (fx, fy, fw, fh) in faces:
                detected_faces.append({
                    "box": [int(fx), int(fy), int(fw), int(fh)],
                    "relative_area": round((fw * fh) / (width * height), 3),
                })
        except Exception:
            pass

        orientation = "square" if aspect_ratio == 1.0 else ("landscape" if aspect_ratio > 1.0 else "portrait")

        summary = (
            f"{width}×{height} ({orientation}, aspect ratio {aspect_ratio}), {img_format} {mode}, "
            f"{round(file_size / 1024, 1)} KB. Exposure: {lighting} (mean brightness {mean_luminance}/255, "
            f"contrast std {contrast_std}). Color tone: {tone}."
        )
        if detected_faces:
            summary += f" Detected {len(detected_faces)} frontal face(s)."

        return {
            "ok": True,
            "path": str(path.resolve()),
            "filename": path.name,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "orientation": orientation,
            "format": img_format,
            "mode": mode,
            "file_size_bytes": file_size,
            "mean_rgb": [mean_r, mean_g, mean_b],
            "luminance": mean_luminance,
            "contrast_std": contrast_std,
            "exposure": lighting,
            "color_tone": tone,
            "faces_detected": len(detected_faces),
            "face_boxes": [f["box"] for f in detected_faces],
            "summary": summary,
        }
    except Exception as exc:
        return {"ok": False, "error": f"Image analysis failed ({type(exc).__name__}): {exc}"}
