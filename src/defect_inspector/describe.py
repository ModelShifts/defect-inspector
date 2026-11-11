"""Describe and label a flagged region with a vision-language model.

``api``: any OpenAI-compatible ``/v1/chat/completions`` endpoint that accepts images.
``hf``:  a local ``transformers`` vision-language model (default Qwen/Qwen2.5-VL-7B-Instruct;
         Qwen/Qwen3-VL-2B-Instruct with MS_HF_DTYPE=bfloat16 is the CPU option).

Both expose ``describe(image) -> (label, description)``. Decoding is greedy.
"""

from __future__ import annotations

import base64
import io
import json
import re
from typing import Any, Protocol

from PIL import Image

from .config import Settings

LABELS = ("scratch", "dent", "spot", "crack", "other")

PROMPT = (
    "This is a close-up of an area of a manufactured surface that an anomaly detector flagged. "
    "What kind of surface defect is visible in the centre of the image? "
    "Choose one label: scratch (a thin line), dent (a small round depression with shading), "
    "spot (a stain or blot of a different colour), crack, or other. "
    'Reply with JSON only: {"label": "<one label>", "description": "<one short sentence about what you see>"}.'

)


class VisionModel(Protocol):
    name: str

    def ask(self, image: Image.Image, prompt: str) -> str: ...


def _data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class APIVision:
    def __init__(self, settings: Settings):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("MS_BACKEND=api needs the 'api' extra: pip install -e '.[api]'") from exc
        self.client = OpenAI(base_url=settings.api_base, api_key=settings.api_key or "not-needed")
        self.name = settings.model
        self.max_new_tokens = settings.max_new_tokens

    def ask(self, image: Image.Image, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _data_url(image)}},
                {"type": "text", "text": prompt},
            ]}],
            temperature=0,
            max_tokens=self.max_new_tokens,
        )
        return response.choices[0].message.content or ""


class HFVision:
    def __init__(self, settings: Settings):
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("MS_BACKEND=hf needs the 'hf' extra: pip install -e '.[hf]'") from exc
        self._torch = torch
        self.name = settings.hf_vlm_model
        self.max_new_tokens = settings.max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if settings.hf_dtype in {"bfloat16", "float32", "float16"}:
            dtype = getattr(torch, settings.hf_dtype)
        else:
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(self.name)
        self.model = AutoModelForImageTextToText.from_pretrained(self.name, dtype=dtype).to(self.device).eval()

    def ask(self, image: Image.Image, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[[image.convert("RGB")]], return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


_CACHE: dict[tuple, Any] = {}


def get_vision_model(settings: Settings) -> VisionModel:
    settings.validate_backend()
    key = (settings.backend, settings.vlm_name, settings.api_base)
    if key not in _CACHE:
        _CACHE[key] = APIVision(settings) if settings.backend == "api" else HFVision(settings)
    return _CACHE[key]


def parse_reply(text: str) -> tuple[str, str]:
    """Pull ``(label, description)`` out of a model reply. Falls back to keyword matching."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", cleaned, re.S)
    label, description = None, None
    if match:
        try:
            obj = json.loads(match.group(0))
            label = str(obj.get("label", "")).strip().lower() or None
            description = str(obj.get("description", "")).strip() or None
        except (json.JSONDecodeError, AttributeError):
            pass
    if label not in LABELS:
        lowered = cleaned.lower()
        found = [(lowered.find(w), w) for w in LABELS if w in lowered]
        label = min(found)[1] if found else "other"
    if not description:
        description = re.sub(r"\s+", " ", re.sub(r"\{.*\}", "", cleaned, flags=re.S)).strip() or "no description"
    return label, description[:240]


def describe(image: Image.Image, model: VisionModel) -> tuple[str, str]:
    return parse_reply(model.ask(image, PROMPT))
