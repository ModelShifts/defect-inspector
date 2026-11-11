"""Settings read from environment variables (and a local ``.env`` file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BACKBONE = "timm/wide_resnet50_2.tv_in1k"
DEFAULT_HF_VLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
SMALL_HF_VLM_MODEL = "Qwen/Qwen3-VL-2B-Instruct"


def load_dotenv_if_present(path: str | os.PathLike | None = None) -> None:
    """Load ``.env`` from the working directory without overriding real env vars."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is a core dependency
        return
    load_dotenv(dotenv_path=path or Path.cwd() / ".env", override=False)


def _env(name: str, default: str = "") -> str:
    value = (os.environ.get(name) or "").strip()
    return value or default


@dataclass
class Settings:
    # anomaly detection (always local)
    backbone: str = DEFAULT_BACKBONE
    image_size: int = 256
    coreset_ratio: float = 0.1
    # defect description: api or hf
    backend: str = "hf"
    api_base: str = ""
    api_key: str = ""
    model: str = ""
    hf_vlm_model: str = DEFAULT_HF_VLM_MODEL
    hf_dtype: str = "auto"  # auto (bfloat16 on GPU, float32 on CPU) | float32 | bfloat16
    max_new_tokens: int = 96
    cache_dir: str = ".cache"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv_if_present()
        return cls(
            backbone=_env("MS_BACKBONE", DEFAULT_BACKBONE),
            image_size=int(_env("MS_IMAGE_SIZE", "256") or 256),
            coreset_ratio=float(_env("MS_CORESET_RATIO", "0.1") or 0.1),
            backend=_env("MS_BACKEND", "hf").lower(),
            api_base=_env("MS_API_BASE"),
            api_key=_env("MS_API_KEY"),
            model=_env("MS_MODEL"),
            hf_vlm_model=_env("MS_HF_VLM_MODEL", DEFAULT_HF_VLM_MODEL),
            hf_dtype=_env("MS_HF_DTYPE", "auto").lower(),
            max_new_tokens=int(_env("MS_MAX_NEW_TOKENS", "96") or 96),
            cache_dir=_env("MS_CACHE_DIR", ".cache"),
        )

    @property
    def vlm_name(self) -> str:
        return self.model if self.backend == "api" else self.hf_vlm_model

    def validate_backend(self) -> None:
        """Check the settings needed for defect descriptions (not needed for scoring)."""
        if self.backend not in {"api", "hf"}:
            raise ValueError(f"MS_BACKEND must be 'api' or 'hf', got {self.backend!r}")
        if self.backend == "api":
            missing = [n for n, v in (("MS_API_BASE", self.api_base), ("MS_MODEL", self.model)) if not v]
            if missing:
                raise ValueError(f"MS_BACKEND=api needs {', '.join(missing)} (see .env.example)")
