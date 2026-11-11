"""Result types (pydantic), shared by the CLI, the web demo and the exports."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Region(BaseModel):
    id: int
    bbox: list[int] = Field(description="x0, y0, x1, y1 in pixels of the input image (x1, y1 exclusive)")
    area_px: int
    peak: float = Field(description="highest heatmap value in the region, as a multiple of the threshold")
    label: str | None = None
    description: str | None = None


class Inspection(BaseModel):
    file: str
    width: int
    height: int
    score: float = Field(description="image score as a multiple of the threshold; above 1 means flagged")
    raw_score: float
    threshold: float
    sensitivity: float = 1.0
    defect: bool
    regions: list[Region] = []
    backbone: str
    describer: str | None = None
    heatmap_png: str | None = None
