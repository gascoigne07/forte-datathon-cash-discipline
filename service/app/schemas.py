"""Схемы ответов API. Формат зафиксирован ТЗ ForteBank, раздел 10."""
from typing import Literal
from pydantic import BaseModel, Field


class EvidenceFrame(BaseModel):
    timestamp_sec: float
    image_base64: str


class RuleRef(BaseModel):
    doc_title: str
    doc_requisites: str
    clause: str | None = None
    quote: str | None = None
    url: str | None = None
    verified: bool = True
    note: str | None = None


class Violation(BaseModel):
    type: str
    timecode_sec: float
    description: str
    evidence_frames: list[EvidenceFrame] = Field(default_factory=list)
    # поля сверх схемы ТЗ — для UI и прозрачности, жюри их может игнорировать
    confidence: float | None = None
    rule: RuleRef | None = None
    window_sec: list[float] | None = None


class VideoInfo(BaseModel):
    duration_sec: float
    fps: float


class JobResult(BaseModel):
    job_id: str
    status: Literal["processing", "done", "failed"]
    verdict: Literal["violation", "no_violation"] | None = None
    violations: list[Violation] = Field(default_factory=list)
    # служебные поля
    progress: float = 0.0
    stage: str | None = None
    video: VideoInfo | None = None
    error: str | None = None
    processing_sec: float | None = None
    # Истина, если модель успела посмотреть слишком мало кадров (жёсткий лимит
    # времени). Вердикт в этом случае недостоверен, и это надо видеть проверяющему.
    low_coverage: bool | None = None
    frames_analyzed: int | None = None
