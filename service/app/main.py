"""HTTP-слой: API для жюри (раздел 10 ТЗ) + статический UI.

Контракт ТЗ:
  POST /api/v1/analyze        file=<.mov>, X-API-Key -> {"job_id": ...}
  GET  /api/v1/jobs/{job_id}  X-API-Key -> {"status": processing|done|failed, ...}
"""
from __future__ import annotations

import os
import secrets
import shutil
import threading
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .guard import repair
from .jobs import JobStore
from .schemas import JobResult, Violation, VideoInfo

API_KEY = os.environ.get("API_KEY", "")
MAX_BYTES = int(os.environ.get("MAX_UPLOAD_MB", 200)) * 1024 * 1024
ALLOWED_EXT = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}  # ТЗ требует .mov, остальное — бонус

STATIC = Path(__file__).resolve().parents[1] / "static"

app = FastAPI(title="ForteBank — контроль кассовой дисциплины по видео",
              version="1.0.0", docs_url="/api/docs", redoc_url=None)
store = JobStore()


def require_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if not API_KEY:
        raise HTTPException(500, "сервис запущен без API_KEY")
    # Сравнение за постоянное время: сервис выставлен в интернет, и обычное !=
    # прекращает сравнение на первом несовпавшем символе.
    if not (x_api_key and secrets.compare_digest(x_api_key, API_KEY)):
        raise HTTPException(401, "неверный или отсутствующий X-API-Key")


_MODELS_READY = {"caption": False, "detector": False}


@app.on_event("startup")
def _warmup() -> None:
    """Прогрев моделей при старте, а не на первом запросе жюри.

    Загрузка весов занимает 20-30 с. Если делать её лениво, это время съест
    пятиминутный лимит на первое же видео.
    """
    def job() -> None:
        from . import caption, perception, vlm
        try:
            caption.warmup()
            # Вторая модель прогревается до флага готовности: иначе её загрузка
            # (около 20 с) легла бы на первое видео жюри. Ключи health прежние —
            # контракт не меняется; не загрузилась — сервис работает без неё.
            if vlm.ENABLED:
                try:
                    vlm.warmup()
                except Exception as exc:                # noqa: BLE001
                    vlm.ENABLED = False
                    print(f"warmup vlm: {exc}")
            _MODELS_READY["caption"] = True
        except Exception as exc:                        # noqa: BLE001
            print(f"warmup caption: {exc}")
        try:
            perception._load_detr(); _MODELS_READY["detector"] = True
        except Exception as exc:                        # noqa: BLE001
            print(f"warmup detector: {exc}")

    threading.Thread(target=job, daemon=True).start()


@app.get("/api/v1/health")
def health() -> dict:
    return {"status": "ok", "models": dict(_MODELS_READY)}


@app.post("/api/v1/analyze", dependencies=[Depends(require_key)])
async def analyze(file: UploadFile = File(...)) -> JSONResponse:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"ожидается видеофайл {sorted(ALLOWED_EXT)}, получено '{ext}'")

    tmp_dir = Path(tempfile.gettempdir()) / "forte_uploads"
    tmp_dir.mkdir(exist_ok=True)
    dst = tmp_dir / f"{os.urandom(6).hex()}{ext}"

    size = 0
    with dst.open("wb") as fh:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_BYTES:
                fh.close(); dst.unlink(missing_ok=True)
                raise HTTPException(413, f"файл больше {MAX_BYTES // 1024 // 1024} МБ")
            fh.write(chunk)
    if size == 0:
        dst.unlink(missing_ok=True)
        raise HTTPException(400, "пустой файл")

    job = store.submit(str(dst), file.filename or dst.name)
    return JSONResponse({"job_id": job.job_id, "status": job.status})


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_key)],
         response_model=JobResult, response_model_exclude_none=False)
def job_status(job_id: str) -> JobResult:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "задание не найдено")
    out = JobResult(job_id=job.job_id, status=job.status,
                    progress=job.progress, stage=job.stage, error=job.error)
    if job.status == "done" and job.result:
        r = job.result
        # Инварианты контракта перед отдачей: вердикт <-> события, 1-3 кадра, JPEG,
        # таймкоды внутри длительности. Идемпотентно, исправления — в журнал.
        fixes = repair(r)
        if fixes:
            print(f"job {job.job_id}: исправлено перед отдачей: {fixes}")
        out.verdict = r["verdict"]
        # Собираем модели явно, а не кладём словари: иначе pydantic сериализует их
        # «как есть» с предупреждением, и при любом изменении схемы поле может
        # молча потеряться по дороге к жюри.
        out.violations = [Violation.model_validate(v) for v in r["violations"]]
        out.video = VideoInfo.model_validate(r["video"])
        # Остальные служебные поля переносим по списку полей схемы, а не руками.
        # Руками уже потерялось `zone_source`: добавил его в результат конвейера и
        # в схему, а строку копирования здесь не дописал — поле молча не доехало
        # до жюри, поймано на приёмочном прогоне (NIGHT-LOG, шаг 9).
        built = {"job_id", "status", "progress", "stage", "error",
                 "verdict", "violations", "video"}
        for name in JobResult.model_fields:
            if name not in built and name in r:
                setattr(out, name, r[name])
    return out


@app.get("/")
def index() -> FileResponse:
    # no-store на саму страницу: ссылки на статику версионированы хэшем,
    # но сам index.html должен приезжать свежим, иначе браузер держит старые ссылки
    return FileResponse(STATIC / "index.html",
                        headers={"Cache-Control": "no-store, must-revalidate"})


app.mount("/static", StaticFiles(directory=STATIC), name="static")
