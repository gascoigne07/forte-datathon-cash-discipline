"""Очередь задач в процессе: POST кладёт задание, воркер обрабатывает по одному.

Один воркер намеренно: VLM занимает память целиком, параллельный прогон двух
видео на 16 ГБ привёл бы к свопу и срыву 5-минутного лимита.
"""
from __future__ import annotations

import os
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Job:
    job_id: str
    path: str
    filename: str
    status: str = "processing"
    progress: float = 0.0
    stage: str = "в очереди"
    result: dict | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)


# Лимит ТЗ — 5 минут на видео, и жюри меряет его от отправки, а не от момента,
# когда воркер взял задание из очереди. Поэтому дедлайн абсолютный.
DEADLINE_SEC = float(os.environ.get("JOB_DEADLINE_SEC", 285))


# Воркер один по умолчанию, и это замер, а не осторожность: на Apple M4 обе модели
# живут на одном MPS-устройстве, два параллельных анализа втрое замедляют друг друга
# и трижды приводили к убийству процесса по памяти. На машине с полноценным GPU
# имеет смысл поднять: WORKERS=2..4.
from .config import env  # noqa: E402

DEFAULT_WORKERS = int(env("WORKERS", 2, 1))


class JobStore:
    def __init__(self, workers: int = DEFAULT_WORKERS):
        self._jobs: dict[str, Job] = {}
        self._q: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.Lock()
        self._threads = [threading.Thread(target=self._loop, daemon=True)
                         for _ in range(workers)]
        for t in self._threads:
            t.start()

    def submit(self, path: str, filename: str) -> Job:
        jid = uuid.uuid4().hex[:8]
        job = Job(job_id=jid, path=path, filename=filename)
        with self._lock:
            self._jobs[jid] = job
        self._q.put(jid)
        return job

    def get(self, jid: str) -> Job | None:
        with self._lock:
            return self._jobs.get(jid)

    def pending(self) -> int:
        """Сколько заданий ждёт очереди. Жюри прогоняет 5+ видео подряд, и если
        каждое обрабатывать по максимуму, последнее дождётся своего лимита 5 минут
        уже в очереди. Пайплайн уменьшает число разбираемых кадров, когда очередь не пуста."""
        return self._q.qsize()

    def _loop(self) -> None:
        from .guard import no_network
        from .pipeline import analyze
        while True:
            jid = self._q.get()
            job = self.get(jid)
            if job is None:
                continue
            try:
                job.stage = "запуск"

                def progress(p: float, s: str, _j=job) -> None:
                    _j.progress = round(min(max(p, 0.0), 1.0), 3)
                    _j.stage = s

                with no_network():
                    job.result = analyze(job.path, progress, queued=self.pending(),
                                         deadline_ts=job.created + DEADLINE_SEC)
                job.status = "done"
                job.progress = 1.0
                job.stage = "готово"
            except Exception as exc:                      # noqa: BLE001
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.stage = "ошибка"
                traceback.print_exc()
            finally:
                try:
                    Path(job.path).unlink(missing_ok=True)   # видео жюри не храним
                except OSError:
                    pass
                self._q.task_done()
