"""Очередь задач в процессе: POST кладёт задание, воркер обрабатывает по одному.

Один воркер намеренно: VLM занимает память целиком, параллельный прогон двух
видео на 16 ГБ привёл бы к свопу и срыву 5-минутного лимита.
"""
from __future__ import annotations

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


class JobStore:
    def __init__(self, workers: int = 1):
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

    def _loop(self) -> None:
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

                job.result = analyze(job.path, progress)
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
