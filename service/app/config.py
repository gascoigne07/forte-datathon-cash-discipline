"""Значения по умолчанию, зависящие от устройства.

На CUDA (боевой GPU-сервер) значения по умолчанию совпадают с юнитом forte — сервис,
запущенный из чистого клона `./run.sh` без переменных, работает в той же
конфигурации, в которой измерено качество. На Apple M4 / CPU остаются прежние
значения: второй описатель там не помещается, а два параллельных анализа на одном
MPS-устройстве приводили к убийству процесса по памяти.
"""
from __future__ import annotations

import os


def _cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:                                   # noqa: BLE001
        return False


ON_CUDA = _cuda()


def env(name: str, gpu, cpu) -> str:
    """Переменная окружения, иначе значение по умолчанию для этого устройства."""
    return os.environ.get(name, str(gpu if ON_CUDA else cpu))
