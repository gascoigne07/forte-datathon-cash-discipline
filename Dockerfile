# Образ для переноса во внутренний контур банка (требование переносимости, раздел 7 ТЗ).
# Локальная разработка идёт нативно на Apple Silicon (MPS), этот образ — для Linux.
# Веса кладутся внутрь образа, чтобы в контуре без интернета ничего не скачивалось.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    PYTHONPATH=/app/service

# ffmpeg/libgl нужны OpenCV для декодирования .mov
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Веса качаем на этапе сборки — в рантайме сеть не нужна
RUN HF_HUB_OFFLINE=0 python -c "\
from huggingface_hub import snapshot_download as d; \
d('florence-community/Florence-2-large'); d('PekingU/rtdetr_v2_r50vd')"

COPY service/ ./service/
COPY kb/ ./kb/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health').status==200 else 1)"

# API_KEY обязателен: сервис без ключа отвечает 500 на защищённые методы
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-keep-alive", "75"]
