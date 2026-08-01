ARG BASE_IMAGE=localhost/cosmos3-rocm-server:7.2.4
FROM ${BASE_IMAGE}

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY static /app/static

ENV PYTHONUNBUFFERED=1 \
    COSMOS_STUDIO_HOST=0.0.0.0 \
    COSMOS_STUDIO_PORT=8000 \
    COSMOS_STUDIO_DATA_DIR=/data \
    COSMOS_STUDIO_OUTPUT_DIR=/outputs \
    COSMOS_STUDIO_STATIC_DIR=/app/static \
    HF_HOME=/models/hf-cache \
    HF_HUB_CACHE=/models/hf-cache/hub \
    TOKENIZERS_PARALLELISM=false

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
