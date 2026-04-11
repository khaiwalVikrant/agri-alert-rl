FROM ghcr.io/meta-pytorch/openenv-base:latest AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential curl && \
    rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt ./requirements.txt
COPY pyproject.toml .
COPY models.py .
COPY __init__.py .
COPY client.py .
COPY server/ ./server/
COPY openenv.yaml .

RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv; \
    fi

RUN pip install --no-cache-dir -r requirements.txt

FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

COPY --from=builder /app /app
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11

ENV PYTHONPATH="/app:$PYTHONPATH"

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
