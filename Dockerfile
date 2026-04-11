FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

# Copy all project files from root context
COPY server/requirements.txt ./requirements.txt
COPY pyproject.toml .
COPY models.py .
COPY __init__.py .
COPY client.py .
COPY openenv.yaml .
COPY server/ ./server/

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH="/app:$PYTHONPATH"
ENV ENABLE_WEB_INTERFACE="true"

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
