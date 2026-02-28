# ── builder ──────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── runtime ──────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="SELF-OS"
LABEL org.opencontainers.image.description="Personal OS based on Knowledge Graph + LLM"

# Non-root user
RUN useradd --create-home --shell /bin/bash selfos

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Ensure data directory exists and is writable
RUN mkdir -p data && chown -R selfos:selfos /app

USER selfos

ENV SELFOS_USE_LLM=1 \
    LOG_LEVEL=INFO \
    DB_PATH=/app/data/self_os.db

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import core.graph.storage; print('ok')" || exit 1

CMD ["python", "main.py"]
