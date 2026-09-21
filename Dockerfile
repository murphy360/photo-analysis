FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

# libgl1/libglib2.0-0: runtime deps of opencv-python-headless (pulled in by ultralytics),
# even the headless wheel dlopens libGL at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Dependencies in their own layer so code-only changes don't reinstall everything.
COPY pyproject.toml uv.lock* ./
RUN uv sync

# Baked into the image so the triage model never needs network access at runtime/test
# time (mirrors how trivia_service bakes its sentence-transformers embedding model).
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

COPY app ./app
COPY alembic.ini ./alembic.ini
COPY alembic ./alembic
COPY tests ./tests

# docker-compose mounts ./data over this, but the image should still start
# (with ephemeral storage) if run standalone without that volume.
RUN mkdir -p /app/data/media

EXPOSE 8000

# alembic upgrade head is the real schema migration path — init_db()'s
# create_all (app/core/db.py) only ever creates tables that don't exist yet,
# so it silently no-ops on an existing database missing a column a newer
# model added. `exec` hands off PID 1 to uvicorn so `docker stop` signals it
# directly instead of a wrapper shell.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
