FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE alembic.ini ./
COPY src ./src
COPY migrations ./migrations
COPY scripts ./scripts

RUN pip install .

CMD ["python", "-m", "leo", "slack"]
