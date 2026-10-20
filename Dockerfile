# ── Stage 1: build the React console ─────────────────────────────────────────
FROM node:22-slim AS react-build
WORKDIR /web
COPY frontend-react/package.json frontend-react/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY frontend-react/ ./
RUN npm run build

# ── Stage 2: Python runtime ──────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /srv

# System deps for psycopg + healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend
COPY --from=react-build /web/dist ./frontend-react/dist

ENV PYTHONPATH=/srv/backend \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "app.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
