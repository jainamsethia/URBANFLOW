# UrbanFlow workbench: `docker compose up`, then open the link printed in the log.

# 1. Build the web UI (Node is only needed here, not in the final image).
FROM node:24-slim AS ui
WORKDIR /src/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# 2. Install UrbanFlow with the built UI inside the package.
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
COPY --from=ui /src/src/urbanflow/_frontend ./src/urbanflow/_frontend
RUN pip install . && rm -rf /build

# Your maps, recordings and results live in /data (a volume in docker-compose.yml).
RUN useradd --create-home urbanflow && mkdir /data && chown urbanflow /data
USER urbanflow
WORKDIR /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health')"
# Listening on all interfaces requires an access token: one is generated at start-up and
# printed as a link (#token=...), or set your own with URBANFLOW_API_TOKEN.
CMD ["urbanflow", "serve", "--host", "0.0.0.0", "--port", "8000"]
