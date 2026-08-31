# syntax=docker/dockerfile:1
# cdesk-mcp — remote (http/OAuth) transport. Build with `docker compose build`.
FROM python:3.12-slim

# uv for fast, reproducible installs (pulled from the official image).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install into the project venv. README.md is referenced by the build backend;
# src is needed because the project installs itself. --frozen pins uv.lock,
# --no-dev skips pytest/mypy/ruff.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

# http-mode defaults; the tenant URL, public URL and encryption key are supplied
# at run time via the env file. Bind all interfaces so the fronting proxy /
# compose network can reach it.
ENV CDESK_TRANSPORT=http \
    CDESK_HTTP_HOST=0.0.0.0 \
    CDESK_HTTP_PORT=8000

EXPOSE 8000

# Run via the module entrypoint — no console-script / uv needed at runtime.
CMD ["/app/.venv/bin/python", "-m", "cdesk_mcp"]
