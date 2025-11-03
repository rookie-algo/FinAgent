FROM python:3.13-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install deps
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-cache

COPY . .

EXPOSE 8000

# Make bootstrap executable
RUN chmod +x run.py

# Use bootstrap as entrypoint
CMD ["./run.py"]