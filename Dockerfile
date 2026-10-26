FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -e ".[dev]" \
    && python -m pytest -q
CMD ["python", "-m", "mp.cli", "--help"]
