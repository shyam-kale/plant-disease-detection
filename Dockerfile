# ── Spinach Disease Detection — Production Dockerfile ────────────────────────
# Python 3.11.9 on Debian slim (matches .python-version and render.yaml)
FROM python:3.11.9-slim

LABEL maintainer="research-team"
LABEL description="Spinach Plant Disease Detection — Flask API"
LABEL version="1.0.0"

# System dependencies required by Pillow and PyMySQL
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        default-libmysqlclient-dev \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies first (layer cache optimisation)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Create required directories
RUN mkdir -p uploads results/figures models/sklearn \
 && chown -R appuser:appuser /app

USER appuser

# Expose port
EXPOSE 5000

# Gunicorn: 2 workers, 120s timeout (model loading takes time)
CMD ["gunicorn", "app:app", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--timeout", "120", \
     "--log-level", "info", \
     "--access-logfile", "-"]
