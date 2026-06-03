FROM jbarlow83/ocrmypdf

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Bring in the uv binary for fast, reproducible installs into the ocrmypdf venv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV VIRTUAL_ENV=/app/.venv

RUN uv pip install --no-cache torch torchvision --index-url https://download.pytorch.org/whl/cu124

RUN uv pip install --no-cache git+https://github.com/ocrmypdf/OCRmyPDF-EasyOCR.git

RUN mkdir -p /root/.EasyOCR/model && \
    cd /root/.EasyOCR/model && \
    curl -fsSL -O https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/latin_g2.zip && \
    curl -fsSL -O https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip && \
    curl -fsSL -O https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip && \
    unzip '*.zip' && rm -f *.zip
