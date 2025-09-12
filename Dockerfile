FROM jbarlow83/ocrmypdf

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Seed pip into the ocrmypdf venv
RUN set -eux; \
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py; \
    /app/.venv/bin/python /tmp/get-pip.py; \
    /app/.venv/bin/pip --version; \
    rm -f /tmp/get-pip.py

RUN /app/.venv/bin/pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124

RUN /app/.venv/bin/pip install --no-cache-dir git+https://github.com/ocrmypdf/OCRmyPDF-EasyOCR.git

RUN mkdir -p /root/.EasyOCR/model && \
    cd /root/.EasyOCR/model && \
    curl -fsSL -O https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/latin_g2.zip && \
    curl -fsSL -O https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip && \
    curl -fsSL -O https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip && \
    unzip '*.zip' && rm -f *.zip
