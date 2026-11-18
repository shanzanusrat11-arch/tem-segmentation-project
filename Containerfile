FROM python:3.11-slim
WORKDIR /work
RUN apt-get update && apt-get install -y --no-install-recommends make libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --timeout 180 --retries 10 --no-cache-dir \
    pandas pillow h5py numpy safetensors matplotlib scikit-image reportlab && \
    pip install --timeout 180 --retries 10 --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu
