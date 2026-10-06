FROM python:3.14-slim

WORKDIR /work

RUN apt-get update \
    && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pandas \
    pillow \
    h5py \
    numpy