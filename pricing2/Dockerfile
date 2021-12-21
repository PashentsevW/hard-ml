FROM --platform=amd64 continuumio/miniconda3:latest

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl wget unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --user -r requirements.txt 