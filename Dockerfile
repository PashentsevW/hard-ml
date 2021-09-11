FROM --platform=amd64 ubuntu:18.04

LABEL maintainer="antiguru110894@gmail.com" version="0.0.0"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       git \
       curl \
       wget \
       unzip \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -L -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/conda \
    && rm -f ~/miniconda.sh 

RUN mkdir -p ~/hard-ml/uplift \
    && git clone https://github.com/Antiguru11/hard-ml.git ~/hard-ml/uplift

ENV PATH /opt/conda/bin:$PATH
RUN activate base \
    && conda install python=3.8.5 -y \
    && pip install --user \
       causalml \
       dask \
       pyarrow \
       category_encoders