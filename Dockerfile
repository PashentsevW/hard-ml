FROM --platform=amd64 ubuntu:18.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       git \
       curl \
       ca-certificates

RUN curl -L -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/conda \
    && rm -f ~/miniconda.sh 

RUN opt/conda/bin/conda create -n hard-ml-3 python=3.8.5

RUN git clone https://github.com/uber/causalml.git ~/causalml \
    && cd ~/causalml \
    && /opt/conda/envs/hard-ml-3/bin/pip install -r requirements.txt \
    && /opt/conda/envs/hard-ml-3/bin/python setup.py build_ext --inplace \
    && /opt/conda/envs/hard-ml-3/bin/python setup.py install \
    && rm -fR ~/causalml