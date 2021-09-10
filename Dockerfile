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

ENV PATH /opt/conda/bin:$PATH
RUN activate base \
    && conda install python=3.8.5 -y

RUN git clone https://github.com/uber/causalml.git ~/causalml \
    && cd ~/causalml \
    && pip install -r requirements.txt \
    && python setup.py build_ext --inplace \
    && python setup.py install \
    && rm -fR ~/causalml

RUN activate base \
    && pip install --user \
       dask \
       pyarrow \
       category_encoders

RUN mkdir -p ~/hard-ml/uplift \
    && git clone https://github.com/Antiguru11/hard-ml.git ~/hard-ml/uplift \
    && cd ~/hard-ml/uplift/final \ 
    && wget "https://downloader.disk.yandex.ru/zip/063ea176fbc2601963e2f0fca240de5c0625bfa53dfec45e90430f4068ed657f/613b7f0b/WUZOandzb1QrMnYrTVROTW5IakZxaU01N3R4T2QyU0xkbWhrVDV3NVlQOTBwS29ZeG9mczJaSlVhdXJrNERHZXEvSjZicG1SeU9Kb25UM1ZvWG5EYWc9PTo=?uid=0&filename=%D0%A4%D0%B8%D0%BD%D0%B0%D0%BB%D1%8C%D0%BD%D1%8B%D0%B9%20%D0%BF%D1%80%D0%BE%D0%B5%D0%BA%D1%82.zip&disposition=attachment&hash=YFNjwsoT%2B2v%2BMTNMnHjFqiM57txOd2SLdmhkT5w5YP90pKoYxofs2ZJUaurk4DGeq/J6bpmRyOJonT3VoXnDag%3D%3D%3A&limit=0&owner_uid=125175473&tknv=v2" -O data.zip \
    && unzip data.zip -d data \
    && mv 'data/Финальный проект'/* data \
    && rm -fR 'data/Финальный проект'