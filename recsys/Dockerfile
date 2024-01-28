FROM ubuntu:22.04

ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=1001

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN sudo apt-get update \
    && sudo apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git

RUN sudo rm -rf /var/cache/apt/archives /var/lib/apt/lists/*
