FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu20.04

ENV TZ=Europe/Minsk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential software-properties-common && \
    apt-get install --no-install-recommends -y python3.6 python3-pip python3-dev python3-setuptools python3-distutils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
    -f https://download.pytorch.org/whl/torch_stable.html

COPY ./requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
RUN rm requirements.txt

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user
