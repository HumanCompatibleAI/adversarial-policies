# Based on OpenAI's mujoco-py Dockerfile

FROM nvidia/cuda:10.0-runtime-ubuntu18.04

RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections \\
    && apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    ffmpeg \
    software-properties-common \
    net-tools \
    parallel \
    rsync \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    ttf-mscorefonts-installer \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN DEBIAN_FRONTEND=noninteractive add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3.6-dev python3.6 python3-pip
RUN virtualenv --python=python3.6 env

RUN rm /usr/bin/python
RUN ln -s /env/bin/python3.6 /usr/bin/python
RUN ln -s /env/bin/pip3.6 /usr/bin/pip
RUN ln -s /env/bin/pytest /usr/bin/pytest

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

RUN DEBIAN_FRONTEND=noninteractive add-apt-repository --yes ppa:marmistrz/openmpi && apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libopenmpi3 libopenmpi-dev

ENV LANG C.UTF-8
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

RUN    mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco150.zip \
    && unzip mujoco150.zip -d /root/.mujoco \
    && rm mujoco150.zip \
    && wget https://www.roboti.us/download/mjpro131_linux.zip -O mujoco131.zip \
    && unzip mujoco131.zip -d /root/.mujoco \
    && rm mujoco131.zip

COPY vendor/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

WORKDIR /adversarial-policies
ARG MUJOCO_KEY
# Copy over just requirements.txt at first. That way, the Docker cache doesn't
# expire until we actually change the requirements.
COPY ./requirements-build.txt /adversarial-policies/
COPY ./requirements.txt /adversarial-policies/
COPY ./requirements-aprl.txt /adversarial-policies/
COPY ./requirements-modelfree.txt /adversarial-policies/
COPY ./ci/build_venv.sh /adversarial-policies/ci/build_venv.sh
RUN    curl -o /root/.mujoco/mjkey.txt ${MUJOCO_KEY} \
    && parallel ci/build_venv.sh {} ::: aprl modelfree \
    && rm /root/.mujoco/mjkey.txt  # remove activation key to avoid leaking it in image

# Delay copying (and installing) the code until the very end
COPY . /adversarial-policies
RUN parallel ". {}venv/bin/activate && pip install ." ::: aprl modelfree

# Default entrypoints
ENTRYPOINT ["/adversarial-policies/vendor/Xdummy-entrypoint"]
CMD ["ci/run_tests.sh"]

