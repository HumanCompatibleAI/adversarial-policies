# Based on OpenAI's mujoco-py Dockerfile

ARG USE_MPI=True

FROM nvidia/cuda:10.0-runtime-ubuntu18.04 AS base
ARG USE_MPI
ARG DEBIAN_FRONTEND=noninteractive

RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections \
    && apt-get update -q \
    && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    net-tools \
    parallel \
    python3.7 \
    python3.7-dev \
    python3-pip \
    rsync \
    software-properties-common \
    unzip \
    vim \
    virtualenv \
    xpra \
    xserver-xorg-dev \
    ttf-mscorefonts-installer \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

ENV LANG C.UTF-8

RUN    mkdir -p /root/.mujoco \
    && curl -o mujoco200.zip https://www.roboti.us/download/mujoco200_linux.zip \
    && unzip mujoco200.zip -d /root/.mujoco \
    && mv /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200 \
    && rm mujoco200.zip \
    && curl -o mujoco131.zip https://www.roboti.us/download/mjpro131_linux.zip \
    && unzip mujoco131.zip -d /root/.mujoco \
    && rm mujoco131.zip

COPY vendor/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

RUN if [ $USE_MPI = "True" ]; then \
    add-apt-repository --yes ppa:marmistrz/openmpi \
    && apt-get update -q \
    && apt-get install -y libopenmpi3 libopenmpi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*; \
    fi

# Set the PATH to the venv before we create the venv, so it's visible in base.
# This is since we may create the venv outside of Docker, e.g. in CI
# or by binding it in for local development.
ENV PATH="/adversarial-policies/venv/bin:$PATH"
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:/root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}

FROM base as python-req
ARG USE_MPI

WORKDIR /adversarial-policies
# Copy over just requirements.txt at first. That way, the Docker cache doesn't
# expire until we actually change the requirements.
COPY ./requirements-build.txt /adversarial-policies/
COPY ./requirements.txt /adversarial-policies/
COPY ./requirements-dev.txt /adversarial-policies/
COPY ./ci/build_venv.sh /adversarial-policies/ci/build_venv.sh
# mjkey.txt needs to exist for build, but doesn't need to be a real key
RUN    touch /root/.mujoco/mjkey.txt && ci/build_venv.sh && rm -rf $HOME/.cache/pip

FROM python-req as full

# Delay copying (and installing) the code until the very end
COPY . /adversarial-policies
# Build a wheel then install to avoid copying whole directory (pip issue #2195)
RUN python3 setup.py sdist bdist_wheel
RUN pip install dist/aprl-*.whl

# Default entrypoints
ENTRYPOINT ["/adversarial-policies/vendor/Xdummy-entrypoint"]
CMD ["ci/run_tests.sh"]
