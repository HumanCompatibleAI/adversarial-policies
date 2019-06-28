# Based on OpenAI's mujoco-py Dockerfile

FROM nvidia/cuda:10.0-runtime-ubuntu18.04
ARG DEBIAN_FRONTEND=noninteractive

RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections \
    && apt-get update -q \
    && apt-get install -y \
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

RUN add-apt-repository --yes ppa:deadsnakes/ppa \
    && apt-get update -q \
    && apt-get install -y python3.7-dev python3.7 python3-pip

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

RUN add-apt-repository --yes ppa:marmistrz/openmpi \
    && apt-get update -q \
    && apt-get install -y libopenmpi3 libopenmpi-dev

ENV LANG C.UTF-8
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

RUN    mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mjpro200_linux.zip -O mujoco200.zip \
    && unzip mujoco200.zip -d /root/.mujoco \
    && mv /root/.mujoco/mjpro200_linux /root/.mujoco/mujoco200 \
    && rm mujoco200.zip \
    && wget https://www.roboti.us/download/mjpro131_linux.zip -O mujoco131.zip \
    && unzip mujoco131.zip -d /root/.mujoco \
    && rm mujoco131.zip

COPY vendor/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

WORKDIR /adversarial-policies
# Copy over just requirements.txt at first. That way, the Docker cache doesn't
# expire until we actually change the requirements.
COPY ./requirements-build.txt /adversarial-policies/
COPY ./requirements.txt /adversarial-policies/
COPY ./requirements-aprl.txt /adversarial-policies/
COPY ./requirements-modelfree.txt /adversarial-policies/
COPY ./ci/build_venv.sh /adversarial-policies/ci/build_venv.sh
# mjkey.txt needs to exist for build, but doesn't need to be a real key
RUN    touch /root/.mujoco/mjkey.txt \
    && parallel ci/build_venv.sh {} ::: aprl modelfree

# Delay copying (and installing) the code until the very end
COPY . /adversarial-policies
# Build a wheel then install to avoid copying whole directory (pip issue #2195)
RUN python3 setup.py sdist bdist_wheel
RUN parallel ". {}venv/bin/activate && \
              pip install dist/aprl-*.whl" ::: aprl modelfree

# Default entrypoints
ENTRYPOINT ["/adversarial-policies/vendor/Xdummy-entrypoint"]
CMD ["ci/run_tests.sh"]

