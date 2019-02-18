# Based on OpenAI's mujoco-py Dockerfile

FROM nvidia/cuda@sha256:4df157f2afde1cb6077a191104ab134ed4b2fd62927f27b69d788e8e79a45fa1

RUN    apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libopenmpi-dev \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    ffmpeg \
    software-properties-common \
    net-tools \
    parallel \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN DEBIAN_FRONTEND=noninteractive add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes python3.6-dev python3.6 python3-pip
RUN virtualenv --python=python3.6 env

RUN rm /usr/bin/python
RUN ln -s /env/bin/python3.6 /usr/bin/python
RUN ln -s /env/bin/pip3.6 /usr/bin/pip
RUN ln -s /env/bin/pytest /usr/bin/pytest

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

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

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
COPY ./vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /adversarial_policies
ARG MUJOCO_KEY
# Copy over just requirements.txt at first. That way, the Docker cache doesn't
# expire until we actually change the requirements.
COPY ./requirements-build.txt /adversarial_policies/
COPY ./requirements.txt /adversarial_policies/
COPY ./requirements-aprl.txt /adversarial_policies/
COPY ./requirements-modelfree.txt /adversarial_policies/
COPY ./ci/build_venv.sh /adversarial_policies/ci/build_venv.sh
RUN    curl -o /root/.mujoco/mjkey.txt ${MUJOCO_KEY} \
    && parallel ci/build_venv.sh {} ::: aprl modelfree \
    && rm /root/.mujoco/mjkey.txt  # remove activation key to avoid leaking it in image

# Delay moving in the entire code until the very end.
ENTRYPOINT ["/adversarial_policies/vendor/Xdummy-entrypoint"]
CMD ["ci/run_tests.sh"]
COPY . /adversarial_policies
