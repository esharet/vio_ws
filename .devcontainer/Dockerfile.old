FROM humble:dev

ARG VERSION=0.0.1


RUN echo "image dev version: ${VERSION}">/etc/docker_image_dev_version

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libx11-xcb1 \
    libglu1-mesa \
    libxrender1 \
    libxi6 \
    libsm6 \
    libxext6 \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    libxcb-cursor0 \
  && rm -rf /var/lib/apt/lists/*

USER user

RUN pip install --no-cache-dir opencv-python

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-rqt \
        ros-humble-rqt-common-plugins \
        ros-humble-vision-msgs \
        ros-humble-rqt-image-overlay \
  && rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y --no-install-recommends \
    iproute2 \
  && rm -rf /var/lib/apt/lists/*