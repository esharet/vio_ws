FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04


ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Check if "ubuntu" user exists, delete it if it does, then create the desired user
RUN if getent passwd ubuntu > /dev/null 2>&1; then \
        userdel -r ubuntu && \
        echo "Deleted existing ubuntu user"; \
    fi && \
    groupadd --gid $USER_GID $USERNAME && \
    useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo "Created new user $USERNAME"

# Add sudo support for the non-root user
RUN apt-get update && apt-get install -y sudo \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
  && chmod 0440 /etc/sudoers.d/$USERNAME \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3-pip \
    python3-numpy \
  && rm -rf /var/lib/apt/lists/*

# Install language
RUN apt-get update && apt-get install -y --no-install-recommends \
  locales \
  && locale-gen en_US.UTF-8 \
  && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
  && rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8

# Install timezone
RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime \
  && export DEBIAN_FRONTEND=noninteractive \
  && apt-get update \
  && apt-get install -y --no-install-recommends tzdata \
  && dpkg-reconfigure --frontend noninteractive tzdata \
  && rm -rf /var/lib/apt/lists/*

# Install common programs
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install ROS2
RUN sudo add-apt-repository universe \
  && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null \
  && apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ros-base \
    python3-argcomplete \
    ros-humble-rmw-cyclonedds-cpp \
    ros-dev-tools \
    python3-colcon-clean \
    python3-ament-package \
    ros-humble-ament-* \
  && rm -rf /var/lib/apt/lists/*

# cv_bridge dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libboost-all-dev \
    libboost-python-dev \
    python3-dev \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get -y install --no-install-recommends \
        libgstreamer1.0-0 \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-tools \
        python3-gi \
        gir1.2-gstreamer-1.0 \
        libgstreamer-plugins-base1.0-dev \
        gstreamer1.0-libav \
        gstreamer-1.0 \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    tmux \
    tmuxp \
    ffmpeg \
  && rm -rf /var/lib/apt/lists/*


# install vision
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-vision-msgs \
  && rm -rf /var/lib/apt/lists/*


# Install OpenCV
COPY opencv_debs/*.deb /tmp/

RUN ARCH=$(uname -m) && \
    apt update && \
    if [ "$ARCH" = "aarch64" ]; then \
        apt install -y /tmp/OpenCV-*-aarch64-*.deb; \
        apt install -y /tmp/ros-humble-cv-bridge_3.2.1-1jammy-cv410_arm64.deb; \
    elif [ "$ARCH" = "x86_64" ]; then \
        apt install -y /tmp/OpenCV-*-x86_64-*.deb; \
        apt install -y /tmp/ros-humble-cv-bridge_3.2.1-1jammy-cv410_amd64.deb; \
    else \
        echo "Unsupported architecture: $ARCH"; exit 1; \
    fi 

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-rqt \
    ros-humble-qt-gui \
    python3-matplotlib \
    ros-humble-diagnostic-updater \ 
    ros-humble-diagnostic-aggregator \
    ros-humble-diagnostic-msgs \
    ros-humble-diagnostic-common-diagnostics \
  && rm -rf /var/lib/apt/lists/*