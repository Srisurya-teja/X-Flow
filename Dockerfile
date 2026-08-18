FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.0"


RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*


RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1


RUN python -m pip install --upgrade pip setuptools wheel


RUN pip install \
    torch==2.3.1 \
    torchvision==0.18.1


RUN pip install mxnet==1.9.1

# ---------------------------------------------------------
# Core Python dependencies
# ---------------------------------------------------------

RUN pip install \
    numpy==1.23.5 \
    opencv-python-headless \
    Pillow \
    easydict \
    pyyaml \
    tensorboardX \
    scikit-learn \
    scipy \
    tqdm

# ---------------------------------------------------------
# Azure File Share upload helpers (only used when USE_AZURE=True).
# Adjust to match your /home/shared_scripts/azure_test requirements.
# ---------------------------------------------------------

RUN pip install azure-storage-file-share azure-identity requests

# ---------------------------------------------------------
# Project directory
# ---------------------------------------------------------

WORKDIR /workspace/FROM

# ---------------------------------------------------------
# Copy project into container
# ---------------------------------------------------------

COPY . /workspace/FROM/

# ---------------------------------------------------------
# Create project directories
# ---------------------------------------------------------

RUN mkdir -p \
    /workspace/FROM/output \
    /workspace/FROM/pretrained \
    /workspace/FROM/temp

# ---------------------------------------------------------
# Default command
# ---------------------------------------------------------

CMD ["/bin/bash"]