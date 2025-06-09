FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

# Install Python, pip, git, and other dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev git \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ninja-build build-essential \
 && ln -s /usr/bin/python3 /usr/bin/python \
 && pip3 install --upgrade pip

WORKDIR /workspace

# Set CUDA environment variables
ENV CUDA_HOME="/usr/local/cuda"
ENV CUDA_ROOT="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install Detectron2 from GitHub
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Set working directory (this will be overridden when mounting volumes)
WORKDIR /workspace
