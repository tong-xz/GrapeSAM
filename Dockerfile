# Dockerfile for training and inference

# Use NVIDIA's CUDA base image with your CUDA version
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Set environment variables for non-interactive installations
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PATH=/opt/conda/bin:$PATH

# Install Miniconda and other dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    bzip2 \
    libx11-6 \
    curl \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
 

RUN curl -o /miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh && \
    /opt/conda/bin/conda clean -a

# Set Conda environment as default
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

# Set working directory
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt .

# Create a conda environment and install dependencies from requirements.txt
RUN conda create -n grapesam python=3.8 && \
    conda activate grapesam && \
    pip install -r requirements.txt

# Copy your application code into the container
COPY . /app

# Set the conda environment to be used by default
SHELL ["conda", "run", "-n", "grapesam", "/bin/bash", "-c"]

# Make RUN commands use the new environment by default
ENV PATH /opt/conda/envs/grapesam/bin:$PATH