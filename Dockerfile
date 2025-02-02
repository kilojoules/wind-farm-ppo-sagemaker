# Base image with CUDA and Python
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenmpi-dev \
    python3.10 \
    python3-pip \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install "stable-baselines3[extra]" && \
    pip install py-wake && \
    pip install sagemaker-training

# Copy project files
COPY . /opt/program
WORKDIR /opt/program

# SageMaker entry point
ENV SAGEMAKER_PROGRAM train.py
