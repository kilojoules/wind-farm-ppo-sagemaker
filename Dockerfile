FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Ensure that HOME is set correctly
ENV HOME=/root
ENV PIXI_HOME=/root/.pixi

# Use bash for RUN commands 
SHELL ["/bin/bash", "-c"]

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libopenmpi-dev \
        python3.10 \
        python3-pip \
        python3.10-venv \
        curl \
        tar \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install pixi with verification
RUN curl -fsSL https://pixi.sh/install.sh -o install.sh && \
    chmod +x install.sh && \
    ./install.sh && \
    test -d /root/.pixi/bin || exit 1

# Update PATH to include pixi
ENV PATH="/root/.pixi/bin:${PATH}"

# Verify pixi installation
RUN pixi --version

# Create working directory
WORKDIR /opt/program

# Copy only necessary files first
COPY pyproject.toml ./
COPY src/ ./src/

# Install dependencies with pixi
RUN pixi self-update
RUN pixi install

# Copy entire project (including pyproject.toml)
COPY . /opt/program
WORKDIR /opt/program

# Now run Pixi to install everything from pyproject.toml
RUN pixi install  # Or `pixi lock && pixi sync --locked` if you want a reproducible lock

# Possibly do "pip install -e ." if your code has a setup.py or is declared as a PEP 621 project
# with a correct [build-system] in pyproject.toml

# Expose the training script to SageMaker
ENV SAGEMAKER_PROGRAM train.py

