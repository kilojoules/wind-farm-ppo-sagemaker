FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libopenmpi-dev \
        python3.10 \
        python3-pip \
        python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Pixi or micromamba/conda (depending on how you want to handle .toml)
# For Pixi specifically:
RUN pip install pixi  # or `pipx install pixi` if you prefer

# Copy entire project (including pyproject.toml)
COPY . /opt/program
WORKDIR /opt/program

# Now run Pixi to install everything from pyproject.toml
RUN pixi install  # Or `pixi lock && pixi sync --locked` if you want a reproducible lock

# Possibly do "pip install -e ." if your code has a setup.py or is declared as a PEP 621 project
# with a correct [build-system] in pyproject.toml

# Expose the training script to SageMaker
ENV SAGEMAKER_PROGRAM train.py

