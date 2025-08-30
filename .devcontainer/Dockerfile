FROM mcr.microsoft.com/devcontainers/python:1-3.10-bullseye

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# The base image already includes a non-root user 'vscode'
ARG USERNAME=vscode
ARG WORKSPACE_DIR=/workspace

# Create workspace directory and set permissions if it doesn't exist
RUN if [ ! -d "${WORKSPACE_DIR}" ]; then \
        mkdir -p ${WORKSPACE_DIR}; \
        chown ${USERNAME}:${USERNAME} ${WORKSPACE_DIR}; \
    fi

# Switch to non-root user
USER ${USERNAME}

# Set working directory
WORKDIR /workspace

# Create virtual environment (will be populated by setup.py)
RUN python -m venv .venv

# Add virtual environment to PATH
ENV PATH="/workspace/.venv/bin:$PATH"

# Update pip in the virtual environment
RUN .venv/bin/pip install --upgrade pip

# Note: Requirements file will be available in the mounted workspace
# Install packages from requirements file during container creation
ARG REQUIREMENTS_FILE="requirements.txt"
RUN if [ -f "${REQUIREMENTS_FILE}" ]; then \
        .venv/bin/pip install --no-cache-dir -r ${REQUIREMENTS_FILE}; \
    fi
