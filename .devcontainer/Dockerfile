# Start with Python 3.10 base image
FROM mcr.microsoft.com/devcontainers/python:1-3.10-bullseye

ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# The base image already includes a non-root user 'vscode'
USER vscode

# Set working directory
WORKDIR /workspaces/face-recognitions

# Create virtual environment (will be populated by setup.py)
RUN python -m venv .venv

# Update pip in the virtual environment
RUN .venv/bin/pip install --upgrade pip

# Add virtualenv activation to bash startup
RUN echo "source /workspaces/face-recognition/.venv/bin/activate" >> ~/.bashrc
