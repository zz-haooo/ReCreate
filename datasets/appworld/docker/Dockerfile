# AppWorld Agent Environment
# Runs AppWorld Agent inside Docker (Unified mode)
#
# Consistent with other benchmarks (SWE, DA-Code, DS-1000, Math):
# - One container per task
# - Agent executes inside container
# - Evaluation completed inside container
#
# Build (from ReCreate root directory):
#   docker build -t appworld-agent:latest -f datasets/appworld/docker/Dockerfile datasets/appworld
#
# Note:
# - Uses Python 3.11 (AppWorld requirement)
# - Data mounted from host at runtime (saves space)
# - Host environment does NOT need AppWorld SDK installed

FROM python:3.11-slim-bookworm

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    pyyaml \
    requests \
    httpx \
    litellm \
    jinja2

# Copy and install AppWorld source (context is datasets/appworld)
# IMPORTANT: Run 'appworld install' on host BEFORE building this image
# to extract the encrypted app source code
COPY . /tmp/appworld
RUN cd /tmp/appworld && pip install --no-cache-dir . && rm -rf /tmp/appworld

# Create workspace directory structure
RUN mkdir -p /workspace/appworld_data \
    && mkdir -p /workspace/extras/agent_tools \
    && mkdir -p /workspace/extras/agent_memory \
    && mkdir -p /workspace/output

# Set environment variables
ENV APPWORLD_ROOT=/workspace/appworld_data
ENV PYTHONWARNINGS=ignore
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# AppWorld data mounted at runtime to /workspace/appworld_data
# Usage: docker run -v /host/appworld/data:/workspace/appworld_data/data ...

CMD ["/bin/bash"]
