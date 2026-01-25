# Math Domain Evaluation Environment
# For mathematical reasoning tasks (MATH-500, AIME, AMC)
#
# Build:
#   cd datasets/math/docker && docker build -t math-env:latest .

FROM python:3.10-slim

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for math evaluation
RUN pip install --no-cache-dir \
    sympy \
    numpy \
    scipy \
    matplotlib \
    pyyaml

# Suppress warnings
ENV PYTHONWARNINGS=ignore
ENV PYTHONUNBUFFERED=1

CMD ["/bin/bash"]
