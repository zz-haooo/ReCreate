# DS-1000 Evaluation Environment
# Based on official environment.yml

FROM python:3.10-slim

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages (based on environment.yml)
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==1.5.3 \
    matplotlib==3.8.4 \
    seaborn==0.13.2 \
    scipy==1.12.0 \
    scikit-learn==1.4.0 \
    statsmodels==0.14.1 \
    xgboost==2.0.3 \
    gensim==4.3.2 \
    pyyaml \
    tqdm

# Install PyTorch CPU version (use extra-index-url for proper resolution)
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# Install TensorFlow CPU (aligned with environment.yml)
RUN pip install --no-cache-dir tensorflow-cpu==2.16.1

# Disable tensorflow logging
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV CUDA_VISIBLE_DEVICES=-1

# Set matplotlib to non-interactive backend
ENV MPLBACKEND=Agg

# Suppress warnings
ENV PYTHONWARNINGS=ignore

CMD ["/bin/bash"]
