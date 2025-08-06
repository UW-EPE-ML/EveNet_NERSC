# Stage 1: Build environment with CUDA + Python 3.12
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

# Install Python 3.12 and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    g++ \
    git \
    ninja-build \
    curl \
    ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa && apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip and install wheel
RUN pip install --upgrade pip wheel setuptools

# Set CUDA paths
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Copy your requirements
COPY requirements.txt /requirements.txt

# Install Python packages
RUN pip install --no-cache-dir -r /requirements.txt

# Build torch-linear-assignment with CUDA
ENV TLA_BUILD_CUDA=1
RUN pip install --no-cache-dir torch_linear_assignment==0.0.4

# Optional: clean cache
RUN rm -rf /root/.cache/pip/*

# Stage 2: Minimal runtime image
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

# Install Python 3.12 runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa && apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Copy installed Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# CUDA runtime env vars
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Default shell
CMD ["/bin/bash"]