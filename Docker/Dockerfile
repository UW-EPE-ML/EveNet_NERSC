# Stage 1: Build environment with CUDA + Python 3.12
FROM docker.io/nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

# Install dependencies for building Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libssl-dev \
    zlib1g-dev \
    libsqlite3-dev \
    libffi-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Download and install Python 3.12.3
RUN curl -O https://www.python.org/ftp/python/3.12.3/Python-3.12.3.tgz && \
    tar -xzf Python-3.12.3.tgz && \
    cd Python-3.12.3 && \
    ./configure --enable-optimizations --prefix=/usr/local && \
    make -j$(nproc) && make install && \
    cd .. && rm -rf Python-3.12.3*

# Symlink python and pip
RUN ln -sf /usr/local/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/local/bin/pip3.12 /usr/bin/pip

# Set CUDA paths
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Copy your requirements
COPY requirements.txt /requirements.txt

# Install Python packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /requirements.txt

# Build torch-linear-assignment with CUDA
#ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
RUN git clone https://github.com/AvencastF/torch-linear-assignment.git /workspace/torch-linear-assignment && \
    cd /workspace/torch-linear-assignment && \
    pip install --no-cache-dir --no-build-isolation .

# Optional: clean pip cache
RUN rm -rf /root/.cache/pip/*

# Stage 2: Minimal runtime image
FROM docker.io/nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

# Copy Python from builder stage
COPY --from=builder /usr/local /usr/local
COPY --from=builder /usr/bin/python /usr/bin/python
COPY --from=builder /usr/bin/pip /usr/bin/pip

# CUDA env vars
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Default shell
CMD ["/bin/bash"]