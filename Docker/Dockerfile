# Stage 1: Build the environment
FROM python:3.12-slim as builder

# Copy requirements.txt from host to the builder stage
COPY requirements.txt /requirements.txt

# Install all dependencies from requirements.txt
RUN pip3 install --no-cache-dir -r /requirements.txt

RUN rm -rf /root/.cache/pip/*

# Stage 2: Create a minimal runtime image
FROM python:3.12-slim

# Copy the installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set the environment path
ENV PATH="/usr/local/bin:$PATH"

# Default command for the container
CMD ["/bin/bash"]