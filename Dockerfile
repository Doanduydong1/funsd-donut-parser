# Dockerfile

# Base: Python 3.8 slim
FROM python:3.8-slim-bullseye

# Env vars: Python & Pip
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=off
ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV DEBIAN_FRONTEND=noninteractive

# System libs (for OpenCV if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Workdir in image: /app
WORKDIR /app

# Install Python deps
COPY requirements.txt ./requirements.txt

# Install PyTorch (CPU)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other deps from requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY ./app ./app

# Create mount points (optional)
RUN mkdir -p /app/mounted_funsd_input_images
RUN mkdir -p /app/mounted_parsed_outputs

# Default command on container start
CMD ["python", "app/run_funsd_parser.py", \
     "--funsd_image_input_dir", "/app/mounted_funsd_input_images", \
     "--parsed_output_dir", "/app/mounted_parsed_outputs", \
     "--max_images_to_process", "3" \
    ]