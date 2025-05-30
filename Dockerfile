# Dockerfile

# Base Image: Python 3.8 nhẹ
FROM python:3.8-slim-bullseye

# Biến môi trường cho Python & Pip
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=off
ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV DEBIAN_FRONTEND=noninteractive

# Cài thư viện hệ thống nếu cần cho OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Thư mục làm việc trong image
WORKDIR /app

# Cài Python dependencies
COPY requirements.txt ./requirements.txt

# Cài PyTorch (CPU) - Kiểm tra lệnh mới nhất trên trang PyTorch
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Cài các thư viện còn lại
RUN pip install --no-cache-dir -r requirements.txt

# Copy code app vào image
COPY ./app ./app

# Tạo thư mục để mount (tùy chọn)
RUN mkdir -p /app/mounted_funsd_input_images
RUN mkdir -p /app/mounted_parsed_outputs

# Lệnh chạy mặc định khi container khởi động
CMD ["python", "app/run_funsd_parser.py", \
     "--funsd_image_input_dir", "/app/mounted_funsd_input_images", \
     "--parsed_output_dir", "/app/mounted_parsed_outputs", \
     "--max_images_to_process", "3" \
    ]