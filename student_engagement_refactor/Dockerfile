# Student Engagement Detection System - Docker Container

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt requirements_extended.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements_extended.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/labels \
    models/weights images results/evaluation results/visualizations

# Download YOLOv8 weights (optional - comment out if providing custom weights)
RUN wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt \
    -O models/weights/yolov8s.pt || echo "YOLO weights download failed - provide manually"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OPENCV_VIDEOIO_DEBUG=1

# Expose port for potential web interface (future)
EXPOSE 8000

# Default command - run main with webcam (override with custom command)
CMD ["python", "-m", "src.main", "--no-display"]

# Alternative commands:
# docker run <image> python -m src.main --source video.mp4 --no-display
# docker run <image> python scripts/batch_process.py /data/videos
# docker run <image> python scripts/evaluate_model.py --predictions /data/pred.csv --labels /data/labels.csv
