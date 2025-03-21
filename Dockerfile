# Use a Conda-based PyTorch image for better dependency resolution
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Set non-interactive frontend
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    tzdata \
    build-essential \
    curl \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Create a Conda environment
RUN conda create -n ai_env python=3.10 -y

# Set up shell to use the Conda environment
SHELL ["conda", "run", "-n", "ai_env", "/bin/bash", "-c"]

# Install PyTorch with conda (much better dependency resolution)
RUN conda install -y -c pytorch -c nvidia \
    pytorch=2.1.0 \
    torchvision \
    torchaudio \
    pytorch-cuda=12.1 && \
    conda clean -ya

# Install other dependencies with conda where possible
RUN conda install -y -c conda-forge \
    flask=2.3.2 \
    flask-cors=5.0.1 \
    numpy=1.24.3 \
    pillow=10.0.0 \
    requests=2.31.0 \
    pip && \
    conda clean -ya

# Install Hugging Face dependencies and Ultralytics inside Conda
RUN conda run -n ai_env pip install --no-cache-dir \
    "transformers>=4.33.0,<5.0.0" \
    "diffusers>=0.24.0,<0.25.0" \
    "accelerate>=0.23.0,<0.24.0" \
    "huggingface_hub>=0.17.3" \
    "tokenizers>=0.13.3" \
    "safetensors>=0.3.1" \
    "ultralytics"

# Install authentication dependencies
RUN conda run -n ai_env pip install --no-cache-dir \
    flask-sqlalchemy \
    flask-session \
    google-auth \
    google-auth-oauthlib \
    google-auth-httplib2

# Install gdown inside Conda
RUN conda run -n ai_env pip install --no-cache-dir gdown

# Create directories
RUN mkdir -p /app/models /app/output

# Clone required repositories
RUN git clone --depth 1 https://github.com/CASIA-IVA-Lab/FastSAM.git /app/FastSAM && \
    cd /app/FastSAM && \
    conda run -n ai_env pip install -e .

RUN git clone --depth 1 https://github.com/facebookresearch/segment-anything.git /app/segment-anything && \
    cd /app/segment-anything && \
    conda run -n ai_env pip install -e .

# Clone the YOLOv5 repository and install its requirements
RUN git clone https://github.com/ultralytics/yolov5 /app/yolov5 && \
    cd /app/yolov5 && \
    conda run -n ai_env pip install -r requirements.txt

# Download models
RUN gdown https://drive.google.com/uc?id=1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv -O /app/models/FastSAM-s.pt && \
    wget --timeout=30 --tries=3 https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n.pt -O /app/models/yolov5n.pt

# Download the Lustify model
# Note: Using a staged approach with curl to handle redirects and large file downloads better
RUN curl -L -o /app/models/lustifySDXLNSFW_v20-inpainting.safetensors \
    "https://civitai-delivery-worker-prod.5ac0637cfd0766c97916cefa3764fbdf.r2.cloudflarestorage.com/model/4138/lustify6MegafixFp16.0cgI.safetensors?X-Amz-Expires=86400&response-content-disposition=attachment%3B%20filename%3D%22lustifySDXLNSFW_oltONELASTTIME.safetensors%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=e01358d793ad6966166af8b3064953ad/20250320/us-east-1/s3/aws4_request&X-Amz-Date=20250320T100424Z&X-Amz-SignedHeaders=host&X-Amz-Signature=00e86dfddab1ea3f753c53212e1f7d9816bf08c34449161e803915a8d1a11db5" \
    || echo "Warning: Lustify model download failed. You may need to manually download it and place it in /app/models/"

# Create necessary directories with proper permissions
RUN mkdir -p /app/models /app/output && \
    chmod 777 /app/output

# Clone the GitHub repository for reproducibility
RUN apt-get update && apt-get install -y git && \
    git clone https://github.com/diegonunez77/nudify2.git /app/repo && \
    cp -r /app/repo/app /app/ && \
    rm -rf /app/repo/.git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy local files to override the ones from GitHub (for development)
COPY ./app /app/app/

# Make sure the frontend directory exists and has proper permissions
RUN ls -la /app/app && \
    ls -la /app/app/frontend || echo "Frontend directory not found" && \
    chmod -R 755 /app/app

# Set environment variables
ENV FLASK_APP=/app/app/backend/app.py
ENV FLASK_ENV=development
ENV FLASK_DEBUG=1

# Expose the Flask port
EXPOSE 5000

# Generate a lockfile for reproducibility
RUN conda env export > /app/environment.yml
RUN conda run -n ai_env pip freeze > /app/pip-requirements.txt

# Ensure Conda environment is activated properly before running Flask
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["conda run -n ai_env python /app/app/backend/app.py"]
