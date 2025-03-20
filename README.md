# Nudify2 - AI Image Transformation

Nudify2 is an AI-powered image transformation tool that uses advanced computer vision and image generation models to transform images according to user-specified prompts.

## Features

- **Image Transformation**: Transform images using AI models
- **Multiple Transformation Types**: Support for various transformation prompts (bikini, swimsuit, lingerie, etc.)
- **User-Friendly Interface**: Modern, responsive web interface
- **Image History**: Keep track of your transformation history
- **Download Results**: Save transformed images to your device

## Technology Stack

- **Backend**: Python, Flask, PyTorch
- **AI Models**: 
  - FastSAM for image segmentation
  - YOLO for object detection
  - Stable Diffusion for image inpainting
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Containerization**: Docker

## Getting Started

### Prerequisites

- Docker and Docker Compose
- CUDA-compatible GPU (recommended for faster processing)
- Internet connection for downloading models during build

### Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/nudify2.git
   cd nudify2
   ```

2. Start the application using Docker Compose:
   ```bash
   docker-compose up --build
   ```
   This will automatically download all required models, including:
   - FastSAM-s.pt (for image segmentation)
   - yolov5n.pt (for object detection)
   - lustifySDXLNSFW_v20-inpainting.safetensors (for image inpainting)

   > **Note**: If the Lustify model download fails during the build process (due to expired URL or other issues), you'll need to manually download it and place it in the `/app/models/` directory with the filename `lustifySDXLNSFW_v20-inpainting.safetensors`.

3. Access the web interface:
   Open your browser and navigate to `http://localhost:5000`

## Usage

1. Enter an image URL or upload an image (URL method currently supported)
2. Select or enter a transformation prompt
3. Click "Transform Image"
4. View and download the result

## API Endpoints

- `GET /health`: Health check endpoint
- `POST /process-image`: Process an image with a prompt
  - Request body: `{ "image_url": "URL_TO_IMAGE", "prompt": "TRANSFORMATION_PROMPT" }`
  - Returns: The transformed image

## Development

### Project Structure

```
nudify2/
├── app/
│   ├── backend/
│   │   ├── app.py           # Main application code
│   │   └── depend-resolver.py # Dependency resolver
│   └── frontend/
│       └── index.html       # Frontend interface
├── models/                  # Pre-trained models
├── GroundingDINO/           # GroundingDINO repository
├── docker-compose.yml       # Docker Compose configuration
├── Dockerfile               # Docker configuration
├── setup_directories.sh     # Host setup script for remote deployment
├── requirements.txt         # Python dependencies
└── pyproject.toml           # Poetry configuration
```

## Remote Deployment

To deploy Nudify2 on a remote machine, follow these steps:

1. Clone the repository on the remote machine:
   ```bash
   git clone https://github.com/yourusername/nudify2.git
   cd nudify2
   ```

2. Run the setup script to prepare the host environment:
   ```bash
   chmod +x setup_directories.sh
   ./setup_directories.sh
   ```
   This script will:
   - Create necessary directories for Docker volumes
   - Set appropriate permissions
   - Verify Docker and Docker Compose are installed

3. Build and start the application:
   ```bash
   docker-compose up --build
   ```

4. Access the application at `http://remote-machine-ip:5000`

### Volume Configuration

The Docker Compose configuration uses two volume mounts for production:

- `./models:/app/models`: Persists AI models between container rebuilds
- `./app/output:/app/output`: Stores processed images

This optimized volume setup ensures:
- Models are downloaded only once and reused across container rebuilds
- Output images are saved persistently on the host machine
- The application code is contained within the Docker image for better reproducibility

### Adding New Features

To add new transformation types:
1. Update the frontend preset prompts in `app/frontend/index.html`
2. Ensure the backend models can handle the new prompt types

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- FastSAM for image segmentation
- YOLO for object detection
- Stable Diffusion for image generation
