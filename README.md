# Nudify2 - AI Image Transformation

Nudify2 is an AI-powered image transformation tool that uses advanced computer vision and image generation models to transform images according to user-specified prompts.

## Features

- **Image Transformation**: Transform images using AI models
- **Multiple Transformation Types**: Support for various transformation prompts (bikini, swimsuit, lingerie, etc.)
- **User-Friendly Interface**: Modern, responsive web interface
- **Image History**: Keep track of your transformation history
- **Download Results**: Save transformed images to your device
- **Google Sign-In**: Secure authentication using Google OAuth
- **Credits System**: Track and manage user credits for image transformations

## Technology Stack

- **Backend**: Python, Flask, PyTorch, SQLAlchemy
- **AI Models**: 
  - FastSAM for image segmentation
  - YOLO for object detection
  - Stable Diffusion for image inpainting
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Authentication**: Google OAuth 2.0
- **Database**: SQLite (can be configured for other databases)
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

3. Configure Google OAuth:
   - Go to the [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one
   - Navigate to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth client ID"
   - Set up the OAuth consent screen if prompted
   - For Application type, select "Web application"
   - Add authorized JavaScript origins: `http://localhost:5000` (and your production domain if applicable)
   - Add authorized redirect URIs: `http://localhost:5000/auth/google/callback` (and your production domain equivalent)
   - Copy the Client ID and update the `GOOGLE_CLIENT_ID` in your docker-compose.yml file

4. Update environment variables in docker-compose.yml:
   ```yaml
   environment:
     - SECRET_KEY=your_secure_random_string  # Change this to a secure random string
     - GOOGLE_CLIENT_ID=your_google_client_id  # From Google Cloud Console
   ```

5. Access the web interface:
   Open your browser and navigate to `http://localhost:5000`

## User Authentication and Credits

Nudify2 implements a credit-based system for image transformations:

- Users must sign in with their Google account to use the application
- Each new user receives 10 free credits upon registration
- Each image transformation costs 1 credit
- The application prevents transformations if the user has insufficient credits

This system is designed to be scalable for future payment integration, allowing users to purchase additional credits.

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
├── output/                  # Output directory for processed images
├── docker-compose.yml       # Docker Compose configuration
├── Dockerfile               # Docker configuration (clones repo automatically)
├── setup_directories.sh     # Host setup script for remote deployment
├── requirements.txt         # Python dependencies
└── pyproject.toml           # Poetry configuration
```

In this Git-based approach, the Dockerfile automatically clones the repository during container build, ensuring all code is consistent with the GitHub version.

## Remote Deployment

To deploy Nudify2 on a remote machine, follow these steps:

1. Download the docker-compose.yml file:
   ```bash
   curl -O https://raw.githubusercontent.com/diegonunez77/nudify2/main/docker-compose.yml
   ```

2. Build and start the application:
   ```bash
   docker-compose up --build
   ```

3. Access the application at `http://remote-machine-ip:5000`

> **Note**: Docker will automatically create the output directory. Make sure Docker and Docker Compose are installed on your system.

### Git-Based Reproducibility

This approach uses Git for version management and reproducibility. The Dockerfile will automatically:
- Clone the latest version of the repository from GitHub
- Set up all required dependencies
- Download necessary AI models

### Volume Configuration

The Docker Compose configuration uses one volume mount for production:

- `./output:/app/output`: Stores processed images

All code and models are managed through Git, ensuring consistent deployment across environments. This setup ensures:
- The application code is always up-to-date with the latest GitHub version
- Models are downloaded during container build for better reproducibility
- Output images are saved persistently on the host machine

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
