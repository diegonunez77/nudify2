#!/bin/bash

# Setup script for Nudify2 remote deployment
# Creates necessary directories for Docker volumes and sets permissions

echo "=== Nudify2 Remote Deployment Setup ==="

# Create models directory for persisting models
mkdir -p ./models
echo "✓ Created models directory for persisting AI models"

# Create output directory for saving processed images
mkdir -p ./app/output
echo "✓ Created output directory for processed images"

# Set proper permissions for the output directory
# This ensures the container can write to this directory
chmod 777 ./app/output
echo "✓ Set write permissions for output directory"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "⚠️ Docker is not installed. Please install Docker before continuing."
    echo "   Visit https://docs.docker.com/engine/install/ for installation instructions."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "⚠️ Docker Compose is not installed. Please install Docker Compose before continuing."
    echo "   Visit https://docs.docker.com/compose/install/ for installation instructions."
    exit 1
fi

echo "\n✅ Setup complete! You can now run the application with:"
echo "   docker-compose up --build"
echo "\n📋 After the build completes, access the application at:"
echo "   http://localhost:5000"

echo "\n💡 Note: The first build may take some time as it downloads all required models and dependencies."
