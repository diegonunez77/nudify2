#!/usr/bin/env python3
# This script creates a simple Flask server to serve the frontend files with enhanced logging

import os
import logging
from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define the path to the frontend directory
FRONTEND_DIR = '/app/app/frontend'

# Initialize Flask app with static folder pointing to frontend directory
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app)  # Enable CORS for all routes

# Log all requests
@app.before_request
def log_request_info():
    logger.debug('Request Headers: %s', request.headers)
    logger.debug('Request Body: %s', request.get_data())

# Route to serve the frontend
@app.route('/')
def index():
    """Serve the index.html file"""
    logger.info(f"Serving index.html from {FRONTEND_DIR}")
    try:
        return send_from_directory(FRONTEND_DIR, 'index.html')
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}")
        return jsonify({
            "error": "Failed to serve index.html",
            "message": str(e),
            "frontend_dir": FRONTEND_DIR,
            "frontend_dir_exists": os.path.exists(FRONTEND_DIR),
            "index_exists": os.path.exists(os.path.join(FRONTEND_DIR, 'index.html')) if os.path.exists(FRONTEND_DIR) else False
        }), 500

# Route to serve any static files from the frontend directory
@app.route('/<path:path>')
def serve_static(path):
    """Serve static files from the frontend directory"""
    logger.info(f"Request for static file: {path}")
    full_path = os.path.join(FRONTEND_DIR, path)
    logger.info(f"Full path: {full_path}, exists: {os.path.exists(full_path)}")
    
    if os.path.exists(full_path):
        logger.info(f"Serving static file: {path}")
        return send_from_directory(FRONTEND_DIR, path)
    else:
        logger.warning(f"Static file not found: {path}, falling back to index.html")
        return index()

# Health check endpoint
@app.route('/health')
def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    return jsonify({"status": "ok"})

# Mock API endpoint for testing
@app.route('/api/process_image', methods=['POST'])
def mock_process_image():
    """Mock endpoint for processing images"""
    logger.info("Mock process_image endpoint called")
    logger.info(f"Request data: {request.get_data()}")
    return jsonify({
        "status": "success",
        "message": "This is a mock response from the server. The actual image processing is not implemented in this simple server."
    })

if __name__ == '__main__':
    logger.info(f"Frontend directory: {FRONTEND_DIR}")
    logger.info(f"Frontend directory exists: {os.path.exists(FRONTEND_DIR)}")
    if os.path.exists(FRONTEND_DIR):
        logger.info(f"Frontend directory contents: {os.listdir(FRONTEND_DIR)}")
        index_path = os.path.join(FRONTEND_DIR, 'index.html')
        logger.info(f"index.html exists: {os.path.exists(index_path)}")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
