#!/usr/bin/env python3
# This script creates a simple Flask server to serve the frontend files

import os
from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS

# Define the path to the frontend directory
FRONTEND_DIR = '/app/app/frontend'

# Initialize Flask app with static folder pointing to frontend directory
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app)  # Enable CORS for all routes

# Route to serve the frontend
@app.route('/')
def index():
    """Serve the index.html file"""
    print(f"Serving index.html from {FRONTEND_DIR}")
    try:
        return send_from_directory(FRONTEND_DIR, 'index.html')
    except Exception as e:
        print(f"Error serving index.html: {str(e)}")
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
    print(f"Request for static file: {path}")
    full_path = os.path.join(FRONTEND_DIR, path)
    print(f"Full path: {full_path}, exists: {os.path.exists(full_path)}")
    
    if os.path.exists(full_path):
        print(f"Serving static file: {path}")
        return send_from_directory(FRONTEND_DIR, path)
    else:
        print(f"Static file not found: {path}, falling back to index.html")
        return index()

# Health check endpoint
@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    print(f"Frontend directory: {FRONTEND_DIR}")
    print(f"Frontend directory exists: {os.path.exists(FRONTEND_DIR)}")
    if os.path.exists(FRONTEND_DIR):
        print(f"Frontend directory contents: {os.listdir(FRONTEND_DIR)}")
        index_path = os.path.join(FRONTEND_DIR, 'index.html')
        print(f"index.html exists: {os.path.exists(index_path)}")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
