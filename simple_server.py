from flask import Flask, send_from_directory
import os

# Define the path to the frontend directory
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app/frontend')

# Initialize Flask app with static folder pointing to frontend directory
app = Flask(__name__, static_folder=FRONTEND_DIR)

# Route to serve the frontend
@app.route('/')
def index():
    """Serve the index.html file"""
    return send_from_directory(FRONTEND_DIR, 'index.html')

# Route to serve any static files from the frontend directory
@app.route('/<path:path>')
def serve_static(path):
    """Serve static files from the frontend directory"""
    if os.path.exists(os.path.join(FRONTEND_DIR, path)):
        return send_from_directory(FRONTEND_DIR, path)
    else:
        return index()

if __name__ == "__main__":
    print(f"Frontend directory: {FRONTEND_DIR}")
    print(f"Frontend directory exists: {os.path.exists(FRONTEND_DIR)}")
    print(f"Index.html exists: {os.path.exists(os.path.join(FRONTEND_DIR, 'index.html'))}")
    app.run(host='0.0.0.0', port=5000, debug=True)
