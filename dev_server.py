import os
from flask import Flask, send_from_directory, request, Response
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Backend URL - this will be the Docker container when it's running
BACKEND_URL = "http://localhost:5000"

# Serve frontend files
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path == "" or path == "/":
        return send_from_directory('app/frontend', 'index.html')
    try:
        return send_from_directory('app/frontend', path)
    except:
        return send_from_directory('app/frontend', 'index.html')

# Proxy API requests to the backend
@app.route('/process-image', methods=['POST'])
def proxy_process_image():
    try:
        # Forward the request to the backend
        resp = requests.post(
            f"{BACKEND_URL}/process-image",
            json=request.json,
            stream=True
        )
        
        # Stream the response back to the client
        return Response(
            resp.iter_content(chunk_size=10*1024),
            content_type=resp.headers['Content-Type'],
            status=resp.status_code
        )
    except Exception as e:
        return {"error": str(e), "message": "Backend service unavailable"}, 503

@app.route('/health', methods=['GET'])
def proxy_health():
    try:
        resp = requests.get(f"{BACKEND_URL}/health")
        return resp.json(), resp.status_code
    except:
        return {"status": "error", "message": "Backend service unavailable"}, 503

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
