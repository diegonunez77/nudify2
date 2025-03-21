import torch
import numpy as np
import os
from fastsam import FastSAM, FastSAMPrompt
from diffusers import StableDiffusionInpaintPipeline
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check CUDA availability and set device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {DEVICE}")

if DEVICE == 'cuda':
    # Clear CUDA cache to prevent memory issues
    torch.cuda.empty_cache()

# Function to load image from URL
def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

# Define models directory
MODELS_DIR = '/app/models'

# Load YOLO model from the correct path
model = YOLO(os.path.join(MODELS_DIR, "yolov5n.pt"))

# Load FastSAM model from the correct path
fastsam_model = FastSAM(os.path.join(MODELS_DIR, "FastSAM-s.pt"))

# Load the Stable Diffusion Inpainting model
# Note: You'll need to ensure this file is available in your container
sd_model = StableDiffusionInpaintPipeline.from_single_file(
    os.path.join(MODELS_DIR, "lustifySDXLNSFW_v20-inpainting.safetensors"),
    torch_dtype=torch.float16 if DEVICE == 'cuda' else torch.float32
).to(DEVICE)

# Object Detection with YOLO
def detect_objects(image, text_prompt):
    results = model(image)  # Perform inference on the image
    boxes = []
    
    # Process results in the correct format for YOLO v5/v8
    for result in results:
        for box in result.boxes:
            if box.conf > 0.35:  # Set the confidence threshold
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Convert to list
                boxes.append([x1, y1, x2, y2])
    
    return boxes

# Segment the image using FastSAM
def segment_image(image, bbox):
    # Convert PIL image to numpy array if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
        
    everything_results = fastsam_model(image_np, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
    prompt_process = FastSAMPrompt(image_np, everything_results, device=DEVICE)
    
    # Use the bounding box prompt for segmentation
    # Ensure bbox is in the correct format [x1, y1, x2, y2]
    ann = prompt_process.box_prompt(bbox=bbox)
    
    return ann

# Inpaint the image using Stable Diffusion
def inpaint_image(image, mask, prompt):
    # Ensure mask is in the correct format (PIL Image with values 0 or 255)
    if not isinstance(mask, Image.Image):
        # Convert numpy array to PIL Image
        mask_array = mask.astype(np.uint8) * 255
        mask = Image.fromarray(mask_array)
        
    # Ensure image is a PIL Image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
        
    # Resize mask to match image dimensions if needed
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.NEAREST)
    
    inpaint_result = sd_model(
        prompt=prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=50,
        guidance_scale=7.5
    )
    
    return inpaint_result.images[0]

# Main function to run the pipeline
def main(image_url, prompt, output_path):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load the image
    image = load_image_from_url(image_url)
    
    # Detect objects
    boxes = detect_objects(image, prompt)
    
    if not boxes:
        logger.warning("No object detected.")
        return
    
    # Take the first detected bounding box
    bbox = boxes[0]
    
    # Segment the object using FastSAM
    annotations = segment_image(image, bbox)
    
    # The FastSAM annotations should contain the mask you need
    # Get the first annotation if multiple are returned
    if isinstance(annotations, list) and len(annotations) > 0:
        mask = annotations[0]
    else:
        mask = annotations
    
    # Inpaint the image with Stable Diffusion
    inpainted_image = inpaint_image(image, mask, prompt)
    
    # Save the inpainted image to the specified output path
    inpainted_image.save(output_path)
    logger.info(f"Inpainted image saved to: {output_path}")

# Create a Flask API
from flask import Flask, request, jsonify, send_file, abort, make_response, send_from_directory
from flask_cors import CORS
import logging
import os
import uuid
import time
from werkzeug.utils import secure_filename
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the path to the frontend directory
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend')

# Log frontend directory path for debugging
logger.info(f"Frontend directory path: {FRONTEND_DIR}")
logger.info(f"Frontend directory exists: {os.path.exists(FRONTEND_DIR)}")
if os.path.exists(FRONTEND_DIR):
    logger.info(f"Frontend directory contents: {os.listdir(FRONTEND_DIR)}")
    index_path = os.path.join(FRONTEND_DIR, 'index.html')
    logger.info(f"index.html exists: {os.path.exists(index_path)}")

# Initialize Flask app with static folder pointing to frontend directory
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app)  # Enable CORS for all routes

# Configure base output directory
BASE_OUTPUT_DIR = '/app/output'

# Configure upload folder
UPLOAD_FOLDER = os.path.join(BASE_OUTPUT_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure output folder
OUTPUT_FOLDER = os.path.join(BASE_OUTPUT_DIR, 'results')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Service is running"})

@app.route('/process-image', methods=['POST'])
def process_image():
    """Process an image using the AI pipeline"""
    logger.info("Received image processing request")
    start_time = time.time()
    
    try:
        data = request.json
        if not data:
            logger.error("No JSON data received")
            return jsonify({"error": "No data provided"}), 400
        
        # Extract parameters from the request
        image_url = data.get('image_url')
        prompt = data.get('prompt')
        
        # Validate inputs
        if not image_url:
            logger.error("Missing image_url parameter")
            return jsonify({"error": "Missing image_url parameter"}), 400
        
        if not prompt:
            logger.error("Missing prompt parameter")
            return jsonify({"error": "Missing prompt parameter"}), 400
        
        # Generate a unique filename based on timestamp and UUID
        parsed_url = urlparse(image_url)
        original_filename = os.path.basename(parsed_url.path) or 'image'
        filename_base = secure_filename(original_filename.split('.')[0])
        unique_id = str(uuid.uuid4())[:8]
        output_filename = f"{filename_base}_{unique_id}.jpg"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        logger.info(f"Processing image from {image_url} with prompt: {prompt}")
        logger.info(f"Output will be saved to: {output_path}")
        
        # Process the image
        main(image_url, prompt, output_path)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Image successfully processed in {processing_time:.2f} seconds")
        
        # Return the processed image
        return send_file(output_path, mimetype='image/jpeg')
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return jsonify({
            "error": str(e),
            "message": "Failed to process the image. Please try again with a different image or prompt."
        }), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Process an uploaded image file"""
    logger.info("Received file upload request")
    start_time = time.time()
    
    try:
        # Check if file part is in the request
        if 'file' not in request.files:
            logger.error("No file part in the request")
            return jsonify({"error": "No file part in the request"}), 400
            
        file = request.files['file']
        prompt = request.form.get('prompt')
        
        # Check if file is selected
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({"error": "No file selected"}), 400
            
        # Check if prompt is provided
        if not prompt:
            logger.error("No prompt provided")
            return jsonify({"error": "No prompt provided"}), 400
            
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'webp', 'gif'}
        if not file.filename.lower().endswith(tuple('.' + ext for ext in allowed_extensions)):
            logger.error(f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}")
            return jsonify({"error": f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"}), 400
            
        # Create a secure filename and save the uploaded file
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        upload_filename = f"{timestamp}_{unique_id}_{filename}"
        upload_path = os.path.join(UPLOAD_FOLDER, upload_filename)
        file.save(upload_path)
        
        logger.info(f"File saved to {upload_path}")
        
        # Generate output path
        output_filename = f"result_{timestamp}_{unique_id}.jpg"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Process the image
        try:
            # Load the image
            image = Image.open(upload_path)
            
            # Detect objects
            boxes = detect_objects(image, prompt)
            
            if not boxes:
                logger.warning("No object detected in the uploaded image.")
                return jsonify({"error": "No suitable object detected in the image."}), 400
            
            # Take the first detected bounding box
            bbox = boxes[0]
            
            # Segment the object using FastSAM
            annotations = segment_image(image, bbox)
            
            # Get the mask from annotations
            if isinstance(annotations, list) and len(annotations) > 0:
                mask = annotations[0]
            else:
                mask = annotations
            
            # Inpaint the image with Stable Diffusion
            inpainted_image = inpaint_image(image, mask, prompt)
            
            # Save the inpainted image
            inpainted_image.save(output_path)
            logger.info(f"Processed image saved to: {output_path}")
            
            # Return the processed image
            processing_time = time.time() - start_time
            logger.info(f"Image processing completed in {processing_time:.2f} seconds")
            
            return send_file(output_path, mimetype='image/jpeg')
            
        except Exception as e:
            logger.error(f"Error processing uploaded image: {str(e)}")
            return jsonify({
                "error": "Failed to process the uploaded image",
                "message": str(e)
            }), 500
            
    except Exception as e:
        logger.error(f"Error handling file upload: {str(e)}")
        return jsonify({
            "error": "Failed to handle file upload",
            "message": str(e)
        }), 500

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

# Example usage for direct Python execution
if __name__ == "__main__":
    # For development, enable debug mode
    app.run(host='0.0.0.0', port=5000, debug=True)