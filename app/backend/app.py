import torch
import numpy as np
import os
import requests
from PIL import Image
from io import BytesIO
import secrets
from datetime import datetime, timedelta

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check if we should skip model loading (for testing)
SKIP_MODEL_LOADING = os.environ.get('SKIP_MODEL_LOADING', 'false').lower() == 'true'
logger.info(f"Skip model loading: {SKIP_MODEL_LOADING}")

# Check CUDA availability and set device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {DEVICE}")

if DEVICE == 'cuda' and not SKIP_MODEL_LOADING:
    # Clear CUDA cache to prevent memory issues
    torch.cuda.empty_cache()

# Function to load image from URL
def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

# Define models directory
MODELS_DIR = '/app/models'

# Initialize model variables
model = None
fastsam_model = None
sd_model = None

# Only load models if not in testing mode
if not SKIP_MODEL_LOADING:
    try:
        # Import model-specific libraries only when needed
        from fastsam import FastSAM, FastSAMPrompt
        from diffusers import StableDiffusionInpaintPipeline
        from ultralytics import YOLO
        
        # Load YOLO model from the correct path
        model = YOLO(os.path.join(MODELS_DIR, "yolov5n.pt"))
        
        # Load FastSAM model from the correct path
        fastsam_model = FastSAM(os.path.join(MODELS_DIR, "FastSAM-s.pt"))
        
        # Load the Stable Diffusion Inpainting model
        sd_model = StableDiffusionInpaintPipeline.from_single_file(
            os.path.join(MODELS_DIR, "lustifySDXLNSFW_v20-inpainting.safetensors"),
            torch_dtype=torch.float16 if DEVICE == 'cuda' else torch.float32
        ).to(DEVICE)
        
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        # Continue without models for testing purposes

# Object Detection with YOLO
def detect_objects(image, text_prompt):
    # If in testing mode or model not loaded, return a mock bounding box
    if SKIP_MODEL_LOADING or model is None:
        logger.info("Using mock object detection in testing mode")
        # Return a bounding box covering the center area of the image
        width, height = image.size
        center_x, center_y = width // 2, height // 2
        box_width, box_height = width // 2, height // 2
        x1 = center_x - (box_width // 2)
        y1 = center_y - (box_height // 2)
        x2 = center_x + (box_width // 2)
        y2 = center_y + (box_height // 2)
        return [[x1, y1, x2, y2]]
    
    # Normal processing with the model
    try:
        results = model(image)  # Perform inference on the image
        boxes = []
        
        # Process results in the correct format for YOLO v5/v8
        for result in results:
            for box in result.boxes:
                if box.conf > 0.35:  # Set the confidence threshold
                    x1, y1, x2, y2 = box.xyxy[0].tolist()  # Convert to list
                    boxes.append([x1, y1, x2, y2])
        
        return boxes
    except Exception as e:
        logger.error(f"Error in object detection: {e}")
        # Return a fallback bounding box
        width, height = image.size
        return [[width * 0.25, height * 0.25, width * 0.75, height * 0.75]]

# Segment the image using FastSAM
def segment_image(image, bbox):
    # If in testing mode or model not loaded, return a mock mask
    if SKIP_MODEL_LOADING or fastsam_model is None:
        logger.info("Using mock segmentation in testing mode")
        # Create a simple rectangular mask based on the bounding box
        if isinstance(image, Image.Image):
            width, height = image.size
            mask = np.zeros((height, width), dtype=np.uint8)
            # Extract coordinates from bbox [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, bbox)
            # Fill the bounding box area with 1s (foreground)
            mask[y1:y2, x1:x2] = 1
            return mask
        else:
            # If image is already a numpy array
            height, width = image.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            x1, y1, x2, y2 = map(int, bbox)
            mask[y1:y2, x1:x2] = 1
            return mask
    
    # Normal processing with the model
    try:
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
    except Exception as e:
        logger.error(f"Error in segmentation: {e}")
        # Return a fallback mask (simple rectangle)
        if isinstance(image, Image.Image):
            width, height = image.size
            mask = np.zeros((height, width), dtype=np.uint8)
            x1, y1, x2, y2 = map(int, bbox)
            mask[y1:y2, x1:x2] = 1
            return mask
        else:
            height, width = image.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            x1, y1, x2, y2 = map(int, bbox)
            mask[y1:y2, x1:x2] = 1
            return mask

# Inpaint the image using Stable Diffusion
def inpaint_image(image, mask, prompt):
    # If in testing mode or model not loaded, return a modified version of the original image
    if SKIP_MODEL_LOADING or sd_model is None:
        logger.info("Using mock inpainting in testing mode")
        try:
            # Ensure image is a PIL Image
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
                
            # Ensure mask is in the correct format
            if not isinstance(mask, Image.Image):
                # Convert numpy array to PIL Image
                mask_array = mask.astype(np.uint8) * 255
                mask = Image.fromarray(mask_array)
            
            # Resize mask to match image dimensions if needed
            if mask.size != image.size:
                mask = mask.resize(image.size, Image.NEAREST)
            
            # Create a simple mock inpainting by applying a filter to the masked area
            # Convert mask to proper format (0 for background, 255 for foreground)
            mask_array = np.array(mask)
            if mask_array.max() == 1:
                mask_array = mask_array * 255
                
            # Create a copy of the image to modify
            result_image = image.copy()
            result_array = np.array(result_image)
            
            # Apply a simple effect to the masked area (e.g., grayscale or blur)
            # Here we'll just add a colored overlay to simulate processing
            for i in range(3):  # For each RGB channel
                channel = result_array[:,:,i].copy()
                # Apply a color tint to the masked area
                if i == 0:  # Red channel - increase
                    channel[mask_array > 128] = np.clip(channel[mask_array > 128] * 1.5, 0, 255)
                elif i == 1:  # Green channel - decrease
                    channel[mask_array > 128] = np.clip(channel[mask_array > 128] * 0.8, 0, 255)
                else:  # Blue channel - decrease
                    channel[mask_array > 128] = np.clip(channel[mask_array > 128] * 0.8, 0, 255)
                result_array[:,:,i] = channel
                
            # Convert back to PIL Image
            result_image = Image.fromarray(result_array.astype('uint8'))
            
            # Add text to indicate this is a test
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(result_image)
            try:
                font = ImageFont.truetype("Arial", 20)
            except IOError:
                font = ImageFont.load_default()
                
            # Add watermark text at the bottom
            text = f"TEST MODE - Prompt: {prompt}"
            text_width, text_height = draw.textsize(text, font=font) if hasattr(draw, 'textsize') else (300, 20)
            position = ((result_image.width - text_width) // 2, result_image.height - text_height - 10)
            
            # Draw semi-transparent rectangle behind text
            draw.rectangle(
                [position[0]-5, position[1]-5, position[0]+text_width+5, position[1]+text_height+5],
                fill=(0, 0, 0, 128)
            )
            
            # Draw text
            draw.text(position, text, fill=(255, 255, 255), font=font)
            
            return result_image
        except Exception as e:
            logger.error(f"Error in mock inpainting: {e}")
            # If there's an error, return the original image
            if isinstance(image, Image.Image):
                return image
            else:
                return Image.fromarray(image)
    
    # Normal processing with the model
    try:
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
    except Exception as e:
        logger.error(f"Error in inpainting: {e}")
        # If processing fails, return the original image
        return image

# Main function to run the pipeline
def main(image_url, prompt, output_path):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Check if we're in testing mode
    if SKIP_MODEL_LOADING:
        # In testing mode, just save the original image with a watermark
        logger.info("Running in testing mode. Using mock processing.")
        try:
            # Load the image
            image = load_image_from_url(image_url)
            
            # Add a text watermark to indicate this is a test
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(image)
            
            # Try to load a font, use default if not available
            try:
                font = ImageFont.truetype("Arial", 40)
            except IOError:
                font = ImageFont.load_default()
                
            # Add watermark text
            text = "TEST MODE - NO AI PROCESSING"
            text_width, text_height = draw.textsize(text, font=font) if hasattr(draw, 'textsize') else (300, 40)
            position = ((image.width - text_width) // 2, (image.height - text_height) // 2)
            
            # Draw semi-transparent rectangle behind text
            draw.rectangle(
                [position[0]-10, position[1]-10, position[0]+text_width+10, position[1]+text_height+10],
                fill=(0, 0, 0, 128)
            )
            
            # Draw text
            draw.text(position, text, fill=(255, 255, 255), font=font)
            
            # Save the watermarked image
            image.save(output_path)
            logger.info(f"Test mode: Watermarked image saved to: {output_path}")
            return
        except Exception as e:
            logger.error(f"Error in test mode processing: {e}")
            # If there's an error, continue with normal processing attempt
    
    # If not in test mode or test mode failed, try normal processing
    try:
        # Load the image
        image = load_image_from_url(image_url)
        
        # Detect objects
        boxes = detect_objects(image, prompt)
        
        if not boxes:
            logger.warning("No object detected.")
            # In case of no detection, save the original image
            image.save(output_path)
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
    except Exception as e:
        logger.error(f"Error in main processing: {e}")
        # If processing fails, save the original image as fallback
        try:
            image.save(output_path)
            logger.info(f"Saved original image as fallback due to processing error: {output_path}")
        except Exception as inner_e:
            logger.error(f"Failed to save fallback image: {inner_e}")

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

# Configure Flask app
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(16))
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)  # Session lasts for 7 days
app.config['SESSION_TYPE'] = 'filesystem'

# For testing mode, use a simple in-memory SQLite database
if os.environ.get('SKIP_MODEL_LOADING') == 'true':
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    logger.info("Using in-memory SQLite database for testing mode")
else:
    # Ensure database directory exists and is accessible for production
    db_path = '/app/data/nudify2.db'
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', f'sqlite:///{db_path}')
    logger.info(f"Using database at {db_path}")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
import sys
import os

# Add the project root to sys.path to make imports work properly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Now import the models
from app.backend.models import db, User, ImageGeneration

db.init_app(app)

# Create database tables if they don't exist
with app.app_context():
    # Database directory is already created above
    db.create_all()

# Register authentication blueprint
# Note: sys.path is already set up from the models import above
from app.backend.auth import auth_bp, login_required, check_credits

app.register_blueprint(auth_bp)

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
@login_required
@check_credits(credits_required=1.0)
def process_image():
    """Process an image using the AI pipeline"""
    logger.info("Received image processing request")
    start_time = time.time()
    
    # Get the current user
    user_id = session.get('user_id')
    
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
        
        # Record the image generation in the database
        if user_id:
            user = User.query.get(user_id)
            if user:
                # Deduct credits
                credits_used = 1.0
                user.credit_balance -= credits_used
                
                # Create image generation record
                image_gen = ImageGeneration(
                    user_id=user_id,
                    original_image_url=image_url,
                    result_image_path=output_path,
                    prompt=prompt,
                    credits_used=credits_used
                )
                db.session.add(image_gen)
                db.session.commit()
                logger.info(f"Deducted {credits_used} credits from user {user_id}. New balance: {user.credit_balance}")
        
        # Return the processed image
        return send_file(output_path, mimetype='image/jpeg')
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return jsonify({
            "error": str(e),
            "message": "Failed to process the image. Please try again with a different image or prompt."
        }), 500

@app.route('/upload', methods=['POST'])
@login_required
@check_credits(credits_required=1.0)
def upload_file():
    """Process an uploaded image file"""
    logger.info("Received file upload request")
    start_time = time.time()
    
    # Get the current user
    user_id = session.get('user_id')
    
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
            
            # Record the image generation in the database
            if user_id:
                user = User.query.get(user_id)
                if user:
                    # Deduct credits
                    credits_used = 1.0
                    user.credit_balance -= credits_used
                    
                    # Create image generation record
                    image_gen = ImageGeneration(
                        user_id=user_id,
                        original_image_url=None,  # No URL for uploaded files
                        result_image_path=output_path,
                        prompt=prompt,
                        credits_used=credits_used
                    )
                    db.session.add(image_gen)
                    db.session.commit()
                    logger.info(f"Deducted {credits_used} credits from user {user_id}. New balance: {user.credit_balance}")
            
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

# Add user history endpoint
@app.route('/api/user/history', methods=['GET'])
@login_required
def get_user_history():
    """Get the current user's image generation history"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
        
    # Get user's image generations
    generations = ImageGeneration.query.filter_by(user_id=user_id).order_by(ImageGeneration.created_at.desc()).all()
    
    return jsonify({
        'history': [gen.to_dict() for gen in generations]
    })

# Add credits info endpoint
@app.route('/api/user/credits', methods=['GET'])
@login_required
def get_user_credits():
    """Get the current user's credit balance"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
        
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
        
    return jsonify({
        'credit_balance': user.credit_balance
    })

# Example usage for direct Python execution
if __name__ == "__main__":
    # For development, enable debug mode
    app.run(host='0.0.0.0', port=5000, debug=True)