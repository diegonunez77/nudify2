"""
AI Models for Nudify2 Application

This module contains the implementations of AI models used in the Nudify2 application.
It provides both real implementations and mock implementations for testing.
"""

import os
import logging
import numpy as np
from PIL import Image, ImageFilter
from typing import Union, Tuple, Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if we're in testing mode
TESTING_MODE = os.environ.get('FLASK_ENV') == 'testing'

# Only import heavy dependencies if not in testing mode
if not TESTING_MODE:
    try:
        import torch
        from diffusers import StableDiffusionInpaintPipeline
        from transformers import pipeline
        from ultralytics import YOLO
        from segment_anything_fast import SamPredictor, sam_model_registry
    except ImportError as e:
        logger.warning(f"Could not import AI dependencies: {e}")
        logger.warning("Falling back to mock implementations")
        TESTING_MODE = True  # Force testing mode if imports fail

class BaseNudifyModel:
    """Base class for Nudify models"""
    
    def __init__(self, *args, **kwargs):
        """Initialize the model"""
        self.model_type = "base"
    
    def process(self, image: Image.Image, prompt: str, *args, **kwargs) -> Image.Image:
        """Process an image with a prompt"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def __str__(self):
        return f"{self.model_type.capitalize()} Nudify Model"

class MockNudifyModel(BaseNudifyModel):
    """Mock implementation of the Nudify model for testing"""
    
    def __init__(self, *args, **kwargs):
        """Initialize the mock model"""
        super().__init__(*args, **kwargs)
        self.model_type = "mock"
        logger.info("Initialized Mock Nudify Model")
    
    def process(self, image: Image.Image, prompt: str, *args, **kwargs) -> Image.Image:
        """Process an image with a prompt using a mock implementation"""
        logger.info(f"Processing image with mock model. Prompt: {prompt}")
        
        # Ensure the image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply different filters based on the prompt to simulate different transformations
        if "bikini" in prompt.lower():
            return image.filter(ImageFilter.CONTOUR)
        elif "swimsuit" in prompt.lower():
            return image.filter(ImageFilter.EDGE_ENHANCE)
        elif "lingerie" in prompt.lower():
            return image.filter(ImageFilter.EMBOSS)
        else:
            # Default transformation
            return image.filter(ImageFilter.FIND_EDGES)

class RealNudifyModel(BaseNudifyModel):
    """Real implementation of the Nudify model using AI models"""
    
    def __init__(self, 
                 inpainting_model_path: str = "models/lustifySDXLNSFW_v20-inpainting.safetensors",
                 segmentation_model_path: str = "models/FastSAM-s.pt",
                 detection_model_path: str = "models/yolov5n.pt",
                 device: str = None,
                 *args, **kwargs):
        """Initialize the real model with paths to model weights"""
        super().__init__(*args, **kwargs)
        self.model_type = "real"
        
        # Determine device (CPU or GPU)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load models
        try:
            # 1. Load inpainting model (Stable Diffusion)
            logger.info(f"Loading inpainting model from {inpainting_model_path}")
            self.inpainting_model = StableDiffusionInpaintPipeline.from_pretrained(
                inpainting_model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True
            ).to(self.device)
            
            # 2. Load segmentation model (FastSAM)
            logger.info(f"Loading segmentation model from {segmentation_model_path}")
            self.segmentation_model = sam_model_registry["vit_t"](checkpoint=segmentation_model_path)
            self.segmentation_model.to(self.device)
            self.sam_predictor = SamPredictor(self.segmentation_model)
            
            # 3. Load detection model (YOLO)
            logger.info(f"Loading detection model from {detection_model_path}")
            self.detection_model = YOLO(detection_model_path)
            
            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise RuntimeError(f"Failed to load AI models: {e}")
    
    def detect_person(self, image: Image.Image) -> Tuple[bool, Dict[str, Any]]:
        """Detect if there's a person in the image and return bounding box"""
        # Convert PIL Image to numpy array for YOLO
        img_array = np.array(image)
        
        # Run detection
        results = self.detection_model(img_array, classes=[0])  # Class 0 is 'person' in COCO
        
        # Check if any person was detected
        if len(results[0].boxes) > 0:
            # Get the bounding box with highest confidence
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()
            best_idx = np.argmax(confidences)
            
            # Get coordinates (XYXY format)
            box = boxes.xyxy[best_idx].cpu().numpy()
            
            return True, {
                'box': box,
                'confidence': confidences[best_idx]
            }
        
        return False, {}
    
    def segment_person(self, image: Image.Image, box: np.ndarray) -> np.ndarray:
        """Segment the person using FastSAM"""
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Set the image for the SAM predictor
        self.sam_predictor.set_image(img_array)
        
        # Convert box from XYXY to XYWH format for SAM
        x1, y1, x2, y2 = box
        box_for_sam = np.array([x1, y1, x2-x1, y2-y1])
        
        # Get masks from the box prompt
        masks, _, _ = self.sam_predictor.predict(
            box=box_for_sam,
            multimask_output=True
        )
        
        # Return the first mask (usually the best one)
        return masks[0]
    
    def inpaint_image(self, image: Image.Image, mask: np.ndarray, prompt: str) -> Image.Image:
        """Inpaint the masked area with Stable Diffusion"""
        # Convert numpy mask to PIL Image
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        
        # Ensure images are RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Run inpainting
        result = self.inpainting_model(
            prompt=prompt,
            image=image,
            mask_image=mask_pil,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]
        
        return result
    
    def process(self, image: Image.Image, prompt: str, *args, **kwargs) -> Image.Image:
        """Process an image with a prompt using the real AI models"""
        logger.info(f"Processing image with real model. Prompt: {prompt}")
        
        try:
            # 1. Detect person in the image
            person_detected, detection_info = self.detect_person(image)
            
            if not person_detected:
                logger.warning("No person detected in the image")
                return image  # Return original image if no person detected
            
            # 2. Segment the person
            mask = self.segment_person(image, detection_info['box'])
            
            # 3. Inpaint the segmented area with the prompt
            result_image = self.inpaint_image(image, mask, prompt)
            
            return result_image
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            # Fall back to returning the original image
            return image

def get_nudify_model(*args, **kwargs) -> BaseNudifyModel:
    """Factory function to get the appropriate Nudify model based on environment"""
    if TESTING_MODE:
        return MockNudifyModel(*args, **kwargs)
    else:
        return RealNudifyModel(*args, **kwargs)
