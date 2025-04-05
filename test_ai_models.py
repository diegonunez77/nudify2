#!/usr/bin/env python
"""
Simple test script for AI models
"""

import os
import sys
from PIL import Image
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables for testing
os.environ['FLASK_ENV'] = 'testing'
os.environ['SKIP_MODEL_LOADING'] = 'false'  # We want to test the real models

# Import our AI models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.backend.models.ai_models import get_nudify_model, MockNudifyModel, RealNudifyModel

def test_mock_model():
    """Test the mock model implementation"""
    logger.info("Testing MockNudifyModel...")
    
    # Create a simple test image
    test_image = Image.new('RGB', (512, 512), color='white')
    
    # Initialize the mock model
    model = MockNudifyModel()
    
    # Process the image
    result = model.process(test_image, "Test prompt")
    
    # Check the result
    assert result is not None, "Mock model returned None"
    assert isinstance(result, Image.Image), "Mock model did not return a PIL Image"
    
    logger.info("MockNudifyModel test passed!")
    return result

def test_real_model_with_mock_dependencies():
    """Test the real model implementation with mocked dependencies"""
    logger.info("Testing RealNudifyModel with mock dependencies...")
    
    # Create a simple test image
    test_image = Image.new('RGB', (512, 512), color='white')
    
    try:
        # Initialize the real model with mock dependencies
        model = RealNudifyModel()
        model._initialize_mock_dependencies()
        
        # Process the image
        result = model.process(test_image, "Test prompt")
        
        # Check the result
        assert result is not None, "Real model returned None"
        assert isinstance(result, Image.Image), "Real model did not return a PIL Image"
        
        logger.info("RealNudifyModel with mock dependencies test passed!")
        return result
    except Exception as e:
        logger.error(f"Error testing real model: {e}")
        raise

def test_factory_function():
    """Test the model factory function"""
    logger.info("Testing get_nudify_model factory function...")
    
    # Get a model using the factory function
    model = get_nudify_model()
    
    # Check the model type
    if os.environ.get('FLASK_ENV') == 'testing':
        assert isinstance(model, MockNudifyModel), "Factory function did not return a MockNudifyModel in testing mode"
    else:
        assert isinstance(model, RealNudifyModel), "Factory function did not return a RealNudifyModel in production mode"
    
    logger.info("Factory function test passed!")
    return model

if __name__ == "__main__":
    logger.info("Starting AI model tests...")
    
    # Test the mock model
    mock_result = test_mock_model()
    mock_result.save("mock_model_output.png")
    logger.info("Saved mock model output to mock_model_output.png")
    
    # Test the real model with mock dependencies
    try:
        real_result = test_real_model_with_mock_dependencies()
        real_result.save("real_model_output.png")
        logger.info("Saved real model output to real_model_output.png")
    except Exception as e:
        logger.error(f"Real model test failed: {e}")
    
    # Test the factory function
    factory_model = test_factory_function()
    logger.info(f"Factory function returned a {type(factory_model).__name__}")
    
    logger.info("All tests completed!")
