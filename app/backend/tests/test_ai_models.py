"""
Tests for the AI models in the Nudify2 application.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np
import sys
import io

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Mock the imports before importing the module
sys.modules['torch'] = MagicMock()
sys.modules['diffusers'] = MagicMock()
sys.modules['diffusers.StableDiffusionInpaintPipeline'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['ultralytics'] = MagicMock()
sys.modules['ultralytics.YOLO'] = MagicMock()
sys.modules['segment_anything_fast'] = MagicMock()
sys.modules['segment_anything_fast.SamPredictor'] = MagicMock()
sys.modules['segment_anything_fast.sam_model_registry'] = MagicMock()

# Now import the module
from app.backend.models.ai_models import (
    BaseNudifyModel, 
    MockNudifyModel, 
    RealNudifyModel, 
    get_nudify_model
)

class TestBaseNudifyModel(unittest.TestCase):
    """Test the base Nudify model class"""
    
    def test_init(self):
        """Test initialization of the base model"""
        model = BaseNudifyModel()
        self.assertEqual(model.model_type, "base")
    
    def test_process_not_implemented(self):
        """Test that process method raises NotImplementedError"""
        model = BaseNudifyModel()
        with self.assertRaises(NotImplementedError):
            model.process(Image.new('RGB', (100, 100)), "test prompt")

class TestMockNudifyModel(unittest.TestCase):
    """Test the mock Nudify model class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = MockNudifyModel()
        self.test_image = Image.new('RGB', (100, 100), color='white')
    
    def test_init(self):
        """Test initialization of the mock model"""
        self.assertEqual(self.model.model_type, "mock")
    
    def test_process_with_bikini_prompt(self):
        """Test processing with bikini prompt"""
        result = self.model.process(self.test_image, "bikini")
        # Verify result is a PIL Image
        self.assertIsInstance(result, Image.Image)
        # Verify image dimensions are preserved
        self.assertEqual(result.size, self.test_image.size)
    
    def test_process_with_swimsuit_prompt(self):
        """Test processing with swimsuit prompt"""
        result = self.model.process(self.test_image, "swimsuit")
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.test_image.size)
    
    def test_process_with_lingerie_prompt(self):
        """Test processing with lingerie prompt"""
        result = self.model.process(self.test_image, "lingerie")
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.test_image.size)
    
    def test_process_with_default_prompt(self):
        """Test processing with a prompt that doesn't match any specific filter"""
        result = self.model.process(self.test_image, "generic prompt")
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.test_image.size)
    
    def test_process_with_non_rgb_image(self):
        """Test processing with a non-RGB image"""
        # Create a grayscale image
        gray_image = Image.new('L', (100, 100), color=128)
        result = self.model.process(gray_image, "test prompt")
        # Verify result is a PIL Image
        self.assertIsInstance(result, Image.Image)
        # Verify image was converted to RGB
        self.assertEqual(result.mode, 'RGB')
        # Verify image dimensions are preserved
        self.assertEqual(result.size, gray_image.size)

class TestRealNudifyModel(unittest.TestCase):
    """Test the real Nudify model class with mocked dependencies"""
    
    def setUp(self):
        """Set up test fixtures with mocked dependencies"""
        # Mock torch and CUDA
        torch_mock = sys.modules['torch']
        cuda_mock = MagicMock()
        cuda_mock.is_available.return_value = False
        torch_mock.cuda = cuda_mock
        torch_mock.float16 = MagicMock()
        torch_mock.float32 = MagicMock()
        
        # Mock StableDiffusionInpaintPipeline
        diffusers_mock = sys.modules['diffusers']
        self.mock_inpainting_pipeline = MagicMock()
        diffusers_mock.StableDiffusionInpaintPipeline = self.mock_inpainting_pipeline
        self.mock_inpainting_pipeline.from_pretrained.return_value.to.return_value = MagicMock()
        
        # Mock SAM model registry and predictor
        sam_registry_mock = sys.modules['segment_anything_fast.sam_model_registry']
        sam_registry_mock.__getitem__.return_value = MagicMock()
        
        self.mock_sam_predictor = MagicMock()
        sam_predictor_class_mock = sys.modules['segment_anything_fast.SamPredictor']
        sam_predictor_class_mock.return_value = self.mock_sam_predictor
        
        # Mock YOLO
        yolo_mock = sys.modules['ultralytics.YOLO']
        self.mock_detection_model = MagicMock()
        yolo_mock.return_value = self.mock_detection_model
        
        # Create a test image
        self.test_image = Image.new('RGB', (100, 100), color='white')
        
        # Create the model with fake paths
        self.model = RealNudifyModel(
            inpainting_model_path="fake_path",
            segmentation_model_path="fake_path",
            detection_model_path="fake_path"
        )
    
    def test_init(self):
        """Test initialization of the real model"""
        self.assertEqual(self.model.model_type, "real")
    
    def test_detect_person_found(self):
        """Test person detection when a person is found"""
        # Create a mock box with the expected structure
        mock_boxes = MagicMock()
        mock_boxes.__len__.return_value = 1
        mock_boxes.conf = MagicMock()
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.9])
        mock_boxes.xyxy = MagicMock()
        mock_boxes.xyxy.__getitem__.return_value.cpu.return_value.numpy.return_value = np.array([10, 20, 50, 80])
        
        # Create a mock result with the expected structure
        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        
        # Create a new mock for the detection model
        mock_detection_model = MagicMock()
        mock_detection_model.return_value = [mock_result]
        
        # Replace the detection model
        self.model.detection_model = mock_detection_model
        
        # Test the detection
        found, info = self.model.detect_person(self.test_image)
        
        # Verify results
        self.assertTrue(found)
        self.assertIn('box', info)
        self.assertIn('confidence', info)
        np.testing.assert_array_equal(info['box'], np.array([10, 20, 50, 80]))
        self.assertEqual(info['confidence'], 0.9)
    
    def test_detect_person_not_found(self):
        """Test person detection when no person is found"""
        # Mock empty detection results
        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.__len__.return_value = 0
        
        # Create a new mock for the detection model
        mock_detection_model = MagicMock()
        mock_detection_model.return_value = [mock_result]
        
        # Replace the detection model
        self.model.detection_model = mock_detection_model
        
        # Test the detection
        found, info = self.model.detect_person(self.test_image)
        
        # Verify results
        self.assertFalse(found)
        self.assertEqual(info, {})
    
    def test_segment_person(self):
        """Test person segmentation"""
        # Create a mock box
        box = np.array([10, 20, 50, 80])
        
        # Create a mock mask
        mock_mask = np.zeros((100, 100), dtype=bool)
        mock_mask[30:70, 30:70] = True
        
        # Create a mock for the SAM predictor's predict method
        self.model.sam_predictor = self.mock_sam_predictor
        self.mock_sam_predictor.predict = MagicMock(return_value=(np.array([mock_mask]), None, None))
        
        # Test the segmentation
        result = self.model.segment_person(self.test_image, box)
        
        # Verify results
        self.assertEqual(result.shape, (100, 100))
        np.testing.assert_array_equal(result[30:70, 30:70], True)
        np.testing.assert_array_equal(result[0:30, 0:30], False)
    
    def test_inpaint_image(self):
        """Test image inpainting"""
        # Create a mock mask
        mock_mask = np.zeros((100, 100), dtype=bool)
        mock_mask[30:70, 30:70] = True
        
        # Mock inpainting results
        mock_result_image = Image.new('RGB', (100, 100), color='blue')
        mock_result = MagicMock()
        mock_result.images = [mock_result_image]
        
        # Create a new mock for the inpainting model
        mock_inpainting_model = MagicMock()
        mock_inpainting_model.return_value = mock_result
        
        # Replace the inpainting model
        self.model.inpainting_model = mock_inpainting_model
        
        # Test the inpainting
        result = self.model.inpaint_image(self.test_image, mock_mask, "test prompt")
        
        # Verify results
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (100, 100))
        
        # Verify the inpainting model was called correctly
        self.model.inpainting_model.assert_called_once()
    
    def test_process_success(self):
        """Test the full processing pipeline when everything succeeds"""
        # Mock detection results
        mock_boxes = MagicMock()
        mock_boxes.__len__.return_value = 1
        mock_boxes.conf = MagicMock()
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.9])
        mock_boxes.xyxy = MagicMock()
        mock_boxes.xyxy.__getitem__.return_value.cpu.return_value.numpy.return_value = np.array([10, 20, 50, 80])
        
        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        
        # Create a new mock for the detection model
        mock_detection_model = MagicMock()
        mock_detection_model.return_value = [mock_result]
        
        # Replace the detection model
        self.model.detection_model = mock_detection_model
        
        # Mock segmentation results
        mock_mask = np.zeros((100, 100), dtype=bool)
        mock_mask[30:70, 30:70] = True
        
        # Create a mock for the SAM predictor's predict method
        self.model.sam_predictor = self.mock_sam_predictor
        self.mock_sam_predictor.predict = MagicMock(return_value=(np.array([mock_mask]), None, None))
        
        # Mock inpainting results
        mock_result_image = Image.new('RGB', (100, 100), color='blue')
        mock_inpaint_result = MagicMock()
        mock_inpaint_result.images = [mock_result_image]
        
        # Create a new mock for the inpainting model
        mock_inpainting_model = MagicMock()
        mock_inpainting_model.return_value = mock_inpaint_result
        
        # Replace the inpainting model
        self.model.inpainting_model = mock_inpainting_model
        
        # Test the full process
        result = self.model.process(self.test_image, "test prompt")
        
        # Verify results
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (100, 100))
    
    def test_process_no_person_detected(self):
        """Test processing when no person is detected"""
        # Mock empty detection results
        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.__len__.return_value = 0
        
        # Create a new mock for the detection model
        mock_detection_model = MagicMock()
        mock_detection_model.return_value = [mock_result]
        
        # Replace the detection model
        self.model.detection_model = mock_detection_model
        
        # Test the process
        result = self.model.process(self.test_image, "test prompt")
        
        # Verify the original image is returned
        self.assertEqual(result, self.test_image)
    
    def test_process_exception_handling(self):
        """Test exception handling during processing"""
        # Create a mock that raises an exception when called
        mock_detection_model = MagicMock()
        mock_detection_model.side_effect = Exception("Test error")
        
        # Replace the detection model
        self.model.detection_model = mock_detection_model
        
        # Test the process
        result = self.model.process(self.test_image, "test prompt")
        
        # Verify the original image is returned
        self.assertEqual(result, self.test_image)

class TestGetNudifyModel(unittest.TestCase):
    """Test the get_nudify_model factory function"""
    
    def test_get_mock_model_in_testing_mode(self):
        """Test that get_nudify_model returns a MockNudifyModel in testing mode"""
        # Save original value
        original_env = os.environ.get('FLASK_ENV')
        
        try:
            # Set testing mode
            os.environ['FLASK_ENV'] = 'testing'
            
            # Force TESTING_MODE to True for this test
            import app.backend.models.ai_models
            app.backend.models.ai_models.TESTING_MODE = True
            
            # Get the model
            model = get_nudify_model()
            
            # Verify it's a MockNudifyModel
            self.assertIsInstance(model, MockNudifyModel)
        finally:
            # Reset to original value
            if original_env:
                os.environ['FLASK_ENV'] = original_env
            else:
                del os.environ['FLASK_ENV']
    
    def test_get_real_model_in_production_mode(self):
        """Test that get_nudify_model returns a RealNudifyModel in production mode"""
        # Save original value
        original_env = os.environ.get('FLASK_ENV')
        
        try:
            # Set production mode
            os.environ['FLASK_ENV'] = 'production'
            
            # Setup torch mock
            torch_mock = sys.modules['torch']
            cuda_mock = MagicMock()
            cuda_mock.is_available.return_value = False
            torch_mock.cuda = cuda_mock
            
            # Force TESTING_MODE to False for this test
            import app.backend.models.ai_models
            app.backend.models.ai_models.TESTING_MODE = False
            
            # Get the model
            model = get_nudify_model()
            
            # Verify it's a RealNudifyModel
            self.assertIsInstance(model, RealNudifyModel)
        finally:
            # Reset to original value
            if original_env:
                os.environ['FLASK_ENV'] = original_env
            else:
                del os.environ['FLASK_ENV']
                
            # Reset TESTING_MODE
            app.backend.models.ai_models.TESTING_MODE = True

if __name__ == '__main__':
    unittest.main()
