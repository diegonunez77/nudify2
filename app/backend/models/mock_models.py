"""
Mock implementations of AI models for development and testing
"""

import os
import time
import random
import math
from PIL import Image, ImageDraw, ImageFont, ImageFilter

class MockNudifyModel:
    """Mock implementation of the Nudify AI model"""
    
    def __init__(self):
        """Initialize the mock model"""
        print("Initializing MockNudifyModel (no actual AI model loading)")
    
    def process_image(self, image, prompt=None):
        """
        Process an image with the mock model
        
        Args:
            image: PIL Image to process
            prompt: Optional text prompt
            
        Returns:
            Processed PIL Image
        """
        # Create a copy of the image and ensure it's in RGB mode
        processed = image.copy()
        if processed.mode != 'RGB':
            processed = processed.convert('RGB')
        
        # Add a text overlay to indicate this is a mock result
        draw = ImageDraw.Draw(processed)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", size=int(processed.width/20))
        except IOError:
            font = ImageFont.load_default()
        
        # Add text to indicate this is a mock result
        mock_text = "MOCK AI RESULT"
        
        # Handle both old and new Pillow versions for text size calculation
        if hasattr(draw, 'textsize'):
            text_width, text_height = draw.textsize(mock_text, font)
        elif hasattr(draw, 'textbbox'):
            left, top, right, bottom = draw.textbbox((0, 0), mock_text, font)
            text_width, text_height = right - left, bottom - top
        else:
            text_width, text_height = processed.width//3, processed.height//20
        
        # Position text in the top-left corner
        position = (20, 20)
        
        # Draw background for text
        padding = 10
        draw.rectangle(
            [position[0]-padding, position[1]-padding, position[0]+text_width+padding, position[1]+text_height+padding],
            fill=(0, 0, 0)  # Black
        )
        
        # Draw text
        draw.text(position, mock_text, fill=(255, 255, 255), font=font)
        
        # Apply a subtle filter to make it look "processed"
        processed = processed.filter(ImageFilter.CONTOUR)
        
        # Simulate processing time
        time.sleep(0.5)
        
        return processed

def get_mock_model():
    """Get an instance of the mock model"""
    return MockNudifyModel()

# Function to apply free trial blur and watermark
def apply_free_trial_effects(image):
    """
    Apply blur and watermark effects for free trial users
    
    Args:
        image: PIL Image to process
        
    Returns:
        Processed PIL Image with blur and watermark
    """
    # Create a copy of the image and ensure it's in RGB mode
    watermarked = image.copy()
    if watermarked.mode != 'RGB':
        watermarked = watermarked.convert('RGB')
    
    # Create a draw object for the image
    draw = ImageDraw.Draw(watermarked)
    
    # Add watermark text
    try:
        font = ImageFont.truetype("arial.ttf", size=int(watermarked.width/30))
    except IOError:
        font = ImageFont.load_default()
    
    # Add watermark text
    watermark_text = "FREE TRIAL - SIGN IN TO UNLOCK"
    
    # Handle both old and new Pillow versions for text size calculation
    if hasattr(draw, 'textsize'):
        text_width, text_height = draw.textsize(watermark_text, font)
    elif hasattr(draw, 'textbbox'):
        left, top, right, bottom = draw.textbbox((0, 0), watermark_text, font)
        text_width, text_height = right - left, bottom - top
    else:
        text_width, text_height = int(watermarked.width/3), int(watermarked.height/20)
    
    # Add watermark in bottom right corner
    position = (watermarked.width - text_width - 20, watermarked.height - text_height - 20)
    
    # Draw background for text
    padding = 10
    draw.rectangle(
        [position[0]-padding, position[1]-padding, position[0]+text_width+padding, position[1]+text_height+padding],
        fill=(0, 0, 0)  # Black
    )
    
    # Draw text
    draw.text(position, watermark_text, fill=(255, 255, 255), font=font)
    
    # Add a small logo/watermark in the center
    logo_size = min(watermarked.width, watermarked.height) // 10
    logo_position = ((watermarked.width - logo_size) // 2, (watermarked.height - logo_size) // 2)
    
    # Draw a logo circle
    draw.ellipse(
        [logo_position[0], logo_position[1], logo_position[0] + logo_size, logo_position[1] + logo_size],
        fill=(255, 255, 255)  # White
    )
    
    # Draw logo text
    logo_text = "TRIAL"
    logo_font_size = logo_size // 3
    try:
        logo_font = ImageFont.truetype("arial.ttf", size=logo_font_size)
    except IOError:
        logo_font = ImageFont.load_default()
    
    # Handle both old and new Pillow versions for logo text size
    if hasattr(draw, 'textsize'):
        logo_text_width, logo_text_height = draw.textsize(logo_text, logo_font)
    elif hasattr(draw, 'textbbox'):
        left, top, right, bottom = draw.textbbox((0, 0), logo_text, logo_font)
        logo_text_width, logo_text_height = right - left, bottom - top
    else:
        logo_text_width, logo_text_height = int(logo_size * 0.7), int(logo_size * 0.3)
    
    logo_text_position = (logo_position[0] + (logo_size - logo_text_width) // 2, logo_position[1] + (logo_size - logo_text_height) // 2)
    draw.text(logo_text_position, logo_text, fill=(0, 0, 0), font=logo_font)
    
    # Create a strategically blurred image that reveals enough to entice but protects from extraction
    width, height = watermarked.size
    
    # Create a frosted glass effect that looks like viewing through textured glass
    # This creates a high-quality blur that preserves details while looking premium
    
    # Start with the original image
    result_image = watermarked.copy()
    
    # Step 1: Create a base blur for the frosted glass effect
    base_blur = result_image.copy().filter(ImageFilter.GaussianBlur(radius=10))
    
    # Step 2: Create a noise texture to simulate frosted glass
    noise = Image.new('RGBA', result_image.size, (0, 0, 0, 0))
    noise_draw = ImageDraw.Draw(noise)
    
    # Generate random noise pattern
    for y in range(0, height, 2):
        for x in range(0, width, 2):
            # Random transparency for noise texture
            alpha = random.randint(5, 30)
            # Slightly bluish-white tint for glass effect
            noise_draw.point((x, y), fill=(240, 245, 255, alpha))
    
    # Step 3: Apply a slight blur to the noise to make it smoother
    noise = noise.filter(ImageFilter.GaussianBlur(radius=1))
    
    # Step 4: Convert the original image to RGBA for compositing
    if result_image.mode != 'RGBA':
        result_image = result_image.convert('RGBA')
    
    # Step 5: Create a frosted glass mask with variable transparency
    glass_mask = Image.new('RGBA', result_image.size, (255, 255, 255, 0))
    mask_draw = ImageDraw.Draw(glass_mask)
    
    # Create a semi-transparent white overlay with varying opacity
    for y in range(height):
        for x in range(width):
            # Create varying transparency (more transparent in some areas)
            # This creates the illusion of shine and highlights on the glass
            distance_from_corner = math.sqrt((x - width*0.8)**2 + (y - height*0.2)**2)
            max_distance = math.sqrt(width**2 + height**2)
            normalized_distance = distance_from_corner / max_distance
            
            # Create a gradient of transparency (more transparent at the "light source")
            alpha = int(100 + 80 * normalized_distance)  # 100-180 range for alpha
            mask_draw.point((x, y), fill=(255, 255, 255, alpha))
    
    # Step 6: Composite the layers together
    # First apply the base blur
    result_image = Image.alpha_composite(result_image, glass_mask)
    
    # Then apply the noise texture for the frosted effect
    result_image = Image.alpha_composite(result_image, noise)
    
    # Step 7: Add a subtle highlight/shine effect in the corner
    highlight = Image.new('RGBA', result_image.size, (0, 0, 0, 0))
    highlight_draw = ImageDraw.Draw(highlight)
    
    # Create a radial gradient for the highlight
    highlight_x, highlight_y = int(width * 0.2), int(height * 0.2)  # Top-left area
    highlight_radius = int(min(width, height) * 0.3)
    
    for y in range(height):
        for x in range(width):
            distance = math.sqrt((x - highlight_x)**2 + (y - highlight_y)**2)
            if distance < highlight_radius:
                # Create a gradient that fades out from the center
                alpha = int(40 * (1 - distance / highlight_radius))
                highlight_draw.point((x, y), fill=(255, 255, 255, alpha))
    
    # Apply the highlight
    result_image = Image.alpha_composite(result_image, highlight)
    
    # Convert back to RGB for further processing
    result_image = result_image.convert('RGB')
    
    # Apply a slight sharpening to enhance details
    result_image = result_image.filter(ImageFilter.SHARPEN)
    
    # Define strategic blur regions that protect key areas while showing transformation
    
    # Calculate dimensions for a central blur region
    center_width = int(width * 0.4)  # Cover 40% of width
    center_height = int(height * 0.4)  # Cover 40% of height
    center_x = (width - center_width) // 2
    center_y = (height - center_height) // 2
    
    # Create a mask for smooth transitions
    mask = Image.new('L', (width, height), 0)  # Black background (transparent)
    mask_draw = ImageDraw.Draw(mask)
    
    # Draw a radial gradient in the center
    # This creates a smooth transition from clear to blurred
    for i in range(50):
        # Create concentric circles with increasing opacity
        ellipse_size = min(center_width, center_height) - i * 2
        if ellipse_size <= 0:
            break
            
        # Calculate opacity based on distance from center
        # Outer rings are more transparent, inner rings are more opaque
        opacity = int(255 * (1 - i / 50))
        
        # Draw the ellipse
        ellipse_bounds = [
            center_x + i, 
            center_y + i, 
            center_x + center_width - i, 
            center_y + center_height - i
        ]
        mask_draw.ellipse(ellipse_bounds, fill=opacity)
    
    # Add some strategic blur spots in other areas
    for _ in range(3):
        spot_size = min(width, height) // 10
        spot_x = random.randint(0, width - spot_size)
        spot_y = random.randint(0, height - spot_size)
        
        # Skip if too close to the center region
        if (abs(spot_x - center_x) < center_width and 
            abs(spot_y - center_y) < center_height):
            continue
            
        # Create a circular gradient for each spot
        for i in range(spot_size // 2):
            opacity = int(255 * (1 - i / (spot_size // 2)))
            spot_bounds = [
                spot_x + i, 
                spot_y + i, 
                spot_x + spot_size - i, 
                spot_y + spot_size - i
            ]
            mask_draw.ellipse(spot_bounds, fill=opacity)
    
    # Apply the medium blur using the mask
    result_image.paste(medium_blur, (0, 0), mask)
    
    # Add a subtle noise texture for security
    # This makes it harder to extract the image through browser tools
    noise = Image.new('RGB', (width, height), (0, 0, 0))
    noise_draw = ImageDraw.Draw(noise)
    
    # Add subtle noise dots
    for _ in range(width * height // 200):  # Reduced number of dots for less visual interference
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        size = random.randint(1, 2)  # Smaller dots
        # Use more subtle colors
        color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
        noise_draw.ellipse([x, y, x + size, y + size], fill=color)
    
    # Blend the noise with very low opacity
    result_image = Image.blend(result_image, noise, 0.02)
    
    # Use the strategically protected image
    watermarked = result_image
    
    return watermarked
