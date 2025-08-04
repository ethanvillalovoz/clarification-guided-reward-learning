"""
Image Border Generator

This utility script adds a red border to images used in the clarification-guided reward learning project.
The border visually indicates when an object is being "held" in the simulation.

Usage:
    python image_border_generator.py

Dependencies:
    - PIL (Python Imaging Library / Pillow)
"""

from PIL import Image, ImageOps
import os
import sys


def add_border_to_image(input_path, output_path, border_width=20, border_color="#f00"):
    """
    Add a colored border to an image and save the result.
    
    Args:
        input_path (str): Path to the input image
        output_path (str): Path to save the output image
        border_width (int): Width of border in pixels
        border_color (str): Hex color code for the border
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Open the original image
        original = Image.open(input_path)
        
        # Add the border
        bordered = ImageOps.expand(original, border=border_width, fill=border_color)
        
        # Save the bordered image
        bordered.save(output_path)
        
        print(f"Successfully created bordered image: {output_path}")
        return True
    except Exception as e:
        print(f"Error processing image: {e}")
        return False


# If this script is run directly (not imported)
if __name__ == "__main__":
    # Define base directory for images
    base_dir = 'data/images'
    
    # Define images to process with their input and output paths
    images_to_process = [
        {'input': 'yellowcup_180.jpeg', 'output': 'yellowcup_180_holding.jpeg'},
        {'input': 'yellowcup.jpeg', 'output': 'yellowcup_holding.jpeg'},
        {'input': 'redcup_180.jpeg', 'output': 'redcup_180_holding.jpeg'},
        {'input': 'redcup.jpeg', 'output': 'redcup_holding.jpeg'}
    ]
    
    # Process each image
    for img in images_to_process:
        input_path = os.path.join(base_dir, img['input'])
        output_path = os.path.join(base_dir, img['output'])
        add_border_to_image(input_path, output_path)