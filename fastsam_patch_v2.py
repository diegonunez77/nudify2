#!/usr/bin/env python3
# This script patches the FastSAM model.py file to work with newer versions of Ultralytics

import os

# Path to the FastSAM model.py file in the container
model_file = '/app/FastSAM/fastsam/model.py'

# Read the current content
with open(model_file, 'r') as f:
    content = f.read()

# Replace old imports with new ones
replacements = [
    ('from ultralytics.yolo.cfg import get_cfg', 'from ultralytics.cfg import get_cfg'),
    ('from ultralytics.yolo.engine.exporter import Exporter', 'from ultralytics.engine.exporter import Exporter'),
    ('from ultralytics.yolo.engine.model import YOLO', 'from ultralytics import YOLO'),
    ('from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, ROOT, is_git_dir', 'from ultralytics.utils import DEFAULT_CFG, LOGGER, ROOT, is_git_dir'),
    ('from ultralytics.yolo.utils.torch_utils import smart_inference_mode', 'from ultralytics.utils.torch_utils import smart_inference_mode'),
]

# Apply all replacements
for old, new in replacements:
    content = content.replace(old, new)

# Write the updated content back to the file
with open(model_file, 'w') as f:
    f.write(content)

print(f"Successfully patched {model_file}")

# Also check and patch the __init__.py file if needed
init_file = '/app/FastSAM/fastsam/__init__.py'
if os.path.exists(init_file):
    with open(init_file, 'r') as f:
        init_content = f.read()
    
    # Replace any old imports
    for old, new in replacements:
        init_content = init_content.replace(old, new)
    
    with open(init_file, 'w') as f:
        f.write(init_content)
    
    print(f"Successfully patched {init_file}")

print("Patching complete!")
