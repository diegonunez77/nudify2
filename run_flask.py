#!/usr/bin/env python3
# This script runs the Flask application with the correct Python path

import os
import sys

# Add the FastSAM directory to the Python path
sys.path.insert(0, '/app/FastSAM')

# Set environment variables
os.environ['FLASK_APP'] = '/app/app/backend/app.py'
os.environ['FLASK_ENV'] = 'production'
os.environ['FLASK_DEBUG'] = '0'

# Run the Flask application
os.system('flask run --host=0.0.0.0')
