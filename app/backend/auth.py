from flask import Blueprint, request, jsonify, session, redirect, url_for, current_app
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import os
import json
from datetime import datetime, timedelta
import secrets
from functools import wraps

from .models import db, User

# Create a Blueprint for authentication routes
auth_bp = Blueprint('auth', __name__)

# Function to get Google client ID from environment or config
def get_google_client_id():
    return os.environ.get('GOOGLE_CLIENT_ID', 'YOUR_GOOGLE_CLIENT_ID')

# Decorator to require authentication for routes
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

# Check if user has sufficient credits
def check_credits(credits_required=1.0):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                return jsonify({'error': 'Authentication required'}), 401
            
            user = User.query.get(session['user_id'])
            if not user:
                return jsonify({'error': 'User not found'}), 404
                
            if user.credit_balance < credits_required:
                return jsonify({
                    'error': 'Insufficient credits',
                    'credits_required': credits_required,
                    'credits_available': user.credit_balance
                }), 403
                
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@auth_bp.route('/auth/google', methods=['POST'])
def google_auth():
    """Handle Google Sign-In token verification and user authentication"""
    try:
        # Get token from request
        token = request.json.get('id_token')
        if not token:
            return jsonify({'error': 'No token provided'}), 400
            
        # Verify the token
        client_id = get_google_client_id()
        id_info = id_token.verify_oauth2_token(
            token, 
            google_requests.Request(), 
            client_id
        )
        
        # Check if token is valid
        if id_info['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
            return jsonify({'error': 'Invalid token issuer'}), 401
            
        # Get user info from token
        google_id = id_info['sub']
        email = id_info['email']
        
        # Optional user info
        display_name = id_info.get('name')
        profile_picture = id_info.get('picture')
        
        # Check if user exists
        user = User.query.filter_by(google_id=google_id).first()
        
        if not user:
            # Create new user
            user = User(
                google_id=google_id,
                email=email,
                display_name=display_name,
                profile_picture=profile_picture
            )
            db.session.add(user)
            db.session.commit()
        else:
            # Update existing user's last login time
            user.last_login = datetime.utcnow()
            # Update profile info in case it changed
            user.display_name = display_name
            user.profile_picture = profile_picture
            db.session.commit()
        
        # Set user session
        session['user_id'] = user.id
        session.permanent = True
        current_app.permanent_session_lifetime = timedelta(days=7)  # Session lasts for 7 days
        
        # Return user info
        return jsonify({
            'success': True,
            'user': user.to_dict()
        })
        
    except Exception as e:
        current_app.logger.error(f"Google auth error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@auth_bp.route('/auth/logout', methods=['POST'])
def logout():
    """Log out the current user"""
    session.pop('user_id', None)
    return jsonify({'success': True})

@auth_bp.route('/auth/user', methods=['GET'])
@login_required
def get_current_user():
    """Get the current authenticated user"""
    user = User.query.get(session['user_id'])
    if not user:
        session.pop('user_id', None)
        return jsonify({'error': 'User not found'}), 404
        
    return jsonify({'user': user.to_dict()})

@auth_bp.route('/auth/deduct-credits', methods=['POST'])
@login_required
def deduct_credits():
    """Deduct credits from user account"""
    try:
        amount = float(request.json.get('amount', 1.0))
        
        user = User.query.get(session['user_id'])
        if not user:
            return jsonify({'error': 'User not found'}), 404
            
        if user.credit_balance < amount:
            return jsonify({
                'error': 'Insufficient credits',
                'credits_required': amount,
                'credits_available': user.credit_balance
            }), 403
            
        user.credit_balance -= amount
        db.session.commit()
        
        return jsonify({
            'success': True,
            'new_balance': user.credit_balance
        })
        
    except Exception as e:
        current_app.logger.error(f"Error deducting credits: {str(e)}")
        return jsonify({'error': str(e)}), 500
