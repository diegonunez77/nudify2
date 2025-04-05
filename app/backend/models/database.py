"""
Database models for Nudify2
"""

import datetime
from flask_sqlalchemy import SQLAlchemy

# Initialize SQLAlchemy
db = SQLAlchemy()

class User(db.Model):
    """User model for authentication and credit tracking"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    name = db.Column(db.String(255), nullable=True)
    google_id = db.Column(db.String(255), unique=True, nullable=True)
    credit_balance = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    last_login = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    image_generations = db.relationship('ImageGeneration', backref='user', lazy=True)
    
    def __repr__(self):
        return f'<User {self.email}>'

class FreeTrialUsage(db.Model):
    """Track free trial usage by users or clients"""
    __tablename__ = 'free_trial_usage'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    client_id = db.Column(db.String(255), nullable=True, unique=True)
    used_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    def __repr__(self):
        return f'<FreeTrialUsage user_id={self.user_id}, client_id={self.client_id}>'

class ImageGeneration(db.Model):
    """Track image generation requests"""
    __tablename__ = 'image_generations'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    prompt = db.Column(db.Text, nullable=True)
    input_image_path = db.Column(db.String(255), nullable=True)
    output_image_path = db.Column(db.String(255), nullable=True)
    credits_used = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    def __repr__(self):
        return f'<ImageGeneration id={self.id}, user_id={self.user_id}>'
