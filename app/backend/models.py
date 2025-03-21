from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    """User account model for storing user-related data"""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    google_id = db.Column(db.String(255), unique=True, nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    display_name = db.Column(db.String(255), nullable=True)
    profile_picture = db.Column(db.String(255), nullable=True)
    credit_balance = db.Column(db.Float, default=10.0)  # Start with 10 credits
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __init__(self, google_id, email, display_name=None, profile_picture=None):
        self.google_id = google_id
        self.email = email
        self.display_name = display_name
        self.profile_picture = profile_picture
    
    def to_dict(self):
        """Convert user object to dictionary"""
        return {
            'id': self.id,
            'email': self.email,
            'display_name': self.display_name,
            'profile_picture': self.profile_picture,
            'credit_balance': self.credit_balance,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }

class ImageGeneration(db.Model):
    """Model for tracking image generation history"""
    __tablename__ = 'image_generations'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    original_image_url = db.Column(db.String(255), nullable=True)
    result_image_path = db.Column(db.String(255), nullable=False)
    prompt = db.Column(db.Text, nullable=False)
    credits_used = db.Column(db.Float, nullable=False, default=1.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    user = db.relationship('User', backref=db.backref('generations', lazy=True))
    
    def to_dict(self):
        """Convert image generation object to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'original_image_url': self.original_image_url,
            'result_image_path': self.result_image_path,
            'prompt': self.prompt,
            'credits_used': self.credits_used,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
