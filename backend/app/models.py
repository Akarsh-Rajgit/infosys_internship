# backend/app/models.py - COMPLETE FIXED VERSION
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, ARRAY, Text
from sqlalchemy.sql import func
from .db import Base

class User(Base):
    """User model for authentication - supports both buyers and sellers"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    user_type = Column(String, nullable=False)
    full_name = Column(String)
    phone = Column(String, nullable=True)
    whatsapp = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

class Property(Base):
    """Property model for real estate listings"""
    __tablename__ = "properties"
    
    id = Column(Integer, primary_key=True, index=True)
    seller_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    property_name = Column(String, nullable=True)
    city_id = Column(Integer, nullable=True)
    location = Column(String, nullable=True)
    area_sqft = Column(Float, nullable=True)
    bhk = Column(Integer, nullable=True)
    is_furnished = Column(Boolean, default=False)
    listing_score = Column(Float, nullable=True)
    property_type = Column(String, nullable=True)
    amenities = Column(ARRAY(String), nullable=True)
    seller_phone = Column(String, nullable=True)
    seller_email = Column(String, nullable=True)
    seller_whatsapp = Column(String, nullable=True)
    images = Column(ARRAY(String), nullable=True)
    status = Column(String, default='active')  # NEW: pending, active, sold
    created_at = Column(DateTime, server_default=func.now())

class Favourite(Base):
    """Favourite properties for buyers"""
    __tablename__ = "favourites"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    property_id = Column(Integer, ForeignKey('properties.id'), nullable=False)
    created_at = Column(DateTime, server_default=func.now())

class Review(Base):
    """Reviews for properties (anonymous to sellers)"""
    __tablename__ = "reviews"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    property_id = Column(Integer, ForeignKey('properties.id'), nullable=False)
    rating = Column(Integer, nullable=False)  # 1-5
    comment = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())