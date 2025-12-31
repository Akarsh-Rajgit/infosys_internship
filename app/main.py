# backend/app/main.py
from fastapi import FastAPI, Depends, HTTPException, Body, UploadFile, File, Header
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from . import db, models
from pydantic import BaseModel, EmailStr
import joblib
import os
import pandas as pd
from typing import Optional, List, Any, Dict
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from passlib.context import CryptContext
from datetime import datetime, timedelta
import json
from jose import JWTError, jwt
import ollama
import shutil
import uuid
import asyncio
import re

app = FastAPI()

# Add this after creating the app
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# CORS Configuration
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication Configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = "7c60338a79efd487573e5af99ecf9266fc6eefee22f15bad7f1fac8ec90b2eed"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

# Create uploads directory
UPLOAD_DIR = "uploads/properties"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load ML Model Pipeline
PIPELINE_PATH = "C:\\Users\\AKARSH RAJ M H\\OneDrive\\Desktop\\Akarsh_Infosys_internship\\milestone 3\\real-estate-app\\backend\\models\\pipeline_v1.pkl"

pipeline = None
if os.path.exists(PIPELINE_PATH):
    try:
        pipeline = joblib.load(PIPELINE_PATH)
        print("Pipeline loaded successfully.")
    except Exception as e:
        print("Failed to load pipeline:", e)
else:
    print("Pipeline file not found at:", PIPELINE_PATH)

FEATURES_42 = [
    'area_num', 'bhk', 'listing_domain_score', 'is_furnished', 'is_rera_registered',
    'is_apartment', 'price_per_bhk', 'area_per_bhk', 'price_per_bath', 'demand_density',
    'price_dev_locality', 'luxury_index', 'dist_city_center',
    'loc_Andheri West', 'loc_Borivali West', 'loc_Chembur', 'loc_Dombivali',
    'loc_Dwarka Mor', 'loc_Kalyan West', 'loc_Kandivali East', 'loc_Kandivali West',
    'loc_Kharghar', 'loc_Malad West', 'loc_Mira Road East', 'loc_New Town',
    'loc_Panvel', 'loc_Powai', 'loc_Rajarhat', 'loc_Thane West', 'loc_Ulwe',
    'loc_Uttam Nagar', 'loc_Virar', 'loc_other',
    'city_Bangalore', 'city_Chennai', 'city_Delhi', 'city_Hyderabad',
    'city_Kolkata', 'city_Lucknow', 'city_Mumbai',
    'area_cat_medium', 'area_cat_large'
]

# Load Google API key securely
# Initialize Gemini client

def get_db():
    db_sess = db.SessionLocal()
    try:
        yield db_sess
    finally:
        db_sess.close()

# FIXED: Token extraction helper
def get_token_from_header(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    try:
        scheme, token = authorization.split()
        if scheme.lower() != 'bearer':
            raise HTTPException(status_code=401, detail="Invalid authentication scheme")
        return token
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header format")

def get_current_user_from_token(authorization: str = Header(None)):
    token = get_token_from_header(authorization)
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Pydantic Models
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    user_type: str
    full_name: str
    phone: Optional[str] = None
    whatsapp: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    token: str
    user: dict

class PropertyIn(BaseModel):
    property_name: Optional[str] = None
    city_id: Optional[int] = None
    area_sqft: Optional[float] = None
    bhk: Optional[int] = None
    is_furnished: Optional[bool] = None
    listing_score: Optional[float] = None
    property_type: Optional[str] = None
    amenities: Optional[List[str]] = []
    location: Optional[str] = None
    seller_phone: Optional[str] = None
    seller_email: Optional[EmailStr] = None
    seller_whatsapp: Optional[str] = None
    images: Optional[List[str]] = []

class PredictShort(BaseModel):
    id: Optional[int] = None
    area_sqft: Optional[float] = None
    bhk: Optional[int] = None
    listing_score: Optional[float] = None
    is_furnished: Optional[bool] = None
    city_id: Optional[int] = None

class FavouriteIn(BaseModel):
    property_id: int

# Add Review models
class ReviewIn(BaseModel):
    property_id: int
    rating: int  # 1-5
    comment: str

class ReviewOut(BaseModel):
    id: int
    property_id: int
    rating: int
    comment: str
    created_at: str
    reviewer_name: str  # Anonymous: "Buyer 1", "Buyer 2"

class ChatRequest(BaseModel):
    message: str
    history: list | None = None  # Optional chat history

def property_to_dict(p: models.Property) -> dict:
    return {
        "id": p.id,
        "property_name": p.property_name,
        "city_id": p.city_id,
        "area_sqft": p.area_sqft,
        "bhk": p.bhk,
        "is_furnished": p.is_furnished,
        "listing_score": p.listing_score,
        "property_type": p.property_type,
        "amenities": p.amenities,
        "location": p.location,
        "seller_phone": p.seller_phone,
        "seller_email": p.seller_email,
        "seller_whatsapp": p.seller_whatsapp,
        "images": p.images,
        "seller_id": p.seller_id
    }

# Authentication Endpoints
@app.post("/api/register", response_model=Token)
def register_user(user: UserRegister, session: Session = Depends(get_db)):
    existing_user = session.query(models.User).filter(
        models.User.email == user.email
    ).first()
    
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    if user.user_type not in ['buyer', 'seller']:
        raise HTTPException(status_code=400, detail="User type must be 'buyer' or 'seller'")
    
    hashed_password = pwd_context.hash(user.password)
    
    db_user = models.User(
        email=user.email,
        password_hash=hashed_password,
        user_type=user.user_type,
        full_name=user.full_name,
        phone=user.phone,
        whatsapp=user.whatsapp
    )
    
    try:
        session.add(db_user)
        session.commit()
        session.refresh(db_user)
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    token = jwt.encode(
        {
            "id": db_user.id,
            "email": db_user.email,
            "user_type": db_user.user_type,
            "exp": datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
        },
        SECRET_KEY,
        algorithm=ALGORITHM
    )
    
    return {
        "token": token,
        "user": {
            "id": db_user.id,
            "email": db_user.email,
            "user_type": db_user.user_type,
            "full_name": db_user.full_name,
            "phone": db_user.phone,
            "whatsapp": db_user.whatsapp
        }
    }

@app.post("/api/login", response_model=Token)
def login_user(credentials: UserLogin, session: Session = Depends(get_db)):
    db_user = session.query(models.User).filter(
        models.User.email == credentials.email
    ).first()
    
    if not db_user:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    if not pwd_context.verify(credentials.password, db_user.password_hash):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    token = jwt.encode(
        {
            "id": db_user.id,
            "email": db_user.email,
            "user_type": db_user.user_type,
            "exp": datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
        },
        SECRET_KEY,
        algorithm=ALGORITHM
    )
    
    return {
        "token": token,
        "user": {
            "id": db_user.id,
            "email": db_user.email,
            "user_type": db_user.user_type,
            "full_name": db_user.full_name,
            "phone": db_user.phone,
            "whatsapp": db_user.whatsapp
        }
    }

def build_messages(user_message: str, history=None):
    """Prepares message format for Ollama chat with a strict system prompt."""
    
    # 🌟 NEW: Stricter System Prompt for Concise Answers
    system_content = (
        "You are a professional, concise AI assistant for a real estate platform. "
        "Your primary goal is to answer user questions **directly and briefly**. "
        "**Answer in a single sentence if possible** and **do not** include extra details, "
        "examples, or conversational filler unless specifically prompted for them. "
        "If the answer is a simple definition (e.g., 'What is real estate?'), give only the definition."
    )
    
    messages = [
        {
            "role": "system",
            "content": system_content,
        }
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages

@app.get("/")
def root():
    return {"message": "🏡 Real Estate Prediction API is running successfully!"}

# Image Upload
@app.post("/api/upload-image")
async def upload_image(file: UploadFile = File(...)):
    try:
        file_extension = file.filename.split(".")[-1]
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {"filename": unique_filename, "url": f"/uploads/properties/{unique_filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# Property Endpoints
@app.post("/api/properties")
def create_property(
    payload: PropertyIn, 
    user: dict = Depends(get_current_user_from_token),
    session: Session = Depends(get_db)
):
    # Security check: Ensure only sellers can create properties
    if user["user_type"] != "seller":
        raise HTTPException(status_code=403, detail="Only sellers can list properties")
        
    try:
        # Convert payload to dict and inject the validated seller's ID
        property_data = payload.dict()
        property_data["seller_id"] = user["id"] # <<< CRITICAL FIX

        prop = models.Property(**property_data)
        session.add(prop)
        session.commit()
        session.refresh(prop)
        return {"id": prop.id, "property_name": prop.property_name, "seller_id": prop.seller_id}
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail="DB error: " + str(e))

@app.get("/api/properties", response_model=List[dict])
def list_properties(
    limit: int = 50, 
    city_id: Optional[int] = None,
    bhk: Optional[int] = None,
    area_sqft: Optional[float] = None,
    seller_id: Optional[int] = None,
    session: Session = Depends(get_db)
):
    query = session.query(models.Property)
    
    if city_id:
        query = query.filter(models.Property.city_id == city_id)
    if bhk:
        query = query.filter(models.Property.bhk == bhk)
    if area_sqft:
        query = query.filter(models.Property.area_sqft <= area_sqft)
    if seller_id:
        query = query.filter(models.Property.seller_id == seller_id)
    
    props = query.limit(limit).all()
    return [property_to_dict(p) for p in props]

# FIXED: Favourites endpoints
@app.post("/api/favourites")
def add_favourite(
    favourite: FavouriteIn, 
    user: dict = Depends(get_current_user_from_token),
    session: Session = Depends(get_db)
):
    existing = session.query(models.Favourite).filter(
        models.Favourite.user_id == user["id"],
        models.Favourite.property_id == favourite.property_id
    ).first()
    
    if existing:
        return {"message": "Already in favourites"}
    
    fav = models.Favourite(user_id=user["id"], property_id=favourite.property_id)
    session.add(fav)
    session.commit()
    
    return {"message": "Added to favourites"}

@app.get("/api/favourites")
def get_favourites(
    user: dict = Depends(get_current_user_from_token),
    session: Session = Depends(get_db)
):
    favs = session.query(models.Favourite).filter(
        models.Favourite.user_id == user["id"]
    ).all()
    
    property_ids = [f.property_id for f in favs]
    properties = session.query(models.Property).filter(
        models.Property.id.in_(property_ids)
    ).all()
    
    return [property_to_dict(p) for p in properties]

@app.delete("/api/favourites/{property_id}")
def remove_favourite(
    property_id: int,
    user: dict = Depends(get_current_user_from_token),
    session: Session = Depends(get_db)
):
    fav = session.query(models.Favourite).filter(
        models.Favourite.user_id == user["id"],
        models.Favourite.property_id == property_id
    ).first()
    
    if fav:
        session.delete(fav)
        session.commit()
        return {"message": "Removed from favourites"}
    
    raise HTTPException(status_code=404, detail="Favourite not found")

# FIXED: Seller Analytics
@app.get("/api/seller/analytics")
def seller_analytics(
    user: dict = Depends(get_current_user_from_token),
    session: Session = Depends(get_db)
):
    if user["user_type"] != "seller":
        raise HTTPException(status_code=403, detail="Seller only")
    
    # Get seller's properties
    properties = session.query(models.Property).filter(
        models.Property.seller_id == user["id"]
    ).all()
    
    property_ids = [p.id for p in properties]
    
    if not property_ids:
        return {
            "total_properties": 0,
            "total_unique_buyers": 0,
            "buyers": [],
            "property_favourites": {},
            "reviews": []
        }
    
    # Get favourites for seller's properties
    favourites = session.query(models.Favourite).filter(
        models.Favourite.property_id.in_(property_ids)
    ).all()
    
    # Get unique buyers
    buyer_ids = list(set([f.user_id for f in favourites]))
    buyers = session.query(models.User).filter(
        models.User.id.in_(buyer_ids)
    ).all()
    
    # Per-property favourite counts
    property_favs = {}
    for p in properties:
        count = len([f for f in favourites if f.property_id == p.id])
        property_favs[p.id] = count
    
    # Get reviews (anonymous)
    reviews = session.query(models.Review).filter(
        models.Review.property_id.in_(property_ids)
    ).all()
    
    review_list = []
    reviewer_map = {}  # Map user_id to anonymous name
    counter = 1
    
    for r in reviews:
        if r.user_id not in reviewer_map:
            reviewer_map[r.user_id] = f"Buyer {counter}"
            counter += 1
        
        review_list.append({
            "id": r.id,
            "property_id": r.property_id,
            "rating": r.rating,
            "comment": r.comment,
            "reviewer_name": reviewer_map[r.user_id],
            "created_at": r.created_at.isoformat() if r.created_at else None
        })
    
    return {
        "total_properties": len(properties),
        "total_unique_buyers": len(buyer_ids),
        "buyers": [{"id": b.id, "name": b.full_name, "email": b.email, "phone": b.phone} for b in buyers],
        "property_favourites": property_favs,
        "reviews": review_list
    }

# NEW: Review endpoints
@app.post("/api/reviews")
def create_review(
    review: ReviewIn,
    user: dict = Depends(get_current_user_from_token),
    session: Session = Depends(get_db)
):
    if user["user_type"] != "buyer":
        raise HTTPException(status_code=403, detail="Only buyers can review")
    
    # Check if property exists
    prop = session.query(models.Property).filter(
        models.Property.id == review.property_id
    ).first()
    
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")
    
    # Check if already reviewed
    existing = session.query(models.Review).filter(
        models.Review.user_id == user["id"],
        models.Review.property_id == review.property_id
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="You have already reviewed this property")
    
    new_review = models.Review(
        user_id=user["id"],
        property_id=review.property_id,
        rating=review.rating,
        comment=review.comment
    )
    
    session.add(new_review)
    session.commit()
    
    return {"message": "Review submitted successfully"}

@app.get("/api/reviews/{property_id}")
def get_property_reviews(
    property_id: int,
    session: Session = Depends(get_db)
):
    reviews = session.query(models.Review).filter(
        models.Review.property_id == property_id
    ).all()
    
    # Anonymize reviewers
    reviewer_map = {}
    counter = 1
    result = []
    
    for r in reviews:
        if r.user_id not in reviewer_map:
            reviewer_map[r.user_id] = f"Buyer {counter}"
            counter += 1
        
        result.append({
            "id": r.id,
            "rating": r.rating,
            "comment": r.comment,
            "reviewer_name": reviewer_map[r.user_id],
            "created_at": r.created_at.isoformat() if r.created_at else None
        })
    
    return result

# FIXED: Chatbot with fallback
@app.post("/chatbot")
async def chatbot(request: ChatRequest):
    """Chatbot with clean response text only."""
    try:
        messages = build_messages(request.message, request.history)
        response = ollama.chat(model="tinyllama", messages=messages, stream=False)

        answer = ""

        # Extract the assistant message text properly
        if isinstance(response, dict):
            # Standard Ollama response (dictionary)
            if "message" in response and isinstance(response["message"], dict):
                answer = response["message"].get("content", "")
            elif "response" in response:
                answer = response["response"]
        
        # Fallback for non-dictionary/object response (where metadata appears)
        if not answer:
            response_str = str(response)
            
            # Use regex to find the 'content="..."' part and extract it
            # This targets the actual message content, ignoring the metadata
            match = re.search(r"content=[\"\'](.*?)(?<!\\)[\"\']", response_str, re.DOTALL)
            
            if match:
                # Group 1 is the content. Replace escaped newlines if present.
                answer = match.group(1).replace("\\n", "\n")
            else:
                # If regex fails, use the whole string as a last resort
                answer = response_str


        # Clean up: remove any unwanted newlines or model metadata fragments
        # The regex should handle most of this, but strip() is still good practice.
        answer = answer.strip()
        
        # Final safety filter to remove any residual metadata fragments
        if answer.startswith("Message(role='assistant', content="):
             # Try to clean if the regex missed it
             answer = answer.split("content=")[-1].strip("')")


        return JSONResponse({"response": answer})

    except Exception as e:
        print("Chatbot error, using fallback:", e)
        # Fallback to FAQ responses
        answer = get_fallback_response(request.message)
        return JSONResponse({"response": answer})


def get_fallback_response(message: str):
    """Intelligent FAQ-based responses when chatbot is unavailable"""
    msg = message.lower()
    
    faqs = {
        'rera': 'RERA (Real Estate Regulatory Authority) registration ensures your property is legally compliant. Always verify the RERA number before purchasing. Registered properties offer better buyer protection.',
        'loan': 'Home loans are available from banks with interest rates of 8-9% p.a. Use our EMI Calculator to plan your payments. Consider factors like down payment (usually 20%), tenure (up to 30 years), and processing fees.',
        'document': 'Essential documents: Title Deed, Sale Agreement, Encumbrance Certificate, Property Tax Receipts, Completion Certificate, Occupancy Certificate, and NOC from society/builder.',
        'price': 'Our AI predicts prices using location, size, amenities, and market trends. Predictions are estimates based on similar properties. Actual prices depend on negotiation and current market conditions.',
        'sell': 'As a seller: Upload quality images, provide accurate details, add amenities, keep contact info updated. Track interested buyers in your analytics dashboard.',
        'buy': 'As a buyer: Filter properties by city/BHK/area, save favourites, check seller ratings, use EMI calculator, and contact sellers directly via call/email/WhatsApp.',
        'invest': 'Investment tips: Research location growth, check infrastructure plans, calculate rental yield, consider appreciation potential, diversify portfolio, and verify legal documents.',
        'tax': 'Property tax varies by city (0.5-2% of property value annually). Capital gains tax applies on sale (20% with indexation for long-term). Consult a CA for specifics.',
        'favourite': 'Click the heart icon to save properties. Sellers can see who favourited their listings and connect with interested buyers.',
        'review': 'Buyers can review properties after viewing. Reviews help other buyers make informed decisions. Your identity remains anonymous to sellers.',
    }
    
    for keyword, response in faqs.items():
        if keyword in msg:
            return response
    
    # Default response
    return ("I'm your EstateAI assistant! Ask me about:\n"
            "• RERA registration\n"
            "• Home loans & EMI\n"
            "• Property documents\n"
            "• Price predictions\n"
            "• Buying/Selling tips\n"
            "• Investment advice\n"
            "• Property tax\n\n"
            "You can also use our EMI Calculator and save favourites!")


@app.post("/chatbot/stream")
async def chatbot_stream(request: ChatRequest):
    """
    Streams responses from Ollama (tinyllama) in real-time (SSE).
    Use this if your frontend supports streaming chat.
    """
    try:
        messages = build_messages(request.message, request.history)

        async def stream_gen():
            try:
                for chunk in ollama.chat(model="tinyllama", messages=messages, stream=True):
                    text = ""
                    if isinstance(chunk, dict):
                        text = (
                            chunk.get("message", {}).get("content", "")
                            or chunk.get("response", "")
                            or chunk.get("delta", "")
                        )
                    else:
                        text = str(chunk)

                    if text:
                        yield f"data: {json.dumps({'delta': text})}\n\n".encode("utf-8")
                    await asyncio.sleep(0)
                yield f"data: {json.dumps({'done': True})}\n\n".encode("utf-8")
            except Exception as err:
                print("Streaming error:", err)
                yield f"data: {json.dumps({'error': 'stream failed'})}\n\n".encode("utf-8")

        return StreamingResponse(stream_gen(), media_type="text/event-stream")

    except Exception as e:
        print("Chatbot stream setup error:", e)
        raise HTTPException(status_code=500, detail="Streaming failed")


# Predictions
@app.post("/api/predict_bulk")
def predict_bulk(items: List[PredictShort] = Body(...)):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not loaded")

    rows = []
    ids = []
    
    for it in items:
        ids.append(it.id)
        row = {f: 0 for f in FEATURES_42}
        row["area_num"] = float(it.area_sqft or 0)
        row["bhk"] = int(it.bhk or 0)
        row["listing_domain_score"] = float(it.listing_score or 5)
        row["is_furnished"] = int(bool(it.is_furnished))
        row["area_per_bhk"] = (float(it.area_sqft) / it.bhk) if it.area_sqft and it.bhk else 0.0

        city_map = {
            1: "city_Bangalore", 2: "city_Chennai", 3: "city_Delhi", 4: "city_Hyderabad",
            5: "city_Kolkata", 6: "city_Lucknow", 7: "city_Mumbai"
        }
        
        if it.city_id in city_map:
            row[city_map[it.city_id]] = 1

        rows.append(row)

    X = pd.DataFrame(rows, columns=FEATURES_42)

    try:
        preds = pipeline.predict(X)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

    out = []
    for i, p in enumerate(preds):
        out.append({"id": ids[i], "prediction": float(p * 5)})

    return out