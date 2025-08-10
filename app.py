from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import List
from fastapi.middleware.cors import CORSMiddleware

# ================================
# Load model and vectorizer
# ================================
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Load products dataset (cleaned Aggarwal CSV)
df = pd.read_csv("products.csv")  # columns: id, title, color, category, price, image_url, product_url

# ================================
# FastAPI setup
# ================================
app = FastAPI(title="StyleMate API", description="Backend for AI Fashion Stylist", version="1.0")

# CORS middleware: allow frontend to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace "*" with your frontend URL(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# Request/Response Models
# ================================
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    intent: str
    match_score: float

class Product(BaseModel):
    title: str
    price: float
    image_url: str
    product_url: str

class RecommendRequest(BaseModel):
    color: str
    category: str

class RecommendResponse(BaseModel):
    items: List[Product]

# ================================
# Endpoints
# ================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Directly send raw message twice separated by [SEP] to model
        inp = f"{request.message} [SEP] {request.message}"
        vec = vectorizer.transform([inp])
        score = model.predict_proba(vec)[0][1]

        if score > 0.5:
            reply = "Yes, that combination works well! Shall I show you similar items?"
            intent = "ask_recommend"
        else:
            reply = "I wouldn't suggest pairing those together. Do you want to see alternative matches?"
            intent = "ask_alternative"

        return ChatResponse(reply=reply, intent=intent, match_score=score)

    except Exception as e:
        return ChatResponse(reply="Sorry, something went wrong.", intent="error", match_score=0.0)


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    # Filter products based on color & category
    filtered = df[
        (df['color'].str.lower() == request.color.lower()) &
        (df['category'].str.lower() == request.category.lower())
    ]
    # Pick up to 3 random items
    items = filtered.sample(min(3, len(filtered))).to_dict('records')

    products = [
        Product(
            title=item['title'],
            price=item['price'],
            image_url=item['image_url'],
            product_url=item['product_url']
        ) for item in items
    ]

    return RecommendResponse(items=products)
