from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
import os

app = FastAPI(title="Tejasvin AI Backend", version="1.0.0")

# ✅ CORS - Allow tejasvin.in to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://tejasvin.in",
        "https://www.tejasvin.in",
        "http://localhost",         # for local testing
        "http://127.0.0.1",
        "http://localhost:5500",    # VS Code Live Server
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ✅ Google AI client (uses GOOGLE_API_KEY env variable)
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

# ─────────────────────────────────────────────
# Request model - matches your Firestore fields
# Add/remove fields based on your product schema
# ─────────────────────────────────────────────
class ProductData(BaseModel):
    name: str
    price: str
    description: str
    category: str | None = None
    fabric: str | None = None
    sizes: list[str] | None = None
    color: str | None = None
    stock: str | None = None


# ─────────────────────────────────────────────
# Health check endpoint
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "✅ Tejasvin AI Backend is running!", "model": "Gemma 4"}


# ─────────────────────────────────────────────
# Main endpoint - explain a product using Gemma 4
# ─────────────────────────────────────────────
@app.post("/explain-product")
async def explain_product(product: ProductData):
    try:
        # Build product info string from Firestore data
        product_info = f"Product Name: {product.name}\n"
        product_info += f"Price: ₹{product.price}\n"
        product_info += f"Description: {product.description}\n"

        if product.category:
            product_info += f"Category: {product.category}\n"
        if product.fabric:
            product_info += f"Fabric: {product.fabric}\n"
        if product.color:
            product_info += f"Color: {product.color}\n"
        if product.sizes:
            product_info += f"Available Sizes: {', '.join(product.sizes)}\n"
        if product.stock:
            product_info += f"Stock: {product.stock}\n"

        # Gemma 4 prompt
        prompt = f"""You are a friendly and enthusiastic fashion assistant for Tejasvin — a cultural Indian streetwear brand that blends traditional Indian culture with modern street style.

A customer is viewing a product and wants to understand it better. Read the product details below and explain it to them in a warm, exciting, and conversational way. 

Rules:
- Keep it to 3-4 sentences maximum
- Highlight what makes the product unique or special
- Mention the price naturally
- Sound like a helpful friend, not a robot
- Do NOT use bullet points, just natural flowing text

Product Details:
{product_info}

Give a natural explanation that would make someone excited to buy it."""

        # Call Gemma 4 via Google AI Studio API
        # ⚠️ NOTE: Gemma 4 was just released on April 3 2026.
        # Check https://aistudio.google.com/ for the exact model ID.
        # Likely: "gemma-4-27b-it" or "gemma-4-31b-it"
        # If it fails, fallback to "gemma-3-27b-it" temporarily.
        response = client.models.generate_content(
            model="gemma-4-27b-it",   # ← Update this if needed after checking AI Studio
            contents=prompt,
        )

        return {
            "success": True,
            "explanation": response.text.strip(),
            "product": product.name
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gemma 4 error: {str(e)}"
        )
