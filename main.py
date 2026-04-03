from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
import os

app = FastAPI(title="Tejasvin AI Backend", version="2.0.0")

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://tejasvin.in",
        "https://www.tejasvin.in",
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ✅ Google AI client
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))


# ─────────────────────────────────────────────
# Request model
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
    question: str | None = None   # ← Customer's actual question (if any)


# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "✅ Tejasvin AI Backend is running!", "model": "Gemma 4"}


# ─────────────────────────────────────────────
# Helper: call AI with model fallback chain
# ─────────────────────────────────────────────
def call_ai(prompt: str) -> str:
    model_names = [
        "gemma-4-27b-it",    # Gemma 4 27B
        "gemma-4-26b-it",    # Gemma 4 26B MoE
        "gemma-4-31b-it",    # Gemma 4 31B Dense
        "gemma-3-27b-it",    # Gemma 3 — always works
        "gemini-2.0-flash",  # Gemini safety net
    ]
    last_error = None
    for model_name in model_names:
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            return response.text.strip()
        except Exception as e:
            last_error = e
            continue
    raise last_error


# ─────────────────────────────────────────────
# Main endpoint
# ─────────────────────────────────────────────
@app.post("/explain-product")
async def explain_product(product: ProductData):
    try:
        # Build product context block
        product_info = f"Product Name: {product.name}\n"
        product_info += f"Price: ₹{product.price}\n"
        product_info += f"Description: {product.description}\n"
        if product.fabric:
            product_info += f"Fabric: {product.fabric}\n"
        if product.sizes:
            product_info += f"Available Sizes: {', '.join(product.sizes)}\n"
        if product.stock:
            product_info += f"Stock Status: {product.stock}\n"
        if product.category:
            product_info += f"Category: {product.category}\n"

        # ── MODE 1: Customer asked a specific question ──────────────────────────
        if product.question and product.question.strip():
            prompt = f"""You are a knowledgeable and friendly shopping assistant for Tejasvin — a premium Indian cultural streetwear brand.

A customer is asking a specific question about a product. Your job is to DIRECTLY ANSWER their question using the product details below.

CRITICAL RULES:
- Answer ONLY what they asked. Do NOT re-describe the whole product.
- Be specific and helpful. If they ask about sizes, tell them the sizes. If they ask about fit, explain the fit.
- Keep it to 2-3 sentences maximum.
- Sound like a helpful friend texting them, not a sales robot.
- Do NOT start with "You're going to love..." or any sales pitch opener.

Product Details:
{product_info}

Customer's Question: {product.question.strip()}

Answer their question directly:"""

        # ── MODE 2: First open — give an exciting intro explanation ────────────
        else:
            prompt = f"""You are a friendly and enthusiastic fashion assistant for Tejasvin — a cultural Indian streetwear brand blending Ancient Bharat's legacy with modern street style.

A customer just opened the product page. Give them a warm, exciting 2-3 sentence introduction to this product. Make them feel interested and excited.

Rules:
- Keep it to 2-3 sentences only
- Mention price and 1-2 standout features
- Sound like a knowledgeable friend, not a salesperson
- End with a soft invitation like "Ask me anything!" or "What would you like to know?"
- Do NOT use bullet points

Product Details:
{product_info}

Give the intro:"""

        result = call_ai(prompt)

        return {
            "success": True,
            "explanation": result,
            "product": product.name
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI error: {str(e)}"
        )
