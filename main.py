from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
import os
import json
import re

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
# Request models
# ─────────────────────────────────────────────
class ProductCatalogItem(BaseModel):
    id: str
    name: str
    price: float
    originalPrice: float | None = None
    href: str | None = None
    images: list[str] | None = None
    img1: str | None = None
    img2: str | None = None
    description: str | None = None
    sizes: str | None = None
    stock: int | None = None
    badge: str | None = None


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
    catalog: list[ProductCatalogItem] | None = None
    current_product_id: str | None = None


# ─────────────────────────────────────────────
# Helper: Parse agent response
# ─────────────────────────────────────────────
def parse_agent_response(response_text: str) -> dict:
    # Try to find a JSON block in the text
    json_match = re.search(r"({.*})", response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except Exception:
            pass
    try:
        return json.loads(response_text)
    except Exception:
        # Fallback if LLM output is not valid JSON
        # Clean up text for description
        clean_text = response_text.replace('"', '\\"').replace('\n', ' ')
        return {
            "thoughts": "Agent collaboration completed.",
            "explanation": clean_text,
            "matched_products": []
        }


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
        # Build catalog details block
        catalog_details = ""
        if product.catalog:
            for item in product.catalog:
                catalog_details += f"- ID: {item.id} | Name: {item.name} | Price: ₹{item.price} | Stock: {item.stock} | Sizes: {item.sizes or 'S, M, L, XL'} | Description: {item.description or ''}\n"
        else:
            catalog_details = "No other products in catalog."

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

        prompt = f"""You are the orchestrator of a Multi-Agent AI shopping assistant for Tejasvin, a premium Indian cultural streetwear brand.

The system consists of three specialized agents working together:
1. Classifier Agent: Analyzes the customer's question and current context.
2. Product Retrieval Agent: Scans the product catalog (supplied below) to find exact matches or recommendations.
3. Advisor Agent: Formulates a friendly, conversational response incorporating matching products.

Here is the current product being viewed by the customer:
{product_info}

Here is the full catalog of products available in the store:
{catalog_details}

Customer's Question: {product.question.strip() if (product.question and product.question.strip()) else "Give me a welcome introduction to this product!"}

You MUST execute the collaboration between these agents and respond strictly in JSON format.
In your JSON, you must include:
- thoughts: A detailed breakdown of the multiagent collaboration (e.g. "Agent 1 (Classifier): User is asking for other tees. Agent 2 (Searcher): Scanned catalog, found Bheeman and Krishna tees. Agent 3 (Advisor): Formulating recommendation...").
- explanation: A friendly, enthusiastic 2-3 sentence answer/recommendation to the user.
- matched_products: A list of string product IDs (must match the exact 'id' from the catalog) that are mentioned in the response or are relevant. If no products are relevant, return an empty list [].

Strict JSON Output format:
{{
  \"thoughts\": \"string detailing the multiagent workflow logs\",
  \"explanation\": \"string response\",
  \"matched_products\": [\"id1\", \"id2\"]
}}

Output ONLY valid JSON. Do not include any markdown wrappers or introductory text.
"""

        result_text = call_ai(prompt)
        parsed = parse_agent_response(result_text)

        return {
            "success": True,
            "thoughts": parsed.get("thoughts", "Agent collaboration completed."),
            "explanation": parsed.get("explanation", result_text),
            "matched_products": parsed.get("matched_products", []),
            "product": product.name
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI error: {str(e)}"
        )

