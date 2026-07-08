from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
import os
import json
import re
import base64

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
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ✅ Google AI client
api_key = os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None


# ─────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────
class ProductData(BaseModel):
    id: str | None = None
    name: str
    price: str
    description: str
    category: str | None = None
    fabric: str | None = None
    sizes: list[str] | None = None
    color: str | None = None
    stock: str | None = None
    question: str | None = None   # ← Customer's actual question (if any)

class CatalogProduct(BaseModel):
    id: str
    name: str
    price: float
    description: str
    category: str | None = None
    sizes: list[str] | None = None
    stock: str | None = None

class ProductExplainRequest(BaseModel):
    # Old schema fields (made optional for backward compatibility)
    name: str | None = None
    price: str | None = None
    description: str | None = None
    category: str | None = None
    fabric: str | None = None
    sizes: list[str] | None = None
    color: str | None = None
    stock: str | None = None
    question: str | None = None
    
    # New schema fields
    current_product: ProductData | None = None
    catalog: list[CatalogProduct] | None = None
    username: str | None = None
    ai_name: str | None = None
    ai_style: str | None = None
    ai_bio: str | None = None
    image_base64: str | None = None

# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "✅ Tejasvin AI Backend is running!", "model": "Gemma 4 Multi-Agent", "online": client is not None}


# ─────────────────────────────────────────────
# Helper: call AI with model fallback chain
# ─────────────────────────────────────────────
def call_ai(prompt: str, req_catalog: list = [], current_product_id: str | None = None, image_b64: str | None = None) -> str:
    if not client:
        # Determine recommended IDs for dry-run offline testing
        rec_ids = [current_product_id] if current_product_id else []
        if not rec_ids and req_catalog:
            rec_ids = [req_catalog[0].id]
        return json.dumps({
            "search_agent_thoughts": f"🔍 Search Agent: [DRY-RUN] Scanned catalog of {len(req_catalog)} products. Recommended ID: {rec_ids[0] if rec_ids else 'None'}",
            "stylist_agent_thoughts": f"🎨 Stylist Agent: [DRY-RUN] Style review: perfect 240 GSM bio-washed cotton, ancient Indian streetwear theme.",
            "coordinator_response": "✦ Tejasvin: Namaste! The local AI is running in offline dry-run mode (no GOOGLE_API_KEY set). Set the environment variable to query the live Gemini/Gemma models.",
            "recommended_product_ids": rec_ids
        })

    model_names = [
        "gemma-4-27b-it",    # Gemma 4 27B
        "gemma-4-26b-it",    # Gemma 4 26B MoE
        "gemma-4-31b-it",    # Gemma 4 31B Dense
        "gemma-3-27b-it",    # Gemma 3 — always works
        "gemini-2.0-flash",  # Gemini safety net
    ]
    last_error = None
    
    contents = [prompt]
    if image_b64:
        # Expected format: data:image/jpeg;base64,...
        if "," in image_b64:
            header, b64_data = image_b64.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0]
        else:
            b64_data = image_b64
            mime_type = "image/jpeg"
        
        try:
            image_bytes = base64.b64decode(b64_data)
            contents.append({
                "mime_type": mime_type,
                "data": image_bytes
            })
        except Exception as e:
            print(f"Failed to decode image: {e}")

    for model_name in model_names:
        try:
            response = client.models.generate_content(model=model_name, contents=contents)
            if response.text:
                return response.text.strip()
        except Exception as e:
            last_error = e
            continue
    raise last_error if last_error else RuntimeError("AI model failure")


# ─────────────────────────────────────────────
# Helper: parse JSON response from LLM
# ─────────────────────────────────────────────
def parse_agent_response(response_text: str, default_product_id: str | None = None) -> dict:
    # Attempt to extract JSON from markdown code block if present
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Try finding the first '{' and last '}'
        start = response_text.find('{')
        end = response_text.rfind('}')
        if start != -1 and end != -1:
            json_str = response_text[start:end+1]
        else:
            json_str = response_text

    try:
        data = json.loads(json_str)
        # Ensure correct keys
        res = {
            "search_agent_thoughts": data.get("search_agent_thoughts", "🔍 Search Agent: Catalog analysis completed."),
            "stylist_agent_thoughts": data.get("stylist_agent_thoughts", "🎨 Stylist Agent: Styling evaluation completed."),
            "coordinator_response": data.get("coordinator_response", ""),
            "recommended_product_ids": data.get("recommended_product_ids", [])
        }
        if not res["coordinator_response"]:
            # If coordinator response is missing, use search/stylist thoughts as explanation
            res["coordinator_response"] = response_text
        return res
    except Exception:
        pass

    # Fallback to text explanation if JSON parsing fails entirely
    return {
        "search_agent_thoughts": "🔍 Search Agent: Scanned the catalog based on your interests.",
        "stylist_agent_thoughts": "🎨 Stylist Agent: Selected the best premium fabric and fit matching your vibe.",
        "coordinator_response": response_text,
        "recommended_product_ids": [default_product_id] if default_product_id else []
    }


# ─────────────────────────────────────────────
# Main endpoint
# ─────────────────────────────────────────────
@app.post("/explain-product")
async def explain_product(req: ProductExplainRequest):
    try:
        # Resolve backward compatibility
        req_current = req.current_product
        req_question = req.question
        req_username = req.username
        req_catalog = req.catalog or []

        if not req_current:
            # Fallback to root fields (old format)
            req_current = ProductData(
                id=None,
                name=req.name or "TEJASVIN Product",
                price=req.price or "0",
                description=req.description or "Premium Indian cultural streetwear.",
                category=req.category,
                fabric=req.fabric,
                sizes=req.sizes,
                color=req.color,
                stock=req.stock,
                question=req.question
            )
            req_question = req.question or (req.current_product.question if req.current_product else req.question)

        # Build catalog text info for LLM prompt context
        catalog_text = ""
        for p in req_catalog:
            catalog_text += f"- ID: {p.id} | Name: {p.name} | Price: ₹{p.price} | Category: {p.category or 'Streetwear'} | Description: {p.description[:180]}\n"

        # Current product details block
        current_product_id = req_current.id
        current_product_info = f"Product Name: {req_current.name}\n"
        current_product_info += f"Price: ₹{req_current.price}\n"
        current_product_info += f"Description: {req_current.description}\n"
        if req_current.fabric:
            current_product_info += f"Fabric: {req_current.fabric}\n"
        if req_current.sizes:
            current_product_info += f"Available Sizes: {', '.join(req_current.sizes)}\n"
        if req_current.stock:
            current_product_info += f"Stock Status: {req_current.stock}\n"
        if req_current.category:
            current_product_info += f"Category: {req_current.category}\n"

        # Personalization profile extraction
        personalization_context = ""
        ai_persona_name = req.ai_name or "Tejasvin"
        
        if req.ai_style or req.ai_bio:
            personalization_context = "\nPERSONALIZATION PROFILE:\n"
            if req.ai_style:
                personalization_context += f"- Customer Style Preference: {req.ai_style}\n"
            if req.ai_bio:
                personalization_context += f"- Customer Bio & Preferences: {req.ai_bio}\n"
            personalization_context += f"You MUST align all styling thoughts and coordinator recommendations specifically to match this customer's style profile and preferences.\n"

        # ── MODE 1: Customer asked a specific question ──────────────────────────
        if req_question and req_question.strip():
            prompt = f"""You are a multi-agent AI system for Tejasvin, a premium Indian cultural streetwear brand blending Ancient Bharat's legacy with modern street style.
The team consists of:
1. SearchAgent: Scans the product catalog to find product(s) that match the customer's request.
2. StylistAgent: Evaluates styling, fit, and cultural design, providing a personalized style recommendation.
3. CoordinatorAgent: Synthesizes the response to directly answer the customer's question.

Your CoordinatorAgent identity name is "{ai_persona_name}". You MUST speak under this name.

Customer Name: {req_username or 'Customer'}
Customer's Query: "{req_question.strip() if req_question else 'Analyze this image.'}"
{personalization_context}
Current Product details:
{current_product_info}

Available Catalog:
{catalog_text}

Your task is to collaborate as these agents to answer the user's question.
You MUST respond with a JSON object in this exact format (do not output any other text or markdown wrapping except valid JSON):
{{
  "search_agent_thoughts": "🔍 Search Agent: [Describe what catalog search matches you found, or if you recommended the current product]",
  "stylist_agent_thoughts": "🎨 Stylist Agent: [Provide styling advice tailored to {req_username or 'the customer'}'s request, detailing fit, fabric, and cultural street vibe]",
  "coordinator_response": "✦ {ai_persona_name}: [Your final warm, friendly 2-3 sentence answer directly to the customer]",
  "recommended_product_ids": [A list of product IDs (strings) from the catalog or current product that match this recommendation. For example, if the current product matches, include its ID. If a catalog item matches, include its ID. Do not return more than 3 IDs.]
}}

Ensure all JSON keys and values are correctly escaped. Remember to write from the perspective of a high-end cultural brand.
"""

        # ── MODE 2: First open — give an exciting intro explanation ────────────
        else:
            prompt = f"""You are a multi-agent AI system for Tejasvin, a premium Indian cultural streetwear brand blending Ancient Bharat's legacy with modern street style.
The team consists of:
1. SearchAgent: Scans the product catalog.
2. StylistAgent: Evaluates styling, fit, and cultural design.
3. CoordinatorAgent: Synthesizes a warm introduction for the customer.

Your CoordinatorAgent identity name is "{ai_persona_name}". You MUST speak under this name.

Customer Name: {req_username or 'Customer'}
The customer just opened the product page for "{req_current.name}". Give them a warm, exciting introduction.
{personalization_context}
Current Product details:
{current_product_info}

Available Catalog:
{catalog_text}

Collaborate as these agents to introduce this product.
You MUST respond with a JSON object in this exact format (do not output any other text or markdown wrapping except valid JSON):
{{
  "search_agent_thoughts": "🔍 Search Agent: [Explain why this specific product is in focus and matches the catalog style]",
  "stylist_agent_thoughts": "🎨 Stylist Agent: [Highlight the premium feel, fabric, fit, and cultural style story of this product for {req_username or 'the customer'}]",
  "coordinator_response": "✦ {ai_persona_name}: [A welcoming, exciting 2-3 sentence intro to this product, ending with a friendly invitation to ask anything]",
  "recommended_product_ids": [A list containing the current product's ID if available. e.g., ["{current_product_id or ''}"]]
}}

Ensure all JSON keys and values are correctly escaped. Remember to write from the perspective of a high-end cultural brand.
"""

        result = call_ai(prompt, req_catalog, current_product_id, req.image_base64)
        parsed = parse_agent_response(result, current_product_id)

        return {
            "success": True,
            "search_agent_thoughts": parsed["search_agent_thoughts"],
            "stylist_agent_thoughts": parsed["stylist_agent_thoughts"],
            "explanation": parsed["coordinator_response"],
            "recommended_product_ids": parsed["recommended_product_ids"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI error: {str(e)}"
        )
