import time
import logging
import os
import urllib.request
import urllib.parse
import json
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from anthropic import AsyncAnthropic
from duckduckgo_search import DDGS
from mangum import Mangum
 
# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("inveniq")
 
# ── Anthropic client (async, with timeout) ────────────────────────────────────
# Z.AI exposes an Anthropic-compatible endpoint. Set ZAI_API_KEY in Vercel
# Dashboard → Settings → Environment Variables, then redeploy.
client = AsyncAnthropic(
    base_url="https://api.z.ai/api/anthropic",
    api_key=os.environ.get("ZAI_API_KEY", ""),
    timeout=9.0,  # Vercel Hobby kills at 10s; leave 1s for response serialization
)
 
app = FastAPI(title="InvenIQ Backend (ILMU)", version="4.2.0-vercel")
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# ── Request models ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    messages: list
    model: str = "glm-5-turbo"
    temperature: float = 0.1
    max_tokens: int = 1500
 
class SignalSearchRequest(BaseModel):
    category: str   # "weather", "calendar", "news", "raw"
    location: str = "Malaysia"
    context: str = ""  # CSV-derived keywords e.g. "beverages, snacks, sugar"
 
# ── Web search tool definition (Anthropic schema) ─────────────────────────────
WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": (
        "Search the web for current news, events, calendar dates, market trends, "
        "weather, or any unstructured data the user asks about. Use this whenever "
        "the user's question requires up-to-date or real-world information."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up on the web",
            }
        },
        "required": ["query"],
    },
}
 
 
# ── DuckDuckGo helper ─────────────────────────────────────────────────────────
# Single attempt + tight DDGS socket timeout. On serverless we don't get a
# second chance — the whole function has 10s.
def _ddgs_search(
    query: str,
    max_results: int = 5,
    max_retries: int = 1,
) -> tuple[list, str | None]:
    """
    DuckDuckGo text search with retry + rate-limit handling.
    Returns (results, error). On success, error is None.
    """
    last_err: str | None = None
    for attempt in range(max_retries):
        try:
            with DDGS(timeout=3) as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if results:
                return results, None
            last_err = "empty"
            if attempt < max_retries - 1:
                time.sleep(1.0)
        except Exception as e:
            err_msg = str(e).lower()
            is_rate_limit = any(
                kw in err_msg for kw in ["429", "rate", "limit", "too many", "throttl", "captcha"]
            )
            last_err = "rate_limit" if is_rate_limit else f"error: {e}"
            if attempt < max_retries - 1:
                wait = 1.5 if is_rate_limit else 1.0
                logger.warning("DDGS %s — waiting %ss before retry %d", last_err, wait, attempt + 1)
                time.sleep(wait)
    return [], last_err
 
 
def perform_web_search(query: str) -> str:
    """Web search used by the AI's tool-use loop. Fails fast on Vercel."""
    results, err = _ddgs_search(query, max_results=5, max_retries=1)
    if results:
        return "\n\n".join(
            f"- **{r['title']}**\n  {r['body']}\n  Source: {r['href']}"
            for r in results
        )
    if err == "rate_limit":
        return ("Web search is rate-limited. Answer using existing knowledge and "
                "clearly note that live data was unavailable.")
    if err and err != "empty":
        return (f"Web search failed ({err}). Answer using existing knowledge and "
                "note that live data was unavailable.")
    return ("No web results found. Answer using existing knowledge and note that "
            "live data was unavailable.")
 
 
# ── System prompt ─────────────────────────────────────────────────────────────
MASTER_PROMPT = """You are InvenIQ, an elite, professional Inventory Intelligence AI.
Your primary job is to help the user manage their stock, analyze sales data, and predict inventory shortages.
 
RULES:
1. Be highly analytical, precise, and professional.
2. Format your answers clearly using bullet points or short paragraphs.
3. You do NOT have live web access in this chat. Answer from your existing knowledge.
4. If the user asks about current news, weather, calendar dates, or market trends, tell them to use the dedicated "Signal Fetch" buttons in the side panel (Calendar / Weather / News / Raw) which fetch live data, then paste the results into the chat for you to analyze.
5. If you need more data (like a CSV file or numbers) to answer a question, ask the user to provide it.
"""
 
# ── Signal query templates ────────────────────────────────────────────────────
def _signal_queries(year: int) -> dict[str, list[str]]:
    return {
        "weather": [],  # handled by wttr.in
        "calendar": [
            f"upcoming Malaysia public holidays {year}",
            f"Malaysia school holidays calendar {year}{{ctx}}",
            f"Malaysia events and festivals {year}{{ctx}}",
        ],
        "news": [
            f"Malaysia{{ctx}}retail market news {year}",
            "Malaysia{ctx}supply chain business news",
            f"Malaysia consumer market updates {year}",
        ],
        "raw": [
            f"Malaysia{{ctx}}consumer trends economic outlook {year}",
            f"Malaysia GDP inflation food prices {year}",
            f"Malaysia{{ctx}}retail industry forecast {year}",
        ],
    }
 
 
# ── Weather (wttr.in) ─────────────────────────────────────────────────────────
def fetch_weather_wttr(location: str = "Malaysia") -> str:
    """Free, no-API-key weather. Kept under 8s total on Vercel."""
    # Limited to 2 cities to stay under Vercel Hobby's 10s function timeout
    # (each city has a 2s budget, plus JSON parsing overhead).
    cities = ["Kuala Lumpur", "Penang"]
    parts = []
    for city in cities:
        try:
            url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
            req = urllib.request.Request(url, headers={"User-Agent": "curl/7.68.0"})
            with urllib.request.urlopen(req, timeout=2) as resp:  # tight per-city budget
                data = json.loads(resp.read().decode())
 
            current = data.get("current_condition", [{}])[0]
            weather_desc = current.get("weatherDesc", [{}])[0].get("value", "N/A")
            temp_c = current.get("temp_C", "N/A")
            humidity = current.get("humidity", "N/A")
            feels = current.get("FeelsLikeC", "N/A")
 
            forecast_lines = []
            for day in data.get("weather", []):
                date = day.get("date", "")
                max_t = day.get("maxtempC", "")
                min_t = day.get("mintempC", "")
                hourly = day.get("hourly", [])
                desc = (
                    hourly[4].get("weatherDesc", [{}])[0].get("value", "")
                    if len(hourly) > 4
                    else "N/A"
                )
                forecast_lines.append(f"  {date}: {min_t}°C–{max_t}°C, {desc}")
            forecast_str = "\n".join(forecast_lines[:5])
 
            parts.append(
                f"📍 {city}\n"
                f"  Now: {temp_c}°C (feels {feels}°C), {weather_desc}, Humidity {humidity}%\n"
                f"  Forecast:\n{forecast_str}"
            )
        except Exception as e:
            logger.warning("Weather fetch failed for %s: %s", city, e)
            parts.append(f"📍 {city}: Weather data unavailable")
    return "\n\n".join(parts)
 
 
# ── In-memory cache (helps when serverless container stays warm) ──────────────
_SIGNAL_CACHE: dict[str, tuple[float, dict]] = {}
_SIGNAL_CACHE_TTL = 300.0
 
 
def _cache_get(key: str) -> dict | None:
    entry = _SIGNAL_CACHE.get(key)
    if not entry:
        return None
    ts, payload = entry
    if time.time() - ts > _SIGNAL_CACHE_TTL:
        _SIGNAL_CACHE.pop(key, None)
        return None
    return payload
 
 
def _cache_set(key: str, payload: dict) -> None:
    _SIGNAL_CACHE[key] = (time.time(), payload)
 
 
# ── Signal search endpoint ────────────────────────────────────────────────────
@app.post("/search-signal")
def search_signal(req: SignalSearchRequest):
    cache_key = f"{req.category}|{req.location}|{req.context}"
    cached = _cache_get(cache_key)
    if cached:
        return cached
 
    if req.category == "weather":
        try:
            weather = fetch_weather_wttr(req.location)
            payload = {"results": weather, "query": "wttr.in API"}
            _cache_set(cache_key, payload)
            return payload
        except Exception as e:
            logger.exception("Weather endpoint failed")
            return {"results": f"Weather fetch failed: {e}", "query": "wttr.in API"}
 
    queries_map = _signal_queries(datetime.now().year)
    templates = queries_map.get(req.category, queries_map["raw"])
    ctx_part = f" {req.context} " if req.context else " "
 
    rendered_queries: list[str] = []
    all_results: list[str] = []
    any_rate_limited = False
 
    # On Vercel: only attempt the FIRST template to stay under 10s.
    for template in templates[:1]:
        query = " ".join(template.format(location=req.location, ctx=ctx_part).split())
        rendered_queries.append(query)
 
        results, err = _ddgs_search(query, max_results=8, max_retries=1)
        if err == "rate_limit":
            any_rate_limited = True
        for r in results:
            all_results.append(f"• {r['title']}: {r['body']}")
 
    seen: set[str] = set()
    unique: list[str] = []
    for r in all_results:
        if r in seen:
            continue
        seen.add(r)
        unique.append(r)
        if len(unique) >= 8:
            break
 
    primary_query = rendered_queries[0] if rendered_queries else ""
    if not unique:
        msg = ("Search rate-limited by DuckDuckGo — try again in ~30 seconds."
               if any_rate_limited
               else "Search temporarily unavailable — try again in a moment.")
        return {"results": msg, "query": primary_query}
 
    payload = {"results": "\n".join(unique), "query": primary_query}
    _cache_set(cache_key, payload)
    return payload
 
 
# ── Chat endpoint (single call, no tool loop — fits Vercel's 10s budget) ──────
@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        # Compose system prompt (master + any system messages from history)
        system_msg = MASTER_PROMPT
        chat_messages = []
        for msg in req.messages:
            if msg.get("role") == "system":
                system_msg += "\n" + msg.get("content", "")
            else:
                chat_messages.append(msg)
 
        # Single call. No tools. Live data comes via /search-signal endpoints
        # which the frontend's "Signal Fetch" buttons already use.
        response = await client.messages.create(
            messages=chat_messages,
            model=req.model,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            system=system_msg,
        )
 
        # Extract final text
        final_text = "".join(
            block.text for block in response.content if block.type == "text"
        )
 
        if not final_text:
            final_text = ("(No response generated. The model returned an empty reply — "
                          "please try rephrasing your question.)")
 
        return {
            "content": final_text,
            "model": response.model,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }
    except Exception as e:
        logger.exception("ZAI API ERROR")
        return {
            "content": f"⚠️ Backend error: {type(e).__name__}: {e}",
            "error": str(e),
        }
 
 
# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "online",
        "model": "glm-5-turbo",
        "provider": "Z.AI",
        "key_configured": bool(os.environ.get("ZAI_API_KEY")),
    }
 
# ── Vercel ASGI handler ───────────────────────────────────────────────────────
# Vercel's Python runtime needs an adapter to talk to FastAPI (ASGI).
# Mangum exposes `handler` which Vercel will invoke for each request.
handler = Mangum(app, lifespan="off")

# NOTE: No `if __name__ == "__main__"` block — Vercel imports `app`/`handler` directly.
# To run locally:  uvicorn glm_backend:app --reload --port 8000
