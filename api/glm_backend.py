import os
import time
import logging
import urllib.request
import urllib.parse
import json
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from anthropic import AsyncAnthropic
from duckduckgo_search import DDGS

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("inveniq")

# ── Anthropic client (async, with timeout) ────────────────────────────────────
client = AsyncAnthropic(
    base_url="https://api.ilmu.ai/anthropic",
    api_key="sk-4d730bb5cd8b2f8dcdd4c585c10a2910cbf6bccac7bc0714",
    timeout=60.0,  # don't let a hung upstream hold a worker forever
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request models ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    messages: list
    model: str = "ilmu-glm-5.1"
    temperature: float = 0.1
    max_tokens: int = 8192

class SignalSearchRequest(BaseModel):
    category: str   # "weather", "calendar", "news", "raw"
    location: str = "Malaysia"
    context: str = ""  # CSV-derived keywords e.g. "beverages, snacks, sugar"

# ── Web search tool definition ────────────────────────────────────────────────
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


# ── DuckDuckGo helper (single source of truth) ────────────────────────────────
def _ddgs_search(query: str, max_results: int = 5, max_retries: int = 3) -> tuple[list, str | None]:
    """
    Run a DuckDuckGo text search with retry + rate-limit handling.
    Returns (results, error). On success, error is None.
    """
    last_err: str | None = None
    for attempt in range(max_retries):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if results:
                return results, None
            # empty results – brief pause, try again
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
        except Exception as e:
            err_msg = str(e).lower()
            is_rate_limit = any(
                kw in err_msg for kw in ["429", "rate", "limit", "too many", "throttl", "captcha"]
            )
            last_err = "rate_limit" if is_rate_limit else f"error: {e}"
            if attempt < max_retries - 1:
                wait = (4 if is_rate_limit else 2) * (attempt + 1)
                logger.warning("DDGS %s – waiting %ss before retry %d", last_err, wait, attempt + 1)
                time.sleep(wait)
    return [], last_err


def perform_web_search(query: str) -> str:
    """Web search used by the AI's tool-use loop. Returns formatted markdown string."""
    results, err = _ddgs_search(query, max_results=5)
    if results:
        return "\n\n".join(
            f"- **{r['title']}**\n  {r['body']}\n  Source: {r['href']}"
            for r in results
        )
    if err == "rate_limit":
        return "Search rate limited — please wait a moment and try again."
    if err:
        return f"Search error: {err}"
    return "No results found for this query."


# ── System prompt (the AI's personality) ──────────────────────────────────────
# 🧠 THIS IS YOUR AI'S BRAIN/PERSONALITY!
# Edit this text to make the AI behave exactly how you want.
MASTER_PROMPT = """You are InvenIQ, an elite, professional Inventory Intelligence AI.
Your primary job is to help the user manage their stock, analyze sales data, and predict inventory shortages.

RULES:
1. Be highly analytical, precise, and professional.
2. Format your answers clearly using bullet points or short paragraphs.
3. When the user asks about current news, events, market trends, weather, calendar info, or any real-time unstructured data, use the web_search tool to fetch up-to-date information before answering.
4. If you need more data (like a CSV file or numbers) to answer a question, ask the user to provide it.
"""

# ── Signal query templates (year is filled in dynamically) ────────────────────
def _signal_queries(year: int) -> dict[str, list[str]]:
    return {
        "weather": [],  # handled by wttr.in API directly
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
    """Fetch real weather from wttr.in — free, no API key, fast."""
    cities = ["Kuala Lumpur", "Johor Bahru", "Penang", "Kota Kinabalu"]
    parts = []
    for city in cities:
        try:
            url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
            req = urllib.request.Request(url, headers={"User-Agent": "curl/7.68.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
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
                # wttr returns 8 three-hour blocks; index 4 ≈ midday (12:00)
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


# ── Signal search endpoint ────────────────────────────────────────────────────
@app.post("/search-signal")
def search_signal(req: SignalSearchRequest):
    # ── Weather: use wttr.in directly ──
    if req.category == "weather":
        try:
            weather = fetch_weather_wttr(req.location)
            return {"results": weather, "query": "wttr.in API"}
        except Exception as e:
            logger.exception("Weather endpoint failed")
            return {"results": f"Weather fetch failed: {e}", "query": "wttr.in API"}

    # ── Other categories: aggregate across all template queries ──
    queries_map = _signal_queries(datetime.now().year)
    templates = queries_map.get(req.category, queries_map["raw"])

    # Build ctx fragment with clean spacing
    ctx_part = f" {req.context} " if req.context else " "

    rendered_queries: list[str] = []
    all_results: list[str] = []

    for template in templates:
        # Render the template, then collapse any double-spacing
        query = " ".join(template.format(location=req.location, ctx=ctx_part).split())
        rendered_queries.append(query)

        results, _ = _ddgs_search(query, max_results=6, max_retries=2)
        for r in results:
            all_results.append(f"• {r['title']}: {r['body']}")

    # Deduplicate while preserving order, cap at 8
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
        return {
            "results": "Search temporarily unavailable — try again in a moment.",
            "query": primary_query,
        }
    return {"results": "\n".join(unique), "query": primary_query}


# ── Chat endpoint (async + tool-use loop) ─────────────────────────────────────
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

        common_kwargs = {
            "model": req.model,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "system": system_msg,
            "tools": [WEB_SEARCH_TOOL],
        }

        # First call
        response = await client.messages.create(messages=chat_messages, **common_kwargs)

        # Tool-use loop
        max_iterations = 5
        for iteration in range(max_iterations):
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                break

            # Append assistant's tool-call turn
            chat_messages.append({"role": "assistant", "content": response.content})

            # Execute each tool call
            tool_results = []
            unknown_tool_seen = False
            for block in tool_use_blocks:
                if block.name == "web_search":
                    query = block.input.get("query", "")
                    search_result = perform_web_search(query)
                else:
                    logger.warning("Unknown tool requested: %s", block.name)
                    search_result = f"Unknown tool: {block.name}"
                    unknown_tool_seen = True
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": search_result,
                })

            chat_messages.append({"role": "user", "content": tool_results})

            # If the model hallucinated a tool, do one more turn so it can recover
            # with the error message, then stop to avoid burning iterations.
            response = await client.messages.create(messages=chat_messages, **common_kwargs)
            if unknown_tool_seen:
                break
        else:
            logger.warning("Tool-use loop hit max_iterations=%d", max_iterations)

        # Extract final text
        final_text = "".join(
            block.text for block in response.content if block.type == "text"
        )

        return {
            "content": final_text,
            "model": response.model,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }
    except Exception as e:
        logger.exception("ILMU API ERROR")
        return {"error": str(e)}
