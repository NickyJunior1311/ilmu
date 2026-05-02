import os
import time
import logging
import urllib.request
import urllib.parse
import urllib.error
import json
import re
from datetime import datetime
from html.parser import HTMLParser
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from anthropic import AsyncAnthropic

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
    api_key=os.getenv("ZAI_API_KEY", "7959c551678b4ff2ad679e6994d49017.4XAHAELzkqpWdutF"),
    timeout=60.0,
)

app = FastAPI(title="InvenIQ Backend (ILMU)", version="4.3.0-vercel")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request models ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    messages: list
    model: str = "glm-5.1"
    temperature: float = 0.1
    # DEF-01 FIX: Increased from 2000 → 4096 to prevent mid-response JSON
    # truncation. Previously the LLM would hit the token cap and omit the
    # closing `}` bracket, causing frontend JSON.parse() to throw
    # "Unexpected end of JSON input" and the dashboard cards to crash.
    max_tokens: int = 4096

class SignalSearchRequest(BaseModel):
    category: str   # "weather", "calendar", "news", "raw"
    location: str = "Malaysia"
    context: str = ""

# ── Optional API keys (set in Vercel/Render env vars for best reliability) ────
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "").strip()

# Common browser User-Agent to look like a real client
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


# ── Search providers (multi-provider fallback chain) ──────────────────────────
def _http_get_json(url: str, headers: dict | None = None, timeout: float = 5.0) -> dict | None:
    """GET a URL and parse JSON. Returns None on failure."""
    try:
        req = urllib.request.Request(url, headers=headers or {"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as e:
        logger.warning("HTTP GET failed for %s: %s", url[:80], e)
        return None


def _http_post_json(url: str, payload: dict, headers: dict | None = None, timeout: float = 5.0) -> dict | None:
    """POST JSON and parse JSON response. Returns None on failure."""
    try:
        body = json.dumps(payload).encode("utf-8")
        hdrs = {"Content-Type": "application/json", "User-Agent": _UA}
        if headers:
            hdrs.update(headers)
        req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as e:
        logger.warning("HTTP POST failed for %s: %s", url[:80], e)
        return None


def _search_brave(query: str, max_results: int = 8, timeout: float = 5.0) -> list[dict] | None:
    """Brave Search API. Free tier: 2000 queries/month. Most reliable option."""
    if not BRAVE_API_KEY:
        return None
    url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode({
        "q": query,
        "count": max_results,
    })
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_API_KEY,
        "User-Agent": _UA,
    }
    data = _http_get_json(url, headers=headers, timeout=timeout)
    if not data:
        return None
    web = (data.get("web") or {}).get("results") or []
    return [
        {"title": r.get("title", ""), "body": r.get("description", ""), "href": r.get("url", "")}
        for r in web
    ]


def _search_tavily(query: str, max_results: int = 8, timeout: float = 6.0) -> list[dict] | None:
    """Tavily Search API. Built for AI use cases."""
    if not TAVILY_API_KEY:
        return None
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
    }
    data = _http_post_json("https://api.tavily.com/search", payload, timeout=timeout)
    if not data:
        return None
    results = data.get("results") or []
    return [
        {"title": r.get("title", ""), "body": r.get("content", ""), "href": r.get("url", "")}
        for r in results
    ]


def _search_serpapi(query: str, max_results: int = 8, timeout: float = 5.0) -> list[dict] | None:
    """SerpAPI (Google results). Paid but very reliable."""
    if not SERPAPI_KEY:
        return None
    url = "https://serpapi.com/search.json?" + urllib.parse.urlencode({
        "q": query,
        "engine": "google",
        "num": max_results,
        "api_key": SERPAPI_KEY,
    })
    data = _http_get_json(url, timeout=timeout)
    if not data:
        return None
    organic = data.get("organic_results") or []
    return [
        {"title": r.get("title", ""), "body": r.get("snippet", ""), "href": r.get("link", "")}
        for r in organic
    ]


# ── HTML-based fallback: scrape DuckDuckGo's HTML endpoint directly ───────────
class _DDGHTMLParser(HTMLParser):
    """Minimal parser for DuckDuckGo's html.duckduckgo.com results page."""
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict] = []
        self._current: dict | None = None
        self._capture: str | None = None  # "title" | "snippet" | None
        self._buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs_d = dict(attrs)
        cls = attrs_d.get("class", "")
        if tag == "a" and "result__a" in cls:
            if self._current is None:
                self._current = {"title": "", "body": "", "href": ""}
            href = attrs_d.get("href", "")
            # DDG wraps URLs in /l/?uddg=<encoded>
            if "uddg=" in href:
                try:
                    qs = urllib.parse.urlparse(href).query
                    parsed = urllib.parse.parse_qs(qs)
                    href = parsed.get("uddg", [href])[0]
                except Exception:
                    pass
            self._current["href"] = href
            self._capture = "title"
            self._buf = []
        elif tag == "a" and "result__snippet" in cls:
            self._capture = "snippet"
            self._buf = []

    def handle_endtag(self, tag):
        if tag == "a" and self._capture == "title" and self._current is not None:
            self._current["title"] = "".join(self._buf).strip()
            self._capture = None
            self._buf = []
        elif tag == "a" and self._capture == "snippet" and self._current is not None:
            self._current["body"] = "".join(self._buf).strip()
            self._capture = None
            self._buf = []
            # snippet usually closes a result block
            if self._current.get("title") and self._current.get("href"):
                self.results.append(self._current)
            self._current = None

    def handle_data(self, data):
        if self._capture:
            self._buf.append(data)


def _search_ddg_html(query: str, max_results: int = 8, timeout: float = 6.0) -> list[dict] | None:
    """
    Scrape DuckDuckGo's HTML endpoint directly. More reliable than the
    duckduckgo_search Python library because it doesn't go through DDG's
    JS-protected endpoints. Still subject to rate limiting on cloud IPs.
    """
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    headers = {
        "User-Agent": _UA,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning("DDG HTML fetch failed: %s", e)
        return None

    parser = _DDGHTMLParser()
    try:
        parser.feed(html)
    except Exception as e:
        logger.warning("DDG HTML parse failed: %s", e)
        return None

    return parser.results[:max_results] if parser.results else None


def _search_wikipedia(query: str, max_results: int = 5, timeout: float = 4.0) -> list[dict] | None:
    """
    Wikipedia search as last-resort fallback. Always works, no auth needed.
    Useful for general/encyclopedic context (events, holidays, places).
    """
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": max_results,
        "format": "json",
    })
    data = _http_get_json(url, timeout=timeout)
    if not data:
        return None
    hits = ((data.get("query") or {}).get("search")) or []
    results = []
    for h in hits:
        title = h.get("title", "")
        snippet = re.sub(r"<[^>]+>", "", h.get("snippet", ""))  # strip HTML tags
        page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
        results.append({"title": title, "body": snippet, "href": page_url})
    return results or None


# ── Unified search with provider fallback chain ───────────────────────────────
def web_search(query: str, max_results: int = 8) -> tuple[list[dict], str]:
    """
    Try providers in order of reliability. Returns (results, provider_name).
    Provider order:
      1. Brave Search API   (if BRAVE_API_KEY set)   — best
      2. Tavily             (if TAVILY_API_KEY set)
      3. SerpAPI            (if SERPAPI_KEY set)
      4. DuckDuckGo HTML scrape                       — free, no key
      5. Wikipedia                                    — guaranteed fallback
    """
    providers = [
        ("brave", _search_brave),
        ("tavily", _search_tavily),
        ("serpapi", _search_serpapi),
        ("ddg_html", _search_ddg_html),
        ("wikipedia", _search_wikipedia),
    ]
    for name, fn in providers:
        try:
            results = fn(query, max_results=max_results)
        except Exception as e:
            logger.warning("Provider %s threw exception: %s", name, e)
            results = None
        if results:
            logger.info("Search OK via %s: %d results for %r", name, len(results), query[:60])
            return results, name
        logger.info("Provider %s returned no results for %r", name, query[:60])
    return [], "none"


# ── System prompt ─────────────────────────────────────────────────────────────
MASTER_PROMPT = """You are InvenIQ, an elite, professional Inventory Intelligence AI.
Your primary job is to help the user manage their stock, analyze sales data, and predict inventory shortages.

RULES:
1. Be highly analytical, precise, and professional.
2. Format your answers clearly using bullet points or short paragraphs.
3. You do NOT have live web access in this chat. Answer from your existing knowledge.
4. If the user asks about current news, weather, calendar dates, or market trends, tell them to use the dedicated "Signal Fetch" buttons in the side panel (Calendar / Weather / News / Raw) which fetch live data, then paste the results into the chat for you to analyze.
5. If you need more data (like a CSV file or numbers) to answer a question, ask the user to provide it.
6. ANTI-HALLUCINATION PROTOCOL [DEF-02 FIX]: NEVER invent or infer product names. Only reference products whose IDs/names appear verbatim in the user-provided CSV. If no CSV has been loaded, or a product is not in the CSV, say so explicitly — do NOT guess names like "Maggi Noodles" based on weather, news, or contextual signals. Strictly read CSV IDs.
7. When discussing stock impact from external signals (rain, holidays, news), only correlate them to products that ACTUALLY EXIST in the loaded CSV. If the CSV is empty, ask the user to upload one.
"""

# ── Signal query templates ────────────────────────────────────────────────────
def _signal_queries(year: int) -> dict[str, list[str]]:
    return {
        "weather": [],  # handled by wttr.in
        "calendar": [
            f"Malaysia public holidays {year}",
            f"Malaysia school holidays calendar {year}{{ctx}}",
            f"Malaysia events festivals {year}{{ctx}}",
        ],
        "news": [
            f"Malaysia{{ctx}}retail market news {year}",
            f"Malaysia{{ctx}}supply chain business news {year}",
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
    cities = ["Kuala Lumpur", "Penang"]
    parts = []
    for city in cities:
        try:
            url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
            req = urllib.request.Request(url, headers={"User-Agent": "curl/7.68.0"})
            with urllib.request.urlopen(req, timeout=2) as resp:
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
# DEF-03 FIX: Decoupled from /chat to prevent slow third-party APIs (wttr.in,
# search providers) from blocking the AI response. Previously, synchronous
# fetching of 4 weather cities inside the chat handler caused the entire
# request to TimeoutError after 10s when wttr.in was slow. Signal fetching
# now lives in this dedicated endpoint with 5-min in-memory caching, called
# separately by the frontend's "Signal Fetch" buttons. Chat stays snappy;
# signals stay fresh-but-cheap.
@app.post("/search-signal")
def search_signal(req: SignalSearchRequest):
    cache_key = f"{req.category}|{req.location}|{req.context}"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    if req.category == "weather":
        try:
            weather = fetch_weather_wttr(req.location)
            payload = {"results": weather, "query": "wttr.in API", "provider": "wttr.in"}
            _cache_set(cache_key, payload)
            return payload
        except Exception as e:
            logger.exception("Weather endpoint failed")
            return {"results": f"Weather fetch failed: {e}", "query": "wttr.in API", "provider": "wttr.in"}

    queries_map = _signal_queries(datetime.now().year)
    templates = queries_map.get(req.category, queries_map["raw"])
    ctx_part = f" {req.context} " if req.context else " "

    rendered_queries: list[str] = []
    all_results: list[str] = []
    last_provider = "none"

    # Try up to 2 templates to maximise the chance of useful data while
    # staying within typical serverless timeout budgets (~10s on Vercel Hobby).
    for template in templates[:2]:
        query = " ".join(template.format(location=req.location, ctx=ctx_part).split())
        rendered_queries.append(query)

        results, provider = web_search(query, max_results=8)
        last_provider = provider if provider != "none" else last_provider
        for r in results:
            title = r.get("title", "").strip()
            body = r.get("body", "").strip()
            if title or body:
                all_results.append(f"• {title}: {body}")

        # If we already have enough results from the first query, stop early
        if len(all_results) >= 8:
            break

    # Deduplicate while preserving order
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
        msg = (
            "Live search is currently unavailable from all providers. "
            "To enable reliable search, set one of these env vars in your "
            "Vercel/Render dashboard: BRAVE_API_KEY (free 2000/mo at "
            "https://brave.com/search/api/), TAVILY_API_KEY, or SERPAPI_KEY. "
            "You can still ask the AI questions using its existing knowledge."
        )
        return {"results": msg, "query": primary_query, "provider": "none"}

    payload = {
        "results": "\n".join(unique),
        "query": primary_query,
        "provider": last_provider,
    }
    _cache_set(cache_key, payload)
    return payload


# ── Chat endpoint ─────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        system_msg = MASTER_PROMPT
        chat_messages = []
        for msg in req.messages:
            if msg.get("role") == "system":
                system_msg += "\n" + msg.get("content", "")
            else:
                chat_messages.append(msg)

        response = await client.messages.create(
            messages=chat_messages,
            model=req.model,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            system=system_msg,
        )

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
        "model": "glm-5.1",
        "provider": "Z.AI",
        "search_providers": {
            "brave": bool(BRAVE_API_KEY),
            "tavily": bool(TAVILY_API_KEY),
            "serpapi": bool(SERPAPI_KEY),
            "ddg_html": True,
            "wikipedia": True,
        },
    }


# ── Search debug endpoint (useful for diagnosing which provider works) ────────
@app.get("/search-debug")
def search_debug(q: str = "Malaysia public holidays 2026"):
    """Run a search and report which provider succeeded. Hit this in a browser
    to verify search is working: https://your-app/search-debug?q=test"""
    results, provider = web_search(q, max_results=5)
    return {
        "query": q,
        "provider": provider,
        "result_count": len(results),
        "results": results[:5],
    }


# ── Serve frontend ────────────────────────────────────────────────────────────
@app.get("/")
def serve_frontend():
    """Serve inventory_chat_api.html as the homepage."""
    return FileResponse("inventory_chat_api.html")

# NOTE: To run locally:  uvicorn glm_backend:app --reload --port 8000
# On Render: set Start Command to `uvicorn glm_backend:app --host 0.0.0.0 --port $PORT`
#
# To enable reliable web search, set ONE of these env vars:
#   BRAVE_API_KEY  — free 2000/mo, sign up at https://brave.com/search/api/
#   TAVILY_API_KEY — https://tavily.com
#   SERPAPI_KEY    — https://serpapi.com
# Without any keys, the backend falls back to scraping DuckDuckGo HTML and
# Wikipedia — these work but are less reliable on cloud IPs.
