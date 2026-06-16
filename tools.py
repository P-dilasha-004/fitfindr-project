"""
tools.py

The three required FitFindr tools. Each tool is a standalone function that
can be called and tested independently before being wired into the agent loop.

Complete and test each tool before moving to agent.py.

Tools:
    search_listings(description, size, max_price)  → list[dict]
    suggest_outfit(new_item, wardrobe)              → str
    create_fit_card(outfit, new_item)               → str
"""

import os
import json

from dotenv import load_dotenv
from groq import Groq

from utils.data_loader import load_listings

load_dotenv()


# ── Groq client ───────────────────────────────────────────────────────────────

def _get_groq_client():
    """Initialize and return a Groq client using GROQ_API_KEY from .env."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Add it to a .env file in the project root."
        )
    return Groq(api_key=api_key)


# ── Tool 1: search_listings ───────────────────────────────────────────────────

def search_listings(
    description: str,
    size: str | None = None,
    max_price: float | None = None,
) -> list[dict] | str:
    """
    Search the mock listings dataset for items matching the description,
    optional size, and optional price ceiling.

    Args:
        description: Keywords describing what the user is looking for
                     (e.g., "vintage graphic tee").
        size:        Size string to filter by, or None to skip size filtering.
                     Matching is case-insensitive (e.g., "M" matches "S/M").
        max_price:   Maximum price (inclusive), or None to skip price filtering.

    Returns:
        On success: a list of matching listing dicts sorted by relevance
                    (best match first). Each dict has fields:
                    id, title, description, category, style_tags (list), size,
                    condition, price (float), colors (list), brand, platform.
        On failure: a plain string describing exactly which constraint produced
                    no results and what the user can try instead. The agent
                    should surface this message directly rather than continuing
                    the pipeline.

    Failure modes (each returns its own message instead of raising):
        1. Price filter → nothing left: suggests raising the budget.
        2. Size filter → nothing left: suggests a different size or looser budget.
        3. Keyword scoring → no overlap: suggests broader search terms.
    """
    # Words that carry no search signal — skip these when comparing keywords
    STOP_WORDS = {
        "a", "an", "the", "and", "or", "for", "in", "on", "with",
        "is", "it", "to", "of", "at", "by", "this", "that", "i",
        "my", "me", "some", "what", "how", "looking", "want",
    }

    # ── Step 1: load the full catalogue ──────────────────────────────────────
    listings = load_listings()

    # ── Step 2a: price filter ─────────────────────────────────────────────────
    if max_price is not None:
        after_price = [l for l in listings if l["price"] <= max_price]
        if not after_price:
            # Nothing in the entire catalogue fits the budget — tell the user
            # the cheapest item so they have a concrete number to aim for.
            cheapest = min(listings, key=lambda l: l["price"])["price"]
            return (
                f"No listings found under ${max_price:.2f}. "
                f"The lowest-priced item in the catalogue is ${cheapest:.2f}. "
                f"Try raising your budget."
            )
    else:
        after_price = listings

    # ── Step 2b: size filter ──────────────────────────────────────────────────
    if size is not None:
        # Substring match so "M" catches "S/M", "XS/M", etc.
        after_size = [
            l for l in after_price
            if size.upper() in l.get("size", "").upper()
        ]
        if not after_size:
            # Give the user context: how many items existed before the size cut
            budget_note = f" under ${max_price:.2f}" if max_price else ""
            return (
                f"No listings in size '{size}'{budget_note}. "
                f"There are {len(after_price)} items{budget_note}, but none match that size. "
                f"Try a neighbouring size or remove the size filter."
            )
    else:
        after_size = after_price

    # keyword relevance scoring 
    # Build a set of meaningful query tokens from the user's description
    query_tokens = {
        word.lower()
        for word in description.split()
        if word.lower() not in STOP_WORDS
    }

    def _score(listing: dict) -> int:
        """
        Count how many distinct query tokens appear in the listing's searchable
        fields. One point per token regardless of frequency, so common words
        don't inflate a score artificially.
        """
        searchable_parts = [
            listing.get("title", ""),
            listing.get("description", ""),
            listing.get("category", ""),
            listing.get("brand", "") or "",       # brand can be None
            " ".join(listing.get("style_tags", [])),
            " ".join(listing.get("colors", [])),
        ]
        listing_text = " ".join(searchable_parts).lower()
        return sum(1 for token in query_tokens if token in listing_text)

    scored = [(listing, _score(listing)) for listing in after_size]

    # drop listings with zero keyword overlap 
    scored = [(listing, score) for listing, score in scored if score > 0]

    if not scored:
        # Filters passed but no listing text matched the search terms at all.
        # Show the user which tokens were searched so they can adjust.
        tokens_searched = ", ".join(f'"{t}"' for t in sorted(query_tokens))
        budget_note = f" under ${max_price:.2f}" if max_price else ""
        size_note = f" in size '{size}'" if size else ""
        return (
            f"Found {len(after_size)} listing(s){size_note}{budget_note}, "
            f"but none matched the search terms ({tokens_searched}). "
            f"Try broader keywords — for example, use a category like "
            f"'tops', 'jeans', or 'jacket' instead of a specific style."
        )

    # sort best match first, return only the listing dicts 
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [listing for listing, _ in scored]


# ── Tool 2: suggest_outfit ────────────────────────────────────────────────────
def suggest_outfit(new_item: dict, wardrobe: dict) -> list[dict] | str:
    """
    Given a thrifted item and the user's wardrobe, suggest 1–2 complete outfits.

    Args:
        new_item: A listing dict (the item the user is considering buying).
        wardrobe: A wardrobe dict with an 'items' key containing a list of
                  wardrobe item dicts. May be empty — handle this gracefully.

    Returns:
        Case 1 — empty wardrobe: a plain string with general styling advice
                 (no owned pieces to reference, so the agent surfaces this
                 directly rather than passing it to create_fit_card).
        Case 2 — populated wardrobe: a list[dict] of the selected wardrobe
                 items that complete the outfit. Each dict is taken verbatim
                 from wardrobe['items'] — the LLM never invents new fields.
                 Pass this list straight to create_fit_card.

    Failure mode (Case 2 only):
        If the LLM response cannot be parsed as a JSON array, returns a
        descriptive error string so the agent can report the issue without
        crashing the pipeline.
    """
    client = _get_groq_client()

    # Concise item description shared by both prompt branches.
    # Price needs a numeric guard — :.2f crashes if the field is absent.
    _price = new_item.get('price')
    _price_str = f"${_price:.2f}" if isinstance(_price, (int, float)) else "$?"

    item_summary = (
        f"{new_item.get('title', 'Unknown item')} "
        f"({_price_str}, size {new_item.get('size', '?')}, "
        f"{new_item.get('condition', '?')} condition) — "
        f"colors: {', '.join(new_item.get('colors', []))}; "
        f"style tags: {', '.join(new_item.get('style_tags', []))}."
    )

    # The correct input shape is {"items": [...]}.
    # Guard against the full schema wrapper being passed accidentally —
    # that file has keys "schema", "example_wardrobe", "empty_wardrobe"
    # but no top-level "items" key, so a naive .get("items", []) returns []
    # and incorrectly triggers the empty-wardrobe branch (Case 1).
    if "items" in wardrobe:
        wardrobe_items = wardrobe["items"]
    elif "example_wardrobe" in wardrobe:
        # Caller passed the whole wardrobe_schema.json wrapper — unwrap it
        wardrobe_items = wardrobe["example_wardrobe"].get("items", [])
    else:
        wardrobe_items = []

    # ── Case 1: empty wardrobe — return conversational styling advice ─────────
    if not wardrobe_items:
        prompt = (
            f"A user just thrifted this item: {item_summary}\n\n"
            "They haven't saved any wardrobe items yet, so you can't reference "
            "specific pieces they own. Instead, suggest 1–2 complete outfit ideas "
            "that would work well with this item. Describe what category and vibe "
            "of pieces pair best (e.g., 'a slim black trouser and loafers for a "
            "smart-casual look'). Keep the tone casual and practical — two short "
            "paragraphs at most."
        )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are FitFindr, a friendly thrift-fashion stylist. "
                        "Give practical, specific outfit suggestions."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()

    # ── Case 2: populated wardrobe — return a JSON array of selected items ─────

    # Serialise the wardrobe as JSON so the LLM works from the exact same
    # data structure it must echo back — reducing the chance of drift.
    wardrobe_json = json.dumps(wardrobe_items, indent=2)

    prompt = (
        f"New thrifted item: {item_summary}\n\n"
        f"User's wardrobe (JSON):\n{wardrobe_json}\n\n"
        "Select 2–4 items from the wardrobe above that pair well with the new "
        "thrifted item to form 1–2 complete outfits.\n\n"
        "OUTPUT RULES — you must follow these exactly:\n"
        "1. Output ONLY a valid JSON array. No prose, no markdown, no code fences.\n"
        "2. Each element must be a dict copied verbatim from the wardrobe JSON above.\n"
        "3. Do NOT add, rename, or invent any fields. Do NOT include any item not "
        "present in the wardrobe JSON.\n"
        "4. Your entire response must be parseable by json.loads(). "
        "If you output anything outside the JSON array, the pipeline breaks."
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a JSON-only API endpoint. "
                    "You output raw JSON arrays and nothing else. "
                    "No explanations, no markdown, no extra text of any kind."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,   # low temperature keeps output deterministic and format-safe
        max_tokens=600,
    )

    raw = response.choices[0].message.content.strip()

    # ── Bulletproof JSON Extraction ───────────────────────────────────────────
    # Find the bounds of the JSON array, ignoring any hallucinated text outside it
    start_idx = raw.find('[')
    end_idx = raw.rfind(']')
    
    if start_idx != -1 and end_idx != -1:
        # Slice out just the array
        raw = raw[start_idx:end_idx + 1]

    # Parse and validate — return an informative error string on failure so the
    # agent can report the problem without propagating an exception
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            return (
                "suggest_outfit returned unexpected JSON (expected a list, "
                f"got {type(parsed).__name__}). Cannot build an outfit."
            )
        return parsed
    except json.JSONDecodeError as exc:
        return (
            f"suggest_outfit could not parse the LLM response as JSON ({exc}). "
            "Try re-running or simplifying the wardrobe."
        )


# ── Tool 3: create_fit_card ───────────────────────────────────────────────────

def create_fit_card(outfit: list, new_item: dict) -> dict:
    """
    Translate a structured outfit list into a social-media-ready fit card.

    Args:
        outfit:   list[dict] — wardrobe items selected by suggest_outfit (Case 2).
                  An empty list or any non-list value triggers the fallback.
        new_item: dict — the thrifted listing from search_listings. Must contain
                  at least 'title'; uses 'price' and 'platform' when present.

    Returns:
        A dict with two keys:
            "caption"  (str)       — 2–4 sentence casual OOTD caption that
                                     mentions item name, price, and platform
                                     once each in a natural way.
            "hashtags" (list[str]) — 3–5 relevant hashtags (e.g. "#thrifted").

    Fallback (no LLM call):
        If outfit is empty/not a list, or if the LLM call raises any exception,
        the function returns a hardcoded fallback dict built from new_item data
        so the pipeline always terminates with a usable response.
    """
    # ── Guard: empty or invalid outfit — bypass LLM entirely ─────────────────
    # An empty outfit means suggest_outfit found nothing to pair; a non-list
    # means it returned an error string (Case 1 / parse failure).
    if not isinstance(outfit, list) or not outfit:
        title = new_item.get("title", "this thrifted find")
        return {
            "caption": f"Check out my new {title}! Grab it before it's gone.",
            "hashtags": ["#thrifted", "#secondhand", "#ootd", "#sustainablefashion"],
        }

    # ── Build human-readable outfit summary for the prompt ───────────────────
    # Each wardrobe item contributes its name and dominant color(s).
    outfit_lines = []
    for piece in outfit:
        name = piece.get("name", "wardrobe piece")
        colors = ", ".join(piece.get("colors", []))
        line = f"- {name}" + (f" ({colors})" if colors else "")
        outfit_lines.append(line)
    outfit_summary = "\n".join(outfit_lines)

    # Extract new_item metadata — gracefully handle missing fields
    item_title = new_item.get("title", "thrifted find")
    item_price = new_item.get("price")
    price_str = f"${item_price:.2f}" if isinstance(item_price, (int, float)) else "a steal"
    platform = new_item.get("platform", "a thrift app")

    prompt = (
        f"New thrifted item: {item_title} — found on {platform} for {price_str}.\n\n"
        f"Outfit it's styled with:\n{outfit_summary}\n\n"
        "Write a fit card for this outfit. Follow these rules exactly:\n"
        "1. Output ONLY a valid JSON object — no prose, no markdown, no code fences.\n"
        "2. The JSON must have exactly two keys:\n"
        '   "caption": a 2–4 sentence OOTD caption. Casual, specific, authentic — '
        "not a product description. Mention the item name, price, and platform "
        "once each in a natural way. Describe the outfit vibe concretely.\n"
        '   "hashtags": a JSON array of 3–5 hashtag strings (e.g. "#thrifted").\n'
        "3. Your entire response must be parseable by json.loads()."
    )

    # ── LLM call — catch all exceptions so the pipeline never crashes ─────────
    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a JSON-only API. Output raw JSON objects, nothing else. "
                        "No explanations, no markdown, no extra text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,   # high temperature produces varied, creative captions
            max_tokens=300,
        )

        raw = response.choices[0].message.content.strip()

        # Extract the JSON object even if the LLM wraps it in stray text
        start_idx = raw.find("{")
        end_idx = raw.rfind("}")
        if start_idx != -1 and end_idx != -1:
            raw = raw[start_idx : end_idx + 1]

        parsed = json.loads(raw)

        # Validate the two required keys are present strings/lists
        if not isinstance(parsed.get("caption"), str):
            raise ValueError("'caption' key missing or not a string")
        if not isinstance(parsed.get("hashtags"), list):
            raise ValueError("'hashtags' key missing or not a list")

        return parsed

    except Exception:
        # Any failure (network, parse error, missing key) → hardcoded fallback
        return {
            "caption": (
                f"Just scored this {item_title} on {platform} for {price_str}. "
                "Styled it with pieces from my own closet for an effortless look."
            ),
            "hashtags": ["#thrifted", "#secondhand", "#ootd", "#sustainablefashion"],
        }


results_a = search_listings(
        description="vintage denim",
        size="M",
        max_price=50.00
    )

# ── STEP 2: The Handoff Logic ─────────────────────────────────────────────
    # We must verify Tool 1 returned a list (success) and not a string (failure)
if isinstance(results_a, list) and len(results_a) > 0:
        top_thrifted_item = results_a[0]
        print(f"Extracted Top Match: {top_thrifted_item['title']}")
        
        # Load the user's closet
        try:
            with open("data/wardrobe_schema.json", "r") as f:
                real_wardrobe = json.load(f)
        except FileNotFoundError:
            print("Error: Could not find 'data/wardrobe_schema.json'.")
            exit(1)

        print("Tool 1 Output")
        print(results_a)

        # ── STEP 3: Call Tool 2 (Suggest Outfit) ──────────────────────────────
        outfit_output = suggest_outfit(top_thrifted_item, real_wardrobe)
        
        print("\n--- FINAL TOOL 2 OUTPUT ---")
        print(f"Data Type: {type(outfit_output)}")
        
        # Pretty-print the JSON if it's a list, otherwise just print the string
        if isinstance(outfit_output, list):
            print(json.dumps(outfit_output, indent=2))
        else:
            print(outfit_output)

     # ── STEP 3: Call Tool 2 (Suggest Outfit) ──────────────────────────────
        print("\n--- STEP 3: GENERATING OUTFIT ---")
        # CRITICAL FIX: Extract the actual example_wardrobe dict so it isn't "empty"
        user_closet = real_wardrobe.get("example_wardrobe", {})
        
        outfit_output = suggest_outfit(top_thrifted_item, user_closet)
        
        print(f"Tool 2 Output Type: {type(outfit_output)}")
        if isinstance(outfit_output, list):
            print("Successfully built outfit from wardrobe.")
        else:
            print("Notice: Wardrobe empty or error. Tool 2 returned a string fallback.")

        # ── STEP 4: Call Tool 3 (Create Fit Card) ─────────────────────────────
        print("\n--- STEP 4: GENERATING FIT CARD ---")
        fit_card = create_fit_card(outfit_output, top_thrifted_item)
        
        print("\n==================================================")
        print("FINAL AGENT OUTPUT (SOCIAL MEDIA POST)")
        print("==================================================")
        print(json.dumps(fit_card, indent=2))

else:
        # If Tool 1 failed (e.g., budget too low), print the fallback string
        print(f"Pipeline halted at Tool 1. Reason:\n{results_a}")


