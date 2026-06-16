"""
agent.py

FitFindr ReAct agent. At each turn the LLM reasons about what to do next
(Thought), calls a tool (Action), and reads the result (Observation) —
repeating until the full pipeline is complete or a failure halts it.

Usage:
    from agent import run_agent
    from utils.data_loader import get_example_wardrobe

    result = run_agent(
        query="vintage graphic tee under $30, size M",
        wardrobe=get_example_wardrobe(),
    )
    print(result["fit_card"])
    print(result["error"])   # None on success
"""

import json
import os

from dotenv import load_dotenv
from groq import Groq
from tools import search_listings, suggest_outfit, create_fit_card

load_dotenv()


# ── Tool schemas (Groq function-calling format) ───────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_listings",
            "description": (
                "Search the secondhand clothing catalogue for items that match "
                "the user's request. Always call this tool first. "
                "Returns a JSON array of matching listings on success, or a plain "
                "error string if nothing passes the filters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": (
                            "Style keywords describing the item "
                            "(e.g. 'vintage graphic tee', 'chunky knit sweater')."
                        ),
                    },
                    "size": {
                        "type": "string",
                        "description": (
                            "Size to filter by (e.g. 'M', 'S/M', '32', 'US 8'). "
                            "Omit entirely if the user did not mention a size."
                        ),
                    },
                    "max_price": {
                        "type": "number",
                        "description": (
                            "Maximum price in USD (inclusive). "
                            "Omit entirely if the user did not mention a budget."
                        ),
                    },
                },
                "required": ["description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_outfit",
            "description": (
                "Build a complete outfit around a thrifted item using the user's wardrobe. "
                "Call this after search_listings has returned at least one item. "
                "Returns a JSON array of wardrobe pieces that complete the outfit, "
                "or a styling advice string when the wardrobe is empty."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {
                        "type": "string",
                        "description": (
                            "The 'id' field of the listing to style, "
                            "taken directly from the search_listings results."
                        ),
                    },
                },
                "required": ["item_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_fit_card",
            "description": (
                "Generate a social-media caption and hashtags for the complete outfit. "
                "Call this after suggest_outfit has returned. "
                "Returns a JSON object with 'caption' (str) and 'hashtags' (list[str])."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are FitFindr, a thrift-fashion assistant. Help the user find a secondhand \
item and style a complete outfit around it.

You have three tools. Call them in this exact order every time:
  1. search_listings  — find items matching the user's query
  2. suggest_outfit   — build an outfit from the user's wardrobe around the best result
  3. create_fit_card  — produce a social-media caption for the finished outfit

ReAct rules:
- Think before each tool call: what do you know, what do you need next?
- Call search_listings first, always.
- If search_listings returns an error string (no matches found), STOP. \
Do NOT call suggest_outfit or create_fit_card. Report the error clearly to the user.
- If search_listings returns items, pick the best match and call suggest_outfit \
with its 'id'.
- After suggest_outfit returns, call create_fit_card to finalise the output.
- After create_fit_card returns, write a short, friendly summary for the user. \
Do not call any more tools after that.\
"""


# ── Session state ─────────────────────────────────────────────────────────────

def _new_session(query: str, wardrobe: dict) -> dict:
    """
    Initialise a fresh session dict for one user interaction.
    All tool outputs are stored here so app.py can read them after the run.
    """
    return {
        "query": query,
        "search_results": [],     # raw list[dict] from search_listings
        "selected_item": None,    # top result chosen for styling
        "wardrobe": wardrobe,     # injected by the agent, never sent to the LLM
        "outfit_suggestion": None,  # list[dict] or str from suggest_outfit
        "fit_card": None,           # dict from create_fit_card
        "error": None,              # set on early exit; nil on success
        "messages": [],             # full conversation transcript (for debugging)
    }


# ── Tool dispatcher ───────────────────────────────────────────────────────────

def _dispatch(tool_name: str, args: dict, session: dict) -> str:
    """
    Execute `tool_name` with `args`, write the result into `session`,
    and return a compact string observation for the LLM's next turn.

    The observation is intentionally trimmed — search_listings can return
    40 items, but the LLM only needs id/title/price/size to decide next steps.
    """
    if tool_name == "search_listings":
        result = search_listings(
            description=args.get("description", ""),
            size=args.get("size"),
            max_price=args.get("max_price"),
        )
        session["search_results"] = result

        if isinstance(result, str):
            # search_listings returned a failure message — propagate it
            session["error"] = result
            return result

        # Success: pin the top result and return a slim preview to save tokens
        session["selected_item"] = result[0]
        preview = [
            {k: item[k] for k in ("id", "title", "price", "size", "platform") if k in item}
            for item in result[:5]
        ]
        tail = f"\n...and {len(result) - 5} more." if len(result) > 5 else ""
        return json.dumps(preview, indent=2) + tail

    if tool_name == "suggest_outfit":
        # Resolve the item by the ID the LLM chose; fall back to the stored top result
        item_id = args.get("item_id")
        if item_id and isinstance(session["search_results"], list):
            matched = next(
                (r for r in session["search_results"] if r.get("id") == item_id),
                session["selected_item"],
            )
            session["selected_item"] = matched

        # Wardrobe is injected here — the LLM never sees or constructs it
        result = suggest_outfit(session["selected_item"], session["wardrobe"])
        session["outfit_suggestion"] = result

        return json.dumps(result, indent=2) if isinstance(result, list) else result

    if tool_name == "create_fit_card":
        result = create_fit_card(session["outfit_suggestion"], session["selected_item"])
        session["fit_card"] = result
        return json.dumps(result, indent=2)

    return f"Unknown tool '{tool_name}' — skipped."


# ── ReAct loop ────────────────────────────────────────────────────────────────

def run_agent(query: str, wardrobe: dict) -> dict:
    """
    ReAct agent entry point.

    Each iteration of the loop is one Thought-Action-Observation cycle:
      - The LLM receives the full message history and decides which tool to call.
      - The agent dispatches the call and appends the observation.
      - If the LLM produces a final answer (no tool_calls), the loop ends.

    Args:
        query:    Natural language user request.
        wardrobe: User's wardrobe dict — injected at suggest_outfit time.

    Returns:
        The session dict. Check session["error"] first; if set, the pipeline
        halted at Tool 1 and outfit_suggestion / fit_card will be None.
    """
    session = _new_session(query, wardrobe)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        session["error"] = "GROQ_API_KEY not set — add it to a .env file."
        return session

    client = Groq(api_key=api_key)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": query},
    ]

    # Safety cap: 3 tool calls + reasoning turns + buffer
    MAX_TURNS = 10

    for _ in range(MAX_TURNS):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.2,    # low temperature keeps tool-call decisions stable
            max_tokens=1024,
        )

        assistant_msg = response.choices[0].message

        # Serialise the assistant turn as a plain dict for the next API call
        # (the SDK object is not directly re-serialisable by all versions)
        msg_dict: dict = {"role": "assistant", "content": assistant_msg.content}
        if assistant_msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,   # already a JSON string
                    },
                }
                for tc in assistant_msg.tool_calls
            ]
        messages.append(msg_dict)

        # No tool calls → the LLM has reached its final answer; exit the loop
        if not assistant_msg.tool_calls:
            break

        # ── Execute every tool call the LLM requested this turn ──────────────
        for tool_call in assistant_msg.tool_calls:
            tool_name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                args = {}

            observation = _dispatch(tool_name, args, session)

            # Append the observation so the LLM can read it on the next turn
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": observation,
            })

        # If search_listings set an error, stop before calling more tools
        if session["error"]:
            break

    session["messages"] = messages
    return session


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from utils.data_loader import get_example_wardrobe, get_empty_wardrobe

    print("=== Happy path: graphic tee, populated wardrobe ===\n")
    session = run_agent(
        query="looking for a vintage graphic tee under $30",
        wardrobe=get_example_wardrobe(),
    )
    if session["error"]:
        print(f"Error: {session['error']}")
    else:
        print(f"Found:   {session['selected_item']['title']}")
        print(f"Outfit:  {session['outfit_suggestion']}")
        print(f"Caption: {session['fit_card']}")

    print("\n\n=== No-results path ===\n")
    session2 = run_agent(
        query="designer ballgown size XXS under $5",
        wardrobe=get_example_wardrobe(),
    )
    print(f"Error: {session2['error']}")

