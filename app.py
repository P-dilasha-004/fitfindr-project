"""
app.py

Gradio interface for FitFindr. The layout and wiring are already set up —
your job is to fill in handle_query() so it calls run_agent() and maps
the session results to the three output panels.

Run with:
    python app.py

Then open the localhost URL shown in your terminal (usually http://localhost:7860,
but check your terminal — the port may differ).
"""

import gradio as gr

from agent import run_agent
from utils.data_loader import get_example_wardrobe, get_empty_wardrobe


# ── query handler ─────────────────────────────────────────────────────────────

def handle_query(user_query: str, wardrobe_choice: str) -> tuple[str, str, str]:
    """
    Called by Gradio when the user submits a query.

    Args:
        user_query:     The text the user typed into the search box.
        wardrobe_choice: Either "Example wardrobe" or "Empty wardrobe (new user)".

    Returns:
        A tuple of three strings:
            (listing_text, outfit_suggestion, fit_card)
        Each string maps to one of the three output panels in the UI.

    TODO:
        1. Guard against an empty query (return early with an error message).
        2. Select the wardrobe based on wardrobe_choice.
        3. Call run_agent() with the query and selected wardrobe.
        4. If session["error"] is set, return the error in the first panel
           and empty strings for the other two.
        5. Otherwise, format session["selected_item"] into a readable listing_text
           string and return it along with session["outfit_suggestion"] and
           session["fit_card"].
    """
    # Step 1: reject blank input before touching the agent
    if not user_query or not user_query.strip():
        return "Please enter a search query before submitting.", "", ""

    # Step 2: resolve wardrobe from the radio button selection
    wardrobe = (
        get_example_wardrobe()
        if wardrobe_choice == "Example wardrobe"
        else get_empty_wardrobe()
    )

    # Step 3: run the planning loop
    session = run_agent(query=user_query.strip(), wardrobe=wardrobe)

    # Step 4: surface Tool 1 failures (no matching listings) in the first panel
    if session["error"]:
        return session["error"], "", ""

    # Step 5: format the three output panels
    item = session["selected_item"]

    # Panel 1 — structured listing summary
    price = item.get("price")
    price_str = f"${price:.2f}" if isinstance(price, (int, float)) else "Price unknown"
    brand = item.get("brand") or "Unknown brand"
    colors = ", ".join(item.get("colors", [])) or "—"
    tags = ", ".join(item.get("style_tags", [])) or "—"

    listing_text = (
        f"{item.get('title', 'Listing')}\n"
        f"\n"
        f"Price:     {price_str}\n"
        f"Size:      {item.get('size', '—')}\n"
        f"Condition: {item.get('condition', '—')}\n"
        f"Brand:     {brand}\n"
        f"Colors:    {colors}\n"
        f"Tags:      {tags}\n"
        f"Platform:  {item.get('platform', '—')}\n"
        f"\n"
        f"{item.get('description', '')}"
    )

    # Panel 2 — outfit suggestion (str from Case 1, or list[dict] from Case 2)
    outfit_raw = session["outfit_suggestion"]
    if isinstance(outfit_raw, list):
        # Format the selected wardrobe items as a readable list
        lines = []
        for piece in outfit_raw:
            colors_str = ", ".join(piece.get("colors", []))
            line = f"• {piece.get('name', 'item')}"
            if colors_str:
                line += f"  ({colors_str})"
            notes = piece.get("notes")
            if notes:
                line += f"\n  {notes}"
            lines.append(line)
        outfit_text = "\n\n".join(lines)
    else:
        # Case 1: suggest_outfit returned a conversational styling advice string
        outfit_text = outfit_raw or ""

    # Panel 3 — fit card caption + hashtags (always a dict from create_fit_card)
    fit_card_raw = session["fit_card"]
    if isinstance(fit_card_raw, dict):
        caption = fit_card_raw.get("caption", "")
        hashtags = "  ".join(fit_card_raw.get("hashtags", []))
        fit_card_text = f"{caption}\n\n{hashtags}" if hashtags else caption
    else:
        fit_card_text = str(fit_card_raw) if fit_card_raw else ""

    return listing_text, outfit_text, fit_card_text


# ── interface ─────────────────────────────────────────────────────────────────

EXAMPLE_QUERIES = [
    "vintage graphic tee under $30",
    "90s track jacket in size M",
    "flowy midi skirt under $40",
    "black combat boots size 8",
    "designer ballgown size XXS under $5",   # deliberate no-results test
]

def build_interface():
    with gr.Blocks(title="FitFindr") as demo:
        gr.Markdown("""
# FitFindr 🛍️
Find secondhand pieces and get outfit ideas based on your wardrobe.
Describe what you're looking for — include size and price if you want to filter.
        """)

        with gr.Row():
            query_input = gr.Textbox(
                label="What are you looking for?",
                placeholder="e.g. vintage graphic tee under $30, size M",
                lines=2,
                scale=3,
            )
            wardrobe_choice = gr.Radio(
                choices=["Example wardrobe", "Empty wardrobe (new user)"],
                value="Example wardrobe",
                label="Wardrobe",
                scale=1,
            )

        submit_btn = gr.Button("Find it", variant="primary")

        with gr.Row():
            listing_output = gr.Textbox(
                label="🛍️ Top listing found",
                lines=8,
                interactive=False,
            )
            outfit_output = gr.Textbox(
                label="👗 Outfit idea",
                lines=8,
                interactive=False,
            )
            fitcard_output = gr.Textbox(
                label="✨ Your fit card",
                lines=8,
                interactive=False,
            )

        gr.Examples(
            examples=[[q, "Example wardrobe"] for q in EXAMPLE_QUERIES],
            inputs=[query_input, wardrobe_choice],
            label="Try these queries",
        )

        submit_btn.click(
            fn=handle_query,
            inputs=[query_input, wardrobe_choice],
            outputs=[listing_output, outfit_output, fitcard_output],
        )
        query_input.submit(
            fn=handle_query,
            inputs=[query_input, wardrobe_choice],
            outputs=[listing_output, outfit_output, fitcard_output],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
