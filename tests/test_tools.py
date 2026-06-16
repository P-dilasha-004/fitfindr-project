import json
import pytest
from unittest.mock import MagicMock, patch
from tools import search_listings, suggest_outfit


# Mock Database for Deterministic Testing 
MOCK_LISTINGS = [
    {
        "id": "1", "title": "Vintage Graphic Tee", "description": "Cool 90s shirt",
        "category": "tops", "style_tags": ["vintage", "graphic", "90s"],
        "size": "M", "price": 25.0, "colors": ["black"], "brand": "Fruit of the Loom"
    },
    {
        "id": "2", "title": "Levi Blue Jeans", "description": "Baggy fit vintage denim",
        "category": "bottoms", "style_tags": ["vintage", "denim"],
        "size": "S/M", "price": 45.0, "colors": ["blue"], "brand": "Levi"
    },
    {
        "id": "3", "title": "Chunky Knit Sweater", "description": "Warm winter wear",
        "category": "tops", "style_tags": ["knit", "chunky", "winter"],
        "size": "L", "price": 30.0, "colors": ["white"], "brand": None
    },
    {
        # Bare-minimum listing to test missing keys and null arrays
        "id": "4", "title": "Plain White Tee", "price": 15.0, "size": "S"
    }
]

##################################
# Search_listings 
##################################

# Happy Path

@patch('tools.load_listings', return_value=MOCK_LISTINGS)
def test_happy_path_scoring_and_sorting(mock_load):
    """Verifies items are scored properly and sorted with the highest score first."""
    # "vintage" matches Item 1 and 2. "tee" matches Item 1 and 4.
    # Item 1 scores 2. Item 2 and 4 score 1.
    results = search_listings(description="vintage tee")
    
    assert isinstance(results, list)
    assert len(results) == 3  # Corrected to 3
    assert results[0]["id"] == "1"  # Item 1 has the highest score, must be first

@patch('tools.load_listings', return_value=MOCK_LISTINGS)
def test_stop_words_ignored(mock_load):
    """Ensures stop words do not artificially inflate scores or trigger matches."""
    # "I am looking for a" are all stop words. "Sweater" is the only valid token.
    results = search_listings(description="I am looking for a sweater")
    
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["id"] == "3"

@patch('tools.load_listings', return_value=MOCK_LISTINGS)
def test_missing_fields_no_crash(mock_load):
    """Ensures the _score function doesn't crash on dictionaries missing optional keys."""
    # "plain" matches Item 4. "white" matches Item 4 and Item 3 (in colors array).
    # Item 4 scores 2. Item 3 scores 1.
    results = search_listings(description="plain white")
    
    assert isinstance(results, list)
    assert len(results) == 2  # Corrected to 2
    assert results[0]["id"] == "4"  # Item 4 has highest score, must be first


# Hard Filters (Price and Size) 

@patch('tools.load_listings', return_value=MOCK_LISTINGS)
def test_price_filter_success(mock_load):
    """Verifies max_price strictly drops items above the threshold."""
    results = search_listings(description="vintage", max_price=30.0)
    
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["id"] == "1" # Item 2 is $45, should be dropped

@patch('tools.load_listings', return_value=MOCK_LISTINGS)
def test_size_filter_substring_success(mock_load):
    """Verifies size filter handles case-insensitive substring matches."""
    # "M" should catch both "M" (Item 1) and "S/M" (Item 2)
    results = search_listings(description="vintage", size="m")
    
    assert isinstance(results, list)
    assert len(results) == 2
    ids = [r["id"] for r in results]
    assert "1" in ids
    assert "2" in ids


# Fallback Strings (Failure Modes)

@patch('tools.load_listings', return_value=MOCK_LISTINGS)
def test_failure_price_too_low(mock_load):
    """Verifies exact string output when nothing in the DB matches the budget."""
    result = search_listings(description="tee", max_price=5.0)
    
    assert isinstance(result, str)
    assert "No listings found under $5.00" in result
    assert "lowest-priced item in the catalogue is $15.00" in result

@patch('tools.load_listings', return_value=MOCK_LISTINGS)
def test_failure_size_not_found(mock_load):
    """Verifies string output when items pass price, but fail the size filter."""
    result = search_listings(description="tee", size="XL", max_price=50.0)
    
    assert isinstance(result, str)
    assert "No listings in size 'XL' under $50.00" in result
    assert "There are 4 items under $50.00" in result

@patch('tools.load_listings', return_value=MOCK_LISTINGS)
def test_failure_no_keyword_overlap(mock_load):
    """Verifies string output when filters pass but text scores are zero."""
    result = search_listings(description="leather jacket", size="M")
    
    assert isinstance(result, str)
    # The dataset has 2 items that are size "M" or "S/M" (Items 1 and 2)
    assert "Found 2 listing(s) in size 'M'" in result
    assert '"jacket", "leather"' in result

##################################
# Suggest_outfit
##################################

# ── Shared fixtures ───────────────────────────────────────────────────────────

# A complete new_item listing dict (the thrifted find to style)
NEW_ITEM = {
    "id": "lst_038",
    "title": "Denim Vest — Medium Wash, Studded",
    "price": 27.0,
    "size": "M",
    
    "condition": "good",
    "colors": ["medium blue"],
    "style_tags": ["grunge", "vintage", "denim"],
}

# Three wardrobe items the "user" owns
WARDROBE_ITEMS = [
    {
        "id": "w_001", "name": "Baggy straight-leg jeans", "category": "bottoms",
        "colors": ["dark blue"], "style_tags": ["denim"], "notes": "High-waisted",
    },
    {
        "id": "w_002", "name": "White ribbed tank top", "category": "tops",
        "colors": ["white"], "style_tags": ["basics"], "notes": None,
    },
    {
        "id": "w_003", "name": "Black combat boots", "category": "shoes",
        "colors": ["black"], "style_tags": ["grunge"], "notes": "Lace-up",
    },
]

# The correct wardrobe input shape — {"items": [...]}
WARDROBE_WITH_ITEMS = {"items": WARDROBE_ITEMS}
EMPTY_WARDROBE = {"items": []}

# Full wardrobe_schema.json wrapper — simulates accidentally passing
# load_wardrobe_schema() instead of get_example_wardrobe()
SCHEMA_WRAPPER = {
    "_description": "wardrobe schema file",
    "schema": {"id": "str", "name": "str"},
    "example_wardrobe": WARDROBE_WITH_ITEMS,   # {"items": [...]}
    "empty_wardrobe": {"items": []},
}

# What the LLM returns in Case 2 — a JSON array of two wardrobe items
VALID_LLM_JSON = json.dumps([WARDROBE_ITEMS[0], WARDROBE_ITEMS[2]])


def _mock_client(content: str) -> MagicMock:
    """
    Return a mock Groq client whose chat.completions.create() returns a
    response object whose .choices[0].message.content equals `content`.
    Patch tools._get_groq_client to return this instead of making real API calls.
    """
    mock_response = MagicMock()
    mock_response.choices[0].message.content = content
    client = MagicMock()
    client.chat.completions.create.return_value = mock_response
    return client


# ── Case 1: empty wardrobe — returns a plain string ───────────────────────────

@patch('tools._get_groq_client')
def test_case1_empty_items_list_returns_str(mock_get_client):
    """{"items": []} triggers Case 1 — function returns a non-empty plain string."""
    mock_get_client.return_value = _mock_client("Style with jeans and sneakers.")
    result = suggest_outfit(NEW_ITEM, EMPTY_WARDROBE)
    assert isinstance(result, str)
    assert result == "Style with jeans and sneakers."


@patch('tools._get_groq_client')
def test_case1_missing_items_key_defaults_to_empty(mock_get_client):
    """
    A wardrobe dict with neither 'items' nor 'example_wardrobe' key defaults to
    wardrobe_items=[] and triggers Case 1 rather than crashing.
    """
    mock_get_client.return_value = _mock_client("Here are some tips.")
    result = suggest_outfit(NEW_ITEM, {"notes": "random key"})
    assert isinstance(result, str)


@patch('tools._get_groq_client')
def test_case1_prompt_contains_item_title(mock_get_client):
    """The Case 1 prompt must include the new item's title so the LLM has context."""
    mock_get_client.return_value = _mock_client("Pair with dark jeans.")
    suggest_outfit(NEW_ITEM, EMPTY_WARDROBE)

    call_args = mock_get_client.return_value.chat.completions.create.call_args
    user_prompt = call_args.kwargs["messages"][1]["content"]
    assert "Denim Vest" in user_prompt


@patch('tools._get_groq_client')
def test_case1_temperature_is_0_7(mock_get_client):
    """Case 1 uses temperature=0.7 for creative, varied outfit advice."""
    mock_get_client.return_value = _mock_client("Pair with dark jeans.")
    suggest_outfit(NEW_ITEM, EMPTY_WARDROBE)

    call_args = mock_get_client.return_value.chat.completions.create.call_args
    assert call_args.kwargs["temperature"] == 0.7


@patch('tools._get_groq_client')
def test_case1_bare_new_item_no_crash(mock_get_client):
    """
    item_summary construction must not raise when optional fields (colors,
    style_tags, condition, size) are absent from new_item.
    """
    bare_item = {"title": "Mystery Jacket", "price": 20.0}
    mock_get_client.return_value = _mock_client("Style with anything.")
    result = suggest_outfit(bare_item, EMPTY_WARDROBE)
    assert isinstance(result, str)


@patch('tools._get_groq_client')
def test_case1_completely_empty_new_item_no_crash(mock_get_client):
    """An entirely empty new_item {} must not crash — all fields fall back gracefully."""
    mock_get_client.return_value = _mock_client("Some advice.")
    result = suggest_outfit({}, EMPTY_WARDROBE)
    assert isinstance(result, str)


# ── Case 2: populated wardrobe — returns list[dict] ──────────────────────────

@patch('tools._get_groq_client')
def test_case2_populated_wardrobe_returns_list(mock_get_client):
    """{"items": [...]} triggers Case 2 — function returns a list, not a string."""
    mock_get_client.return_value = _mock_client(VALID_LLM_JSON)
    result = suggest_outfit(NEW_ITEM, WARDROBE_WITH_ITEMS)
    assert isinstance(result, list)


@patch('tools._get_groq_client')
def test_case2_every_element_is_a_dict(mock_get_client):
    """Every element in the returned list must be a dict (not a string or None)."""
    mock_get_client.return_value = _mock_client(VALID_LLM_JSON)
    result = suggest_outfit(NEW_ITEM, WARDROBE_WITH_ITEMS)
    assert all(isinstance(item, dict) for item in result)


@patch('tools._get_groq_client')
def test_case2_returned_ids_are_subset_of_wardrobe(mock_get_client):
    """
    Every item ID in the result must exist in the original wardrobe —
    the LLM must not invent items that weren't provided.
    """
    mock_get_client.return_value = _mock_client(VALID_LLM_JSON)
    result = suggest_outfit(NEW_ITEM, WARDROBE_WITH_ITEMS)

    valid_ids = {item["id"] for item in WARDROBE_ITEMS}
    returned_ids = {item["id"] for item in result}
    assert returned_ids.issubset(valid_ids)


@patch('tools._get_groq_client')
def test_case2_wardrobe_serialised_as_json_in_prompt(mock_get_client):
    """
    Case 2 must send the wardrobe as a JSON string in the prompt (not as
    human-readable prose lines) so the LLM can echo items verbatim.
    """
    mock_get_client.return_value = _mock_client(VALID_LLM_JSON)
    suggest_outfit(NEW_ITEM, WARDROBE_WITH_ITEMS)

    call_args = mock_get_client.return_value.chat.completions.create.call_args
    user_prompt = call_args.kwargs["messages"][1]["content"]
    # JSON keys from the wardrobe items should appear literally in the prompt
    assert '"id"' in user_prompt
    assert "w_001" in user_prompt


@patch('tools._get_groq_client')
def test_case2_temperature_is_0_2(mock_get_client):
    """Case 2 uses temperature=0.2 to keep JSON output deterministic and format-safe."""
    mock_get_client.return_value = _mock_client(VALID_LLM_JSON)
    suggest_outfit(NEW_ITEM, WARDROBE_WITH_ITEMS)

    call_args = mock_get_client.return_value.chat.completions.create.call_args
    assert call_args.kwargs["temperature"] == 0.2


@patch('tools._get_groq_client')
def test_case2_bracket_extraction_recovers_array_from_prose(mock_get_client):
    """
    Bracket extraction must recover the JSON array even if the LLM adds surrounding
    prose (e.g. "Sure, here are the items: [...] Hope that helps!").
    """
    polluted = "Sure, here are the items: " + VALID_LLM_JSON + " Hope that helps!"
    mock_get_client.return_value = _mock_client(polluted)
    result = suggest_outfit(NEW_ITEM, WARDROBE_WITH_ITEMS)
    assert isinstance(result, list)
    assert len(result) == 2


@patch('tools._get_groq_client')
def test_case2_empty_json_array_returns_empty_list(mock_get_client):
    """
    LLM returning [] is valid JSON — the function returns an empty list,
    not an error string. An empty selection is a legitimate (if unhelpful) answer.
    """
    mock_get_client.return_value = _mock_client("[]")
    result = suggest_outfit(NEW_ITEM, WARDROBE_WITH_ITEMS)
    assert isinstance(result, list)
    assert result == []


# ── Case 2: failure modes — bad LLM output returns an error string ────────────

@patch('tools._get_groq_client')
def test_case2_unparseable_output_returns_error_string(mock_get_client):
    """
    Completely unparseable LLM output (no JSON at all) must return an informative
    error string — the pipeline must not raise JSONDecodeError.
    """
    mock_get_client.return_value = _mock_client("Sorry, I cannot help with that.")
    result = suggest_outfit(NEW_ITEM, WARDROBE_WITH_ITEMS)
    assert isinstance(result, str)
    assert "could not parse" in result


@patch('tools._get_groq_client')
def test_case2_json_object_not_list_returns_error_string(mock_get_client):
    """
    Valid JSON but wrong type (object instead of array) must return an error
    string — the type check must fire before the result reaches create_fit_card.
    """
    mock_get_client.return_value = _mock_client('{"name": "Baggy jeans"}')
    result = suggest_outfit(NEW_ITEM, WARDROBE_WITH_ITEMS)
    assert isinstance(result, str)
    assert "expected a list" in result


@patch('tools._get_groq_client')
def test_case2_json_number_not_list_returns_error_string(mock_get_client):
    """JSON scalar (42) passes json.loads() but fails the isinstance(list) check."""
    mock_get_client.return_value = _mock_client("42")
    result = suggest_outfit(NEW_ITEM, WARDROBE_WITH_ITEMS)
    assert isinstance(result, str)
    assert "expected a list" in result


# ── Wardrobe extraction edge cases ────────────────────────────────────────────

@patch('tools._get_groq_client')
def test_schema_wrapper_triggers_case2_not_case1(mock_get_client):
    """
    Regression test: passing the full wardrobe_schema.json wrapper (which has no
    top-level 'items' key) must NOT fall into Case 1. The function must unwrap
    'example_wardrobe' and trigger Case 2, returning list[dict].
    """
    mock_get_client.return_value = _mock_client(VALID_LLM_JSON)
    result = suggest_outfit(NEW_ITEM, SCHEMA_WRAPPER)
    assert isinstance(result, list), (
        "Regression: schema wrapper triggered Case 1 (str) instead of Case 2 (list). "
        "The wardrobe extraction is no longer unwrapping 'example_wardrobe'."
    )


@patch('tools._get_groq_client')
def test_schema_wrapper_empty_example_wardrobe_triggers_case1(mock_get_client):
    """
    Schema wrapper where example_wardrobe.items is [] → wardrobe_items resolves
    to [] → Case 1 fires and returns a str (no clothes to work with).
    """
    empty_wrapper = {
        "example_wardrobe": {"items": []},
        "empty_wardrobe": {"items": []},
    }
    mock_get_client.return_value = _mock_client("No clothes yet, here are some tips.")
    result = suggest_outfit(NEW_ITEM, empty_wrapper)
    assert isinstance(result, str)


@patch('tools._get_groq_client')
def test_items_key_takes_priority_over_example_wardrobe(mock_get_client):
    """
    When both 'items' and 'example_wardrobe' are present, 'items' is used.
    Here 'items' is populated, so Case 2 must fire and return list[dict].
    """
    both_keys = {
        "items": WARDROBE_ITEMS,           # populated — should win
        "example_wardrobe": {"items": []}, # empty — should be ignored
    }
    mock_get_client.return_value = _mock_client(VALID_LLM_JSON)
    result = suggest_outfit(NEW_ITEM, both_keys)
    assert isinstance(result, list)


@patch('tools._get_groq_client')
def test_unknown_wardrobe_keys_default_to_case1(mock_get_client):
    """
    A wardrobe dict with unrecognised keys ('clothes', 'owner') has no 'items'
    or 'example_wardrobe', so wardrobe_items defaults to [] and Case 1 fires.
    """
    strange_wardrobe = {"clothes": WARDROBE_ITEMS, "owner": "Alice"}
    mock_get_client.return_value = _mock_client("General styling advice here.")
    result = suggest_outfit(NEW_ITEM, strange_wardrobe)
    assert isinstance(result, str)


##################################
# create_fit_card
##################################

from tools import create_fit_card

# ── Fixtures ──────────────────────────────────────────────────────────────────

FC_NEW_ITEM = {
    "id": "lst_038",
    "title": "Denim Vest — Medium Wash, Studded",
    "price": 27.0,
    "platform": "depop",
    "condition": "good",
    "colors": ["medium blue"],
    "style_tags": ["grunge", "vintage"],
}

FC_OUTFIT = [
    {"id": "w_001", "name": "Baggy straight-leg jeans", "colors": ["dark blue", "indigo"]},
    {"id": "w_002", "name": "White ribbed tank top", "colors": ["white"]},
    {"id": "w_003", "name": "Black combat boots", "colors": ["black"]},
]

VALID_FIT_CARD_JSON = json.dumps({
    "caption": "Just scored this Denim Vest on depop for $27.00 — pure grunge magic.",
    "hashtags": ["#thrifted", "#depop", "#grunge", "#ootd"],
})


# ── Immediate fallback: no LLM call ──────────────────────────────────────────

@patch('tools._get_groq_client')
def test_fc_fallback_empty_list_bypasses_llm(mock_get_client):
    """Empty outfit list triggers the pre-LLM guard — Groq client never called."""
    result = create_fit_card([], FC_NEW_ITEM)
    mock_get_client.assert_not_called()
    assert isinstance(result, dict)


@patch('tools._get_groq_client')
def test_fc_fallback_string_input_bypasses_llm(mock_get_client):
    """A string outfit (error leaked from suggest_outfit) triggers the guard."""
    result = create_fit_card("Some styling advice paragraph", FC_NEW_ITEM)
    mock_get_client.assert_not_called()
    assert isinstance(result, dict)


@patch('tools._get_groq_client')
def test_fc_fallback_none_input_bypasses_llm(mock_get_client):
    """None is not a list — guard fires, LLM not called."""
    result = create_fit_card(None, FC_NEW_ITEM)
    mock_get_client.assert_not_called()
    assert isinstance(result, dict)


@patch('tools._get_groq_client')
def test_fc_fallback_dict_input_bypasses_llm(mock_get_client):
    """A bare dict (not wrapped in a list) triggers the guard."""
    result = create_fit_card({"name": "jeans"}, FC_NEW_ITEM)
    mock_get_client.assert_not_called()
    assert isinstance(result, dict)


def test_fc_fallback_caption_contains_item_title():
    """Fallback caption must include the new item's title so it is informative."""
    result = create_fit_card([], FC_NEW_ITEM)
    assert "Denim Vest" in result["caption"]


def test_fc_fallback_uses_thrifted_find_when_title_missing():
    """When new_item has no 'title', fallback caption uses 'this thrifted find'."""
    result = create_fit_card([], {})
    assert "thrifted find" in result["caption"]


def test_fc_fallback_always_has_required_keys():
    """Fallback dict must always expose both 'caption' and 'hashtags'."""
    result = create_fit_card([], FC_NEW_ITEM)
    assert "caption" in result
    assert "hashtags" in result


def test_fc_fallback_caption_is_nonempty_str():
    """Fallback caption must be a non-empty string, not a list or None."""
    result = create_fit_card([], FC_NEW_ITEM)
    assert isinstance(result["caption"], str)
    assert len(result["caption"]) > 0


def test_fc_fallback_hashtags_is_list_of_strings():
    """Fallback hashtags must be a non-empty list where every element is a string."""
    result = create_fit_card([], FC_NEW_ITEM)
    assert isinstance(result["hashtags"], list)
    assert len(result["hashtags"]) > 0
    assert all(isinstance(h, str) for h in result["hashtags"])


# ── Happy path: LLM returns well-formed JSON ──────────────────────────────────

@patch('tools._get_groq_client')
def test_fc_happy_path_returns_dict(mock_get_client):
    """Non-empty outfit list → LLM called → function returns a dict."""
    mock_get_client.return_value = _mock_client(VALID_FIT_CARD_JSON)
    result = create_fit_card(FC_OUTFIT, FC_NEW_ITEM)
    assert isinstance(result, dict)


@patch('tools._get_groq_client')
def test_fc_happy_path_caption_is_str(mock_get_client):
    """Returned caption must be a non-empty string."""
    mock_get_client.return_value = _mock_client(VALID_FIT_CARD_JSON)
    result = create_fit_card(FC_OUTFIT, FC_NEW_ITEM)
    assert isinstance(result["caption"], str)
    assert len(result["caption"]) > 0


@patch('tools._get_groq_client')
def test_fc_happy_path_hashtags_is_list_of_strings(mock_get_client):
    """Returned hashtags must be a list where every element is a string."""
    mock_get_client.return_value = _mock_client(VALID_FIT_CARD_JSON)
    result = create_fit_card(FC_OUTFIT, FC_NEW_ITEM)
    assert isinstance(result["hashtags"], list)
    assert all(isinstance(h, str) for h in result["hashtags"])


@patch('tools._get_groq_client')
def test_fc_temperature_is_0_9(mock_get_client):
    """
    Temperature must be 0.9 so repeated calls produce varied captions
    rather than the same sentence every time.
    """
    mock_get_client.return_value = _mock_client(VALID_FIT_CARD_JSON)
    create_fit_card(FC_OUTFIT, FC_NEW_ITEM)
    call_args = mock_get_client.return_value.chat.completions.create.call_args
    assert call_args.kwargs["temperature"] == 0.9


@patch('tools._get_groq_client')
def test_fc_prompt_contains_item_title(mock_get_client):
    """Prompt must include the item title so the LLM can name it in the caption."""
    mock_get_client.return_value = _mock_client(VALID_FIT_CARD_JSON)
    create_fit_card(FC_OUTFIT, FC_NEW_ITEM)
    prompt = mock_get_client.return_value.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "Denim Vest" in prompt


@patch('tools._get_groq_client')
def test_fc_prompt_contains_price(mock_get_client):
    """Prompt must include the formatted price so the LLM can mention it."""
    mock_get_client.return_value = _mock_client(VALID_FIT_CARD_JSON)
    create_fit_card(FC_OUTFIT, FC_NEW_ITEM)
    prompt = mock_get_client.return_value.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "$27.00" in prompt


@patch('tools._get_groq_client')
def test_fc_prompt_contains_platform(mock_get_client):
    """Prompt must include the platform name so the LLM can mention it."""
    mock_get_client.return_value = _mock_client(VALID_FIT_CARD_JSON)
    create_fit_card(FC_OUTFIT, FC_NEW_ITEM)
    prompt = mock_get_client.return_value.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "depop" in prompt


@patch('tools._get_groq_client')
def test_fc_prompt_contains_all_outfit_item_names(mock_get_client):
    """Every wardrobe item name must appear in the prompt so the LLM can cite them."""
    mock_get_client.return_value = _mock_client(VALID_FIT_CARD_JSON)
    create_fit_card(FC_OUTFIT, FC_NEW_ITEM)
    prompt = mock_get_client.return_value.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    for piece in FC_OUTFIT:
        assert piece["name"] in prompt


@patch('tools._get_groq_client')
def test_fc_brace_extraction_recovers_from_surrounding_prose(mock_get_client):
    """
    If the LLM wraps the JSON object in surrounding prose, brace extraction
    must recover the object and return a valid dict.
    """
    polluted = "Here's your fit card: " + VALID_FIT_CARD_JSON + " Let me know!"
    mock_get_client.return_value = _mock_client(polluted)
    result = create_fit_card(FC_OUTFIT, FC_NEW_ITEM)
    assert isinstance(result, dict)
    assert "caption" in result
    assert "hashtags" in result


# ── LLM output failures: all must return fallback dict, never raise ───────────

@patch('tools._get_groq_client')
def test_fc_json_decode_error_returns_fallback(mock_get_client):
    """Completely unparseable LLM output must not raise — except block returns fallback."""
    mock_get_client.return_value = _mock_client("Not JSON at all, sorry!")
    result = create_fit_card(FC_OUTFIT, FC_NEW_ITEM)
    assert isinstance(result, dict)
    assert "caption" in result
    assert "hashtags" in result


@patch('tools._get_groq_client')
def test_fc_missing_caption_key_returns_fallback(mock_get_client):
    """
    Valid JSON missing 'caption' raises ValueError in the validation block —
    the except block catches it and returns the fallback dict.
    """
    mock_get_client.return_value = _mock_client(json.dumps({"hashtags": ["#thrifted"]}))
    result = create_fit_card(FC_OUTFIT, FC_NEW_ITEM)
    assert isinstance(result, dict)
    assert "caption" in result


@patch('tools._get_groq_client')
def test_fc_missing_hashtags_key_returns_fallback(mock_get_client):
    """Valid JSON missing 'hashtags' raises ValueError → fallback returned."""
    mock_get_client.return_value = _mock_client(json.dumps({"caption": "Great outfit!"}))
    result = create_fit_card(FC_OUTFIT, FC_NEW_ITEM)
    assert isinstance(result, dict)
    assert "hashtags" in result


@patch('tools._get_groq_client')
def test_fc_caption_wrong_type_returns_fallback(mock_get_client):
    """caption must be a str — an integer fails isinstance check → fallback."""
    mock_get_client.return_value = _mock_client(json.dumps({"caption": 42, "hashtags": ["#thrifted"]}))
    result = create_fit_card(FC_OUTFIT, FC_NEW_ITEM)
    assert isinstance(result["caption"], str)


@patch('tools._get_groq_client')
def test_fc_hashtags_wrong_type_returns_fallback(mock_get_client):
    """hashtags must be a list — a plain string fails isinstance check → fallback."""
    mock_get_client.return_value = _mock_client(json.dumps({"caption": "Great!", "hashtags": "#thrifted #ootd"}))
    result = create_fit_card(FC_OUTFIT, FC_NEW_ITEM)
    assert isinstance(result["hashtags"], list)


@patch('tools._get_groq_client')
def test_fc_groq_client_exception_returns_fallback(mock_get_client):
    """
    If _get_groq_client() itself raises (e.g. missing API key, network error),
    the except block catches it and returns the fallback dict — pipeline never crashes.
    """
    mock_get_client.side_effect = RuntimeError("Simulated network failure")
    result = create_fit_card(FC_OUTFIT, FC_NEW_ITEM)
    assert isinstance(result, dict)
    assert "caption" in result
    assert "hashtags" in result


# ── new_item missing fields in the LLM path ──────────────────────────────────

@patch('tools._get_groq_client')
def test_fc_missing_price_uses_steal_string(mock_get_client):
    """When new_item has no 'price', prompt must use 'a steal' instead of crashing."""
    mock_get_client.return_value = _mock_client(VALID_FIT_CARD_JSON)
    create_fit_card(FC_OUTFIT, {"title": "Mystery Jacket", "platform": "depop"})
    prompt = mock_get_client.return_value.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "a steal" in prompt


@patch('tools._get_groq_client')
def test_fc_missing_platform_uses_thrift_app(mock_get_client):
    """When new_item has no 'platform', prompt must fall back to 'a thrift app'."""
    mock_get_client.return_value = _mock_client(VALID_FIT_CARD_JSON)
    create_fit_card(FC_OUTFIT, {"title": "Mystery Jacket", "price": 15.0})
    prompt = mock_get_client.return_value.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "a thrift app" in prompt


@patch('tools._get_groq_client')
def test_fc_missing_title_uses_thrifted_find(mock_get_client):
    """When new_item has no 'title', prompt must fall back to 'thrifted find'."""
    mock_get_client.return_value = _mock_client(VALID_FIT_CARD_JSON)
    create_fit_card(FC_OUTFIT, {"price": 20.0, "platform": "poshmark"})
    prompt = mock_get_client.return_value.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "thrifted find" in prompt


# ── Outfit line formatting edge cases ─────────────────────────────────────────

@patch('tools._get_groq_client')
def test_fc_outfit_item_missing_name_uses_wardrobe_piece(mock_get_client):
    """
    An outfit item with no 'name' key must not crash — formatting loop falls
    back to 'wardrobe piece' and that string appears in the prompt.
    """
    mock_get_client.return_value = _mock_client(VALID_FIT_CARD_JSON)
    create_fit_card([{"id": "w_099", "colors": ["red"]}], FC_NEW_ITEM)
    prompt = mock_get_client.return_value.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "wardrobe piece" in prompt


@patch('tools._get_groq_client')
def test_fc_outfit_item_missing_colors_omits_parens(mock_get_client):
    """
    An outfit item with no 'colors' key must not produce a trailing '()' in
    the prompt — the color annotation is skipped entirely.
    """
    mock_get_client.return_value = _mock_client(VALID_FIT_CARD_JSON)
    create_fit_card([{"id": "w_100", "name": "Plain Tee"}], FC_NEW_ITEM)
    prompt = mock_get_client.return_value.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "Plain Tee" in prompt
    assert "Plain Tee ()" not in prompt
