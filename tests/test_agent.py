"""
tests/test_agent.py

Comprehensive tests for agent.py covering:
  A. _new_session()          — session initialisation
  B. _dispatch() search      — search_listings routing
  C. _dispatch() suggest     — suggest_outfit routing
  D. _dispatch() fit card    — create_fit_card routing
  E. _dispatch() unknown     — graceful unknown-tool handling
  F. run_agent() API key     — missing GROQ_API_KEY guard
  G. run_agent() happy path  — full 3-tool ReAct pipeline
  H. run_agent() search fail — early exit when Tool 1 returns an error
  I. run_agent() no tool call — LLM produces a final answer immediately
  J. run_agent() MAX_TURNS   — safety cap prevents infinite loops
  K. run_agent() bad JSON    — malformed tool arguments handled gracefully
  L. run_agent() messages    — conversation history structure
  M. TOOL_SCHEMAS constants  — schema contracts the LLM depends on

All Groq API calls are mocked. Tool functions are patched at the agent module
level (agent.search_listings, agent.suggest_outfit, agent.create_fit_card)
so no real HTTP requests or data files are needed.
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch, call

from agent import (
    _new_session,
    _dispatch,
    run_agent,
    TOOL_SCHEMAS,
    SYSTEM_PROMPT,
)


# ── Shared sample data ────────────────────────────────────────────────────────

SAMPLE_LISTING = {
    "id": "lst_001",
    "title": "Vintage Levi's 501 Jeans",
    "description": "Classic 501s in medium wash.",
    "category": "bottoms",
    "style_tags": ["vintage", "denim", "streetwear"],
    "size": "W30 L30",
    "condition": "good",
    "price": 38.0,
    "colors": ["blue", "indigo"],
    "brand": "Levi's",
    "platform": "depop",
}

SAMPLE_LISTING_2 = {
    "id": "lst_002",
    "title": "90s Harley Davidson Tee",
    "price": 25.0,
    "size": "M",
    "platform": "poshmark",
}

SAMPLE_WARDROBE = {
    "items": [
        {
            "id": "w_001",
            "name": "White ribbed tank top",
            "category": "tops",
            "colors": ["white"],
            "style_tags": ["basics"],
            "notes": "Goes with everything",
        },
        {
            "id": "w_002",
            "name": "Black canvas sneakers",
            "category": "shoes",
            "colors": ["black"],
            "style_tags": ["minimal"],
            "notes": None,
        },
    ]
}

EMPTY_WARDROBE = {"items": []}

SAMPLE_OUTFIT = [
    {"id": "w_001", "name": "White ribbed tank top", "colors": ["white"], "notes": "Goes with everything"},
    {"id": "w_002", "name": "Black canvas sneakers", "colors": ["black"], "notes": None},
]

SAMPLE_FIT_CARD = {
    "caption": "Scored these vintage Levi's for $38 on depop. Paired them with a white tank and black sneakers for an effortless look.",
    "hashtags": ["#thrifted", "#vintage", "#ootd", "#depopfind"],
}


# ── Mock builders ─────────────────────────────────────────────────────────────

def _make_tool_response(tool_name: str, args: dict, call_id: str = "tc_1") -> MagicMock:
    """Return a mock Groq response whose message requests one tool call."""
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = tool_name
    tc.function.arguments = json.dumps(args)

    response = MagicMock()
    response.choices[0].message.tool_calls = [tc]
    response.choices[0].message.content = None
    return response


def _make_final_response(content: str = "Here is your outfit!") -> MagicMock:
    """Return a mock Groq response with no tool calls (LLM final answer)."""
    response = MagicMock()
    response.choices[0].message.tool_calls = None
    response.choices[0].message.content = content
    return response


def _make_session(**overrides) -> dict:
    """Return a pre-populated session dict for _dispatch tests."""
    base = {
        "query": "vintage tee",
        "search_results": [SAMPLE_LISTING],
        "selected_item": SAMPLE_LISTING,
        "wardrobe": SAMPLE_WARDROBE,
        "outfit_suggestion": SAMPLE_OUTFIT,
        "fit_card": None,
        "error": None,
        "messages": [],
    }
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# A. _new_session
# ─────────────────────────────────────────────────────────────────────────────

class TestNewSession:
    def test_all_expected_keys_present(self):
        session = _new_session("a query", SAMPLE_WARDROBE)
        expected = {
            "query", "search_results", "selected_item", "wardrobe",
            "outfit_suggestion", "fit_card", "error", "messages",
        }
        assert set(session.keys()) == expected

    def test_does_not_have_parsed_key(self):
        # 'parsed' belonged to the old procedural pipeline and was removed
        session = _new_session("test", SAMPLE_WARDROBE)
        assert "parsed" not in session

    def test_query_stored_verbatim(self):
        session = _new_session("chunky knit under $40", SAMPLE_WARDROBE)
        assert session["query"] == "chunky knit under $40"

    def test_wardrobe_stored_verbatim(self):
        session = _new_session("test", SAMPLE_WARDROBE)
        assert session["wardrobe"] is SAMPLE_WARDROBE

    def test_search_results_starts_as_empty_list(self):
        session = _new_session("test", SAMPLE_WARDROBE)
        assert session["search_results"] == []

    def test_selected_item_starts_as_none(self):
        session = _new_session("test", SAMPLE_WARDROBE)
        assert session["selected_item"] is None

    def test_outfit_suggestion_starts_as_none(self):
        session = _new_session("test", SAMPLE_WARDROBE)
        assert session["outfit_suggestion"] is None

    def test_fit_card_starts_as_none(self):
        session = _new_session("test", SAMPLE_WARDROBE)
        assert session["fit_card"] is None

    def test_error_starts_as_none(self):
        session = _new_session("test", SAMPLE_WARDROBE)
        assert session["error"] is None

    def test_messages_starts_as_empty_list(self):
        session = _new_session("test", SAMPLE_WARDROBE)
        assert session["messages"] == []

    def test_two_sessions_are_independent_objects(self):
        s1 = _new_session("query one", SAMPLE_WARDROBE)
        s2 = _new_session("query two", EMPTY_WARDROBE)
        s1["search_results"].append(SAMPLE_LISTING)
        assert s2["search_results"] == [], "sessions share the same list"

    def test_empty_wardrobe_stored_correctly(self):
        session = _new_session("test", EMPTY_WARDROBE)
        assert session["wardrobe"] == {"items": []}


# ─────────────────────────────────────────────────────────────────────────────
# B. _dispatch — search_listings branch
# ─────────────────────────────────────────────────────────────────────────────

class TestDispatchSearch:
    @patch("agent.search_listings")
    def test_success_sets_search_results(self, mock_sl):
        mock_sl.return_value = [SAMPLE_LISTING]
        session = _make_session(search_results=[], selected_item=None)
        _dispatch("search_listings", {"description": "vintage jeans"}, session)
        assert session["search_results"] == [SAMPLE_LISTING]

    @patch("agent.search_listings")
    def test_success_pins_first_result_as_selected_item(self, mock_sl):
        mock_sl.return_value = [SAMPLE_LISTING, SAMPLE_LISTING_2]
        session = _make_session(search_results=[], selected_item=None)
        _dispatch("search_listings", {"description": "jeans"}, session)
        assert session["selected_item"] == SAMPLE_LISTING

    @patch("agent.search_listings")
    def test_success_observation_is_valid_json(self, mock_sl):
        mock_sl.return_value = [SAMPLE_LISTING]
        session = _make_session(search_results=[], selected_item=None)
        obs = _dispatch("search_listings", {"description": "jeans"}, session)
        parsed = json.loads(obs)
        assert isinstance(parsed, list)

    @patch("agent.search_listings")
    def test_success_preview_contains_slim_fields_only(self, mock_sl):
        mock_sl.return_value = [SAMPLE_LISTING]
        session = _make_session(search_results=[], selected_item=None)
        obs = _dispatch("search_listings", {"description": "jeans"}, session)
        item = json.loads(obs)[0]
        # Full-detail fields like description and style_tags should be stripped
        assert "id" in item
        assert "title" in item
        assert "price" in item
        assert "description" not in item
        assert "style_tags" not in item

    @patch("agent.search_listings")
    def test_success_preview_capped_at_5_items(self, mock_sl):
        ten_items = [{**SAMPLE_LISTING, "id": f"lst_{i:03d}"} for i in range(10)]
        mock_sl.return_value = ten_items
        session = _make_session(search_results=[], selected_item=None)
        obs = _dispatch("search_listings", {"description": "jeans"}, session)
        # Extract the JSON part before any trailing text
        json_part = obs[:obs.rfind("]") + 1]
        assert len(json.loads(json_part)) == 5

    @patch("agent.search_listings")
    def test_success_tail_message_when_more_than_5(self, mock_sl):
        ten_items = [{**SAMPLE_LISTING, "id": f"lst_{i:03d}"} for i in range(10)]
        mock_sl.return_value = ten_items
        session = _make_session(search_results=[], selected_item=None)
        obs = _dispatch("search_listings", {"description": "jeans"}, session)
        assert "and 5 more" in obs

    @patch("agent.search_listings")
    def test_success_no_tail_when_exactly_5_items(self, mock_sl):
        five_items = [{**SAMPLE_LISTING, "id": f"lst_{i:03d}"} for i in range(5)]
        mock_sl.return_value = five_items
        session = _make_session(search_results=[], selected_item=None)
        obs = _dispatch("search_listings", {"description": "jeans"}, session)
        assert "more" not in obs

    @patch("agent.search_listings")
    def test_success_no_tail_when_fewer_than_5_items(self, mock_sl):
        mock_sl.return_value = [SAMPLE_LISTING, SAMPLE_LISTING_2]
        session = _make_session(search_results=[], selected_item=None)
        obs = _dispatch("search_listings", {"description": "jeans"}, session)
        assert "more" not in obs

    @patch("agent.search_listings")
    def test_failure_string_sets_session_error(self, mock_sl):
        error_msg = "No listings found under $5.00."
        mock_sl.return_value = error_msg
        session = _make_session(search_results=[], selected_item=None, error=None)
        _dispatch("search_listings", {"description": "ballgown"}, session)
        assert session["error"] == error_msg

    @patch("agent.search_listings")
    def test_failure_string_returned_as_observation(self, mock_sl):
        error_msg = "No listings in size 'XXS'."
        mock_sl.return_value = error_msg
        session = _make_session(search_results=[], selected_item=None, error=None)
        obs = _dispatch("search_listings", {"description": "blazer"}, session)
        assert obs == error_msg

    @patch("agent.search_listings")
    def test_failure_selected_item_remains_none(self, mock_sl):
        mock_sl.return_value = "No match."
        session = _make_session(search_results=[], selected_item=None)
        _dispatch("search_listings", {"description": "gown"}, session)
        assert session["selected_item"] is None

    @patch("agent.search_listings")
    def test_description_forwarded_to_tool(self, mock_sl):
        mock_sl.return_value = [SAMPLE_LISTING]
        session = _make_session(search_results=[], selected_item=None)
        _dispatch("search_listings", {"description": "chunky knit sweater"}, session)
        mock_sl.assert_called_once_with(
            description="chunky knit sweater",
            size=None,
            max_price=None,
        )

    @patch("agent.search_listings")
    def test_size_forwarded_when_provided(self, mock_sl):
        mock_sl.return_value = [SAMPLE_LISTING]
        session = _make_session(search_results=[], selected_item=None)
        _dispatch("search_listings", {"description": "jeans", "size": "M"}, session)
        _, kwargs = mock_sl.call_args
        assert kwargs["size"] == "M"

    @patch("agent.search_listings")
    def test_max_price_forwarded_when_provided(self, mock_sl):
        mock_sl.return_value = [SAMPLE_LISTING]
        session = _make_session(search_results=[], selected_item=None)
        _dispatch("search_listings", {"description": "tee", "max_price": 30.0}, session)
        _, kwargs = mock_sl.call_args
        assert kwargs["max_price"] == 30.0

    @patch("agent.search_listings")
    def test_missing_description_defaults_to_empty_string(self, mock_sl):
        mock_sl.return_value = [SAMPLE_LISTING]
        session = _make_session(search_results=[], selected_item=None)
        _dispatch("search_listings", {}, session)  # no description key
        _, kwargs = mock_sl.call_args
        assert kwargs["description"] == ""

    @patch("agent.search_listings")
    def test_missing_optional_args_default_to_none(self, mock_sl):
        mock_sl.return_value = [SAMPLE_LISTING]
        session = _make_session(search_results=[], selected_item=None)
        _dispatch("search_listings", {"description": "tee"}, session)
        mock_sl.assert_called_once_with(description="tee", size=None, max_price=None)


# ─────────────────────────────────────────────────────────────────────────────
# C. _dispatch — suggest_outfit branch
# ─────────────────────────────────────────────────────────────────────────────

class TestDispatchSuggest:
    @patch("agent.suggest_outfit")
    def test_resolves_item_by_item_id(self, mock_so):
        mock_so.return_value = SAMPLE_OUTFIT
        session = _make_session(
            search_results=[SAMPLE_LISTING, SAMPLE_LISTING_2],
            selected_item=SAMPLE_LISTING,
        )
        _dispatch("suggest_outfit", {"item_id": "lst_002"}, session)
        assert session["selected_item"] == SAMPLE_LISTING_2

    @patch("agent.suggest_outfit")
    def test_falls_back_to_selected_item_when_id_not_found(self, mock_so):
        mock_so.return_value = SAMPLE_OUTFIT
        session = _make_session(
            search_results=[SAMPLE_LISTING],
            selected_item=SAMPLE_LISTING,
        )
        _dispatch("suggest_outfit", {"item_id": "nonexistent_id"}, session)
        assert session["selected_item"] == SAMPLE_LISTING

    @patch("agent.suggest_outfit")
    def test_keeps_selected_item_when_item_id_not_provided(self, mock_so):
        mock_so.return_value = SAMPLE_OUTFIT
        session = _make_session(selected_item=SAMPLE_LISTING)
        _dispatch("suggest_outfit", {}, session)
        assert session["selected_item"] == SAMPLE_LISTING

    @patch("agent.suggest_outfit")
    def test_updates_outfit_suggestion_with_list_result(self, mock_so):
        mock_so.return_value = SAMPLE_OUTFIT
        session = _make_session(outfit_suggestion=None)
        _dispatch("suggest_outfit", {"item_id": "lst_001"}, session)
        assert session["outfit_suggestion"] == SAMPLE_OUTFIT

    @patch("agent.suggest_outfit")
    def test_updates_outfit_suggestion_with_string_result(self, mock_so):
        advice = "Try pairing with wide-leg trousers and chunky sneakers."
        mock_so.return_value = advice
        session = _make_session(outfit_suggestion=None)
        _dispatch("suggest_outfit", {"item_id": "lst_001"}, session)
        assert session["outfit_suggestion"] == advice

    @patch("agent.suggest_outfit")
    def test_list_result_observation_is_valid_json(self, mock_so):
        mock_so.return_value = SAMPLE_OUTFIT
        session = _make_session()
        obs = _dispatch("suggest_outfit", {"item_id": "lst_001"}, session)
        parsed = json.loads(obs)
        assert isinstance(parsed, list)

    @patch("agent.suggest_outfit")
    def test_string_result_observation_is_raw_string(self, mock_so):
        advice = "Pair with wide-leg trousers."
        mock_so.return_value = advice
        session = _make_session()
        obs = _dispatch("suggest_outfit", {"item_id": "lst_001"}, session)
        assert obs == advice

    @patch("agent.suggest_outfit")
    def test_wardrobe_injected_from_session_not_from_args(self, mock_so):
        mock_so.return_value = SAMPLE_OUTFIT
        session = _make_session(wardrobe=SAMPLE_WARDROBE, selected_item=SAMPLE_LISTING)
        # LLM provides no wardrobe in its args — it should still be injected
        _dispatch("suggest_outfit", {"item_id": "lst_001"}, session)
        _, kwargs = mock_so.call_args
        assert kwargs == {} or mock_so.call_args[0][1] == SAMPLE_WARDROBE

    @patch("agent.suggest_outfit")
    def test_suggest_called_with_selected_item_and_wardrobe(self, mock_so):
        mock_so.return_value = SAMPLE_OUTFIT
        session = _make_session(
            selected_item=SAMPLE_LISTING,
            wardrobe=SAMPLE_WARDROBE,
            search_results=[SAMPLE_LISTING],
        )
        _dispatch("suggest_outfit", {"item_id": "lst_001"}, session)
        mock_so.assert_called_once_with(SAMPLE_LISTING, SAMPLE_WARDROBE)


# ─────────────────────────────────────────────────────────────────────────────
# D. _dispatch — create_fit_card branch
# ─────────────────────────────────────────────────────────────────────────────

class TestDispatchFitCard:
    @patch("agent.create_fit_card")
    def test_updates_fit_card_in_session(self, mock_fc):
        mock_fc.return_value = SAMPLE_FIT_CARD
        session = _make_session(fit_card=None)
        _dispatch("create_fit_card", {}, session)
        assert session["fit_card"] == SAMPLE_FIT_CARD

    @patch("agent.create_fit_card")
    def test_observation_is_valid_json(self, mock_fc):
        mock_fc.return_value = SAMPLE_FIT_CARD
        session = _make_session()
        obs = _dispatch("create_fit_card", {}, session)
        parsed = json.loads(obs)
        assert "caption" in parsed
        assert "hashtags" in parsed

    @patch("agent.create_fit_card")
    def test_observation_equals_json_dumps_of_fit_card(self, mock_fc):
        mock_fc.return_value = SAMPLE_FIT_CARD
        session = _make_session()
        obs = _dispatch("create_fit_card", {}, session)
        assert json.loads(obs) == SAMPLE_FIT_CARD

    @patch("agent.create_fit_card")
    def test_called_with_outfit_suggestion_and_selected_item(self, mock_fc):
        mock_fc.return_value = SAMPLE_FIT_CARD
        session = _make_session(
            outfit_suggestion=SAMPLE_OUTFIT,
            selected_item=SAMPLE_LISTING,
        )
        _dispatch("create_fit_card", {}, session)
        mock_fc.assert_called_once_with(SAMPLE_OUTFIT, SAMPLE_LISTING)

    @patch("agent.create_fit_card")
    def test_works_when_outfit_suggestion_is_string(self, mock_fc):
        """Case 1: suggest_outfit returned advice string; fit card should still work."""
        fallback_card = {
            "caption": "Check out my new thrifted find!",
            "hashtags": ["#thrifted"],
        }
        mock_fc.return_value = fallback_card
        session = _make_session(
            outfit_suggestion="Try pairing with wide-leg trousers.",
            selected_item=SAMPLE_LISTING,
        )
        obs = _dispatch("create_fit_card", {}, session)
        assert json.loads(obs) == fallback_card


# ─────────────────────────────────────────────────────────────────────────────
# E. _dispatch — unknown tool
# ─────────────────────────────────────────────────────────────────────────────

class TestDispatchUnknown:
    def test_unknown_tool_returns_skipped_message(self):
        session = _make_session()
        obs = _dispatch("nonexistent_tool", {}, session)
        assert "nonexistent_tool" in obs
        assert "skipped" in obs.lower() or "unknown" in obs.lower()

    def test_unknown_tool_does_not_set_error(self):
        session = _make_session(error=None)
        _dispatch("nonexistent_tool", {}, session)
        assert session["error"] is None

    def test_unknown_tool_does_not_mutate_session_data(self):
        session = _make_session()
        original_item = session["selected_item"]
        original_outfit = session["outfit_suggestion"]
        _dispatch("mystery_tool", {"arg": "value"}, session)
        assert session["selected_item"] is original_item
        assert session["outfit_suggestion"] is original_outfit


# ─────────────────────────────────────────────────────────────────────────────
# F. run_agent — API key guard
# ─────────────────────────────────────────────────────────────────────────────

class TestRunAgentApiKeyGuard:
    @patch("agent.Groq")
    def test_missing_api_key_sets_error(self, MockGroq):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GROQ_API_KEY", None)
            session = run_agent("tee under $30", SAMPLE_WARDROBE)
        assert session["error"] is not None

    @patch("agent.Groq")
    def test_missing_api_key_error_mentions_groq_key(self, MockGroq):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GROQ_API_KEY", None)
            session = run_agent("tee under $30", SAMPLE_WARDROBE)
        assert "GROQ_API_KEY" in session["error"]

    @patch("agent.Groq")
    def test_missing_api_key_groq_never_instantiated(self, MockGroq):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GROQ_API_KEY", None)
            run_agent("tee under $30", SAMPLE_WARDROBE)
        MockGroq.assert_not_called()

    @patch("agent.Groq")
    def test_missing_api_key_returns_session_dict(self, MockGroq):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GROQ_API_KEY", None)
            result = run_agent("tee", SAMPLE_WARDROBE)
        assert isinstance(result, dict)
        assert "error" in result


# ─────────────────────────────────────────────────────────────────────────────
# G. run_agent — happy path (all three tools called in order)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunAgentHappyPath:
    """
    Simulate: LLM calls search → suggest → create_fit_card → final answer.
    All tool functions mocked; Groq client mocked.
    """

    def _setup(self, mock_sl, mock_so, mock_fc, MockGroq):
        mock_sl.return_value = [SAMPLE_LISTING]
        mock_so.return_value = SAMPLE_OUTFIT
        mock_fc.return_value = SAMPLE_FIT_CARD

        client = MockGroq.return_value
        client.chat.completions.create.side_effect = [
            _make_tool_response("search_listings", {"description": "vintage jeans"}, "tc_1"),
            _make_tool_response("suggest_outfit",  {"item_id": "lst_001"},           "tc_2"),
            _make_tool_response("create_fit_card", {},                                "tc_3"),
            _make_final_response("Here is your complete outfit!"),
        ]

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.create_fit_card")
    @patch("agent.suggest_outfit")
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_returns_dict(self, MockGroq, mock_sl, mock_so, mock_fc):
        self._setup(mock_sl, mock_so, mock_fc, MockGroq)
        result = run_agent("vintage jeans under $40", SAMPLE_WARDROBE)
        assert isinstance(result, dict)

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.create_fit_card")
    @patch("agent.suggest_outfit")
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_error_is_none(self, MockGroq, mock_sl, mock_so, mock_fc):
        self._setup(mock_sl, mock_so, mock_fc, MockGroq)
        session = run_agent("vintage jeans under $40", SAMPLE_WARDROBE)
        assert session["error"] is None

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.create_fit_card")
    @patch("agent.suggest_outfit")
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_selected_item_populated(self, MockGroq, mock_sl, mock_so, mock_fc):
        self._setup(mock_sl, mock_so, mock_fc, MockGroq)
        session = run_agent("vintage jeans under $40", SAMPLE_WARDROBE)
        assert session["selected_item"] == SAMPLE_LISTING

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.create_fit_card")
    @patch("agent.suggest_outfit")
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_outfit_suggestion_populated(self, MockGroq, mock_sl, mock_so, mock_fc):
        self._setup(mock_sl, mock_so, mock_fc, MockGroq)
        session = run_agent("vintage jeans under $40", SAMPLE_WARDROBE)
        assert session["outfit_suggestion"] == SAMPLE_OUTFIT

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.create_fit_card")
    @patch("agent.suggest_outfit")
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_fit_card_populated(self, MockGroq, mock_sl, mock_so, mock_fc):
        self._setup(mock_sl, mock_so, mock_fc, MockGroq)
        session = run_agent("vintage jeans under $40", SAMPLE_WARDROBE)
        assert session["fit_card"] == SAMPLE_FIT_CARD

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.create_fit_card")
    @patch("agent.suggest_outfit")
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_messages_list_is_non_empty(self, MockGroq, mock_sl, mock_so, mock_fc):
        self._setup(mock_sl, mock_so, mock_fc, MockGroq)
        session = run_agent("vintage jeans under $40", SAMPLE_WARDROBE)
        assert len(session["messages"]) > 0

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.create_fit_card")
    @patch("agent.suggest_outfit")
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_first_message_is_system_prompt(self, MockGroq, mock_sl, mock_so, mock_fc):
        self._setup(mock_sl, mock_so, mock_fc, MockGroq)
        session = run_agent("vintage jeans under $40", SAMPLE_WARDROBE)
        assert session["messages"][0]["role"] == "system"
        assert SYSTEM_PROMPT in session["messages"][0]["content"]

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.create_fit_card")
    @patch("agent.suggest_outfit")
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_second_message_is_user_query(self, MockGroq, mock_sl, mock_so, mock_fc):
        self._setup(mock_sl, mock_so, mock_fc, MockGroq)
        session = run_agent("vintage jeans under $40", SAMPLE_WARDROBE)
        assert session["messages"][1]["role"] == "user"
        assert "vintage jeans" in session["messages"][1]["content"]

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.create_fit_card")
    @patch("agent.suggest_outfit")
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_search_listings_called_once(self, MockGroq, mock_sl, mock_so, mock_fc):
        self._setup(mock_sl, mock_so, mock_fc, MockGroq)
        run_agent("vintage jeans under $40", SAMPLE_WARDROBE)
        mock_sl.assert_called_once()

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.create_fit_card")
    @patch("agent.suggest_outfit")
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_suggest_outfit_called_once(self, MockGroq, mock_sl, mock_so, mock_fc):
        self._setup(mock_sl, mock_so, mock_fc, MockGroq)
        run_agent("vintage jeans under $40", SAMPLE_WARDROBE)
        mock_so.assert_called_once()

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.create_fit_card")
    @patch("agent.suggest_outfit")
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_create_fit_card_called_once(self, MockGroq, mock_sl, mock_so, mock_fc):
        self._setup(mock_sl, mock_so, mock_fc, MockGroq)
        run_agent("vintage jeans under $40", SAMPLE_WARDROBE)
        mock_fc.assert_called_once()

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.create_fit_card")
    @patch("agent.suggest_outfit")
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_groq_called_with_correct_model(self, MockGroq, mock_sl, mock_so, mock_fc):
        self._setup(mock_sl, mock_so, mock_fc, MockGroq)
        run_agent("vintage jeans", SAMPLE_WARDROBE)
        _, kwargs = MockGroq.return_value.chat.completions.create.call_args_list[0]
        assert kwargs["model"] == "llama-3.3-70b-versatile"

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.create_fit_card")
    @patch("agent.suggest_outfit")
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_groq_called_with_tool_schemas(self, MockGroq, mock_sl, mock_so, mock_fc):
        self._setup(mock_sl, mock_so, mock_fc, MockGroq)
        run_agent("vintage jeans", SAMPLE_WARDROBE)
        _, kwargs = MockGroq.return_value.chat.completions.create.call_args_list[0]
        assert kwargs["tools"] == TOOL_SCHEMAS

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.create_fit_card")
    @patch("agent.suggest_outfit")
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_groq_called_with_tool_choice_auto(self, MockGroq, mock_sl, mock_so, mock_fc):
        self._setup(mock_sl, mock_so, mock_fc, MockGroq)
        run_agent("vintage jeans", SAMPLE_WARDROBE)
        _, kwargs = MockGroq.return_value.chat.completions.create.call_args_list[0]
        assert kwargs["tool_choice"] == "auto"

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.create_fit_card")
    @patch("agent.suggest_outfit")
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_wardrobe_never_injected_into_user_or_system_messages(
        self, MockGroq, mock_sl, mock_so, mock_fc
    ):
        """The raw wardrobe dict must never be serialised into a user/system message.
        Tool observations from suggest_outfit may legitimately contain wardrobe-derived
        item IDs — only the direct injection of the wardrobe as a user/system prompt
        is forbidden (it would bloat context and leak data the LLM doesn't need).
        """
        self._setup(mock_sl, mock_so, mock_fc, MockGroq)
        run_agent("vintage jeans", SAMPLE_WARDROBE)
        all_calls = MockGroq.return_value.chat.completions.create.call_args_list
        for c in all_calls:
            messages = c[1]["messages"]
            for msg in messages:
                if msg.get("role") in ("user", "system"):
                    content = str(msg.get("content", ""))
                    assert "w_001" not in content, (
                        "wardrobe item id found in user/system message — "
                        "wardrobe must be injected at dispatch time, not via the prompt"
                    )


# ─────────────────────────────────────────────────────────────────────────────
# H. run_agent — search failure / early exit
# ─────────────────────────────────────────────────────────────────────────────

class TestRunAgentSearchFailure:
    def _setup_failure(self, mock_sl, MockGroq, error_msg="No listings found under $5.00."):
        mock_sl.return_value = error_msg
        client = MockGroq.return_value
        client.chat.completions.create.side_effect = [
            _make_tool_response("search_listings", {"description": "ballgown", "max_price": 5.0}, "tc_1"),
            _make_final_response("Sorry, I couldn't find anything."),
        ]

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.create_fit_card")
    @patch("agent.suggest_outfit")
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_error_set_to_search_failure_message(self, MockGroq, mock_sl, mock_so, mock_fc):
        err = "No listings found under $5.00."
        self._setup_failure(mock_sl, MockGroq, err)
        session = run_agent("designer ballgown size XXS under $5", SAMPLE_WARDROBE)
        assert session["error"] == err

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.create_fit_card")
    @patch("agent.suggest_outfit")
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_outfit_suggestion_remains_none(self, MockGroq, mock_sl, mock_so, mock_fc):
        self._setup_failure(mock_sl, MockGroq)
        session = run_agent("ballgown under $5", SAMPLE_WARDROBE)
        assert session["outfit_suggestion"] is None

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.create_fit_card")
    @patch("agent.suggest_outfit")
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_fit_card_remains_none(self, MockGroq, mock_sl, mock_so, mock_fc):
        self._setup_failure(mock_sl, MockGroq)
        session = run_agent("ballgown under $5", SAMPLE_WARDROBE)
        assert session["fit_card"] is None

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.create_fit_card")
    @patch("agent.suggest_outfit")
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_suggest_outfit_never_called(self, MockGroq, mock_sl, mock_so, mock_fc):
        self._setup_failure(mock_sl, MockGroq)
        run_agent("ballgown under $5", SAMPLE_WARDROBE)
        mock_so.assert_not_called()

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.create_fit_card")
    @patch("agent.suggest_outfit")
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_create_fit_card_never_called(self, MockGroq, mock_sl, mock_so, mock_fc):
        self._setup_failure(mock_sl, MockGroq)
        run_agent("ballgown under $5", SAMPLE_WARDROBE)
        mock_fc.assert_not_called()

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.create_fit_card")
    @patch("agent.suggest_outfit")
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_returns_valid_session_dict_on_failure(self, MockGroq, mock_sl, mock_so, mock_fc):
        self._setup_failure(mock_sl, MockGroq)
        result = run_agent("ballgown under $5", SAMPLE_WARDROBE)
        assert isinstance(result, dict)
        assert "error" in result


# ─────────────────────────────────────────────────────────────────────────────
# I. run_agent — LLM produces final answer on first turn (no tool calls)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunAgentImmediateFinalAnswer:
    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.create_fit_card")
    @patch("agent.suggest_outfit")
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_exits_cleanly_with_no_error(self, MockGroq, mock_sl, mock_so, mock_fc):
        MockGroq.return_value.chat.completions.create.return_value = (
            _make_final_response("I need more information.")
        )
        session = run_agent("?", SAMPLE_WARDROBE)
        assert session["error"] is None

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.create_fit_card")
    @patch("agent.suggest_outfit")
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_no_tools_called_when_llm_answers_immediately(self, MockGroq, mock_sl, mock_so, mock_fc):
        MockGroq.return_value.chat.completions.create.return_value = (
            _make_final_response("Please clarify your query.")
        )
        run_agent("?", SAMPLE_WARDROBE)
        mock_sl.assert_not_called()
        mock_so.assert_not_called()
        mock_fc.assert_not_called()

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.create_fit_card")
    @patch("agent.suggest_outfit")
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_groq_called_exactly_once(self, MockGroq, mock_sl, mock_so, mock_fc):
        MockGroq.return_value.chat.completions.create.return_value = (
            _make_final_response("Here's my response.")
        )
        run_agent("?", SAMPLE_WARDROBE)
        assert MockGroq.return_value.chat.completions.create.call_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# J. run_agent — MAX_TURNS safety cap
# ─────────────────────────────────────────────────────────────────────────────

class TestRunAgentMaxTurns:
    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_returns_after_max_turns_even_if_llm_keeps_calling_tools(
        self, MockGroq, mock_sl
    ):
        """The ReAct loop must cap at MAX_TURNS=10 even if the LLM never stops."""
        mock_sl.return_value = [SAMPLE_LISTING]
        # Supply exactly 10 tool-call responses (the MAX_TURNS cap)
        MockGroq.return_value.chat.completions.create.side_effect = [
            _make_tool_response("search_listings", {"description": "tee"}, f"tc_{i}")
            for i in range(10)
        ]
        # Should return (not hang or raise) despite no final answer
        result = run_agent("vintage tee", SAMPLE_WARDROBE)
        assert isinstance(result, dict)

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_groq_called_at_most_max_turns_times(self, MockGroq, mock_sl):
        mock_sl.return_value = [SAMPLE_LISTING]
        MockGroq.return_value.chat.completions.create.side_effect = [
            _make_tool_response("search_listings", {"description": "tee"}, f"tc_{i}")
            for i in range(10)
        ]
        run_agent("vintage tee", SAMPLE_WARDROBE)
        assert MockGroq.return_value.chat.completions.create.call_count <= 10


# ─────────────────────────────────────────────────────────────────────────────
# K. run_agent — malformed JSON in tool arguments
# ─────────────────────────────────────────────────────────────────────────────

class TestRunAgentMalformedToolArgs:
    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_bad_json_in_tool_args_does_not_crash(self, MockGroq, mock_sl):
        mock_sl.return_value = [SAMPLE_LISTING]

        # Build a tool call whose arguments are not valid JSON
        bad_tc = MagicMock()
        bad_tc.id = "tc_bad"
        bad_tc.function.name = "search_listings"
        bad_tc.function.arguments = "{not valid json"

        bad_response = MagicMock()
        bad_response.choices[0].message.tool_calls = [bad_tc]
        bad_response.choices[0].message.content = None

        final = _make_final_response("Done.")
        MockGroq.return_value.chat.completions.create.side_effect = [bad_response, final]

        # Must not raise; gracefully falls back to empty args dict
        result = run_agent("tee under $30", SAMPLE_WARDROBE)
        assert isinstance(result, dict)

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_bad_json_falls_back_to_empty_args(self, MockGroq, mock_sl):
        mock_sl.return_value = [SAMPLE_LISTING]

        bad_tc = MagicMock()
        bad_tc.id = "tc_bad"
        bad_tc.function.name = "search_listings"
        bad_tc.function.arguments = "<<<invalid>>>"

        bad_response = MagicMock()
        bad_response.choices[0].message.tool_calls = [bad_tc]
        bad_response.choices[0].message.content = None

        final = _make_final_response("Done.")
        MockGroq.return_value.chat.completions.create.side_effect = [bad_response, final]

        run_agent("tee", SAMPLE_WARDROBE)
        # search_listings called with empty description (fallback from empty dict)
        mock_sl.assert_called_once_with(description="", size=None, max_price=None)


# ─────────────────────────────────────────────────────────────────────────────
# L. run_agent — conversation history / message structure
# ─────────────────────────────────────────────────────────────────────────────

class TestRunAgentMessageStructure:
    def _run_happy(self, MockGroq, mock_sl, mock_so, mock_fc):
        mock_sl.return_value = [SAMPLE_LISTING]
        mock_so.return_value = SAMPLE_OUTFIT
        mock_fc.return_value = SAMPLE_FIT_CARD
        MockGroq.return_value.chat.completions.create.side_effect = [
            _make_tool_response("search_listings", {"description": "jeans"}, "tc_1"),
            _make_tool_response("suggest_outfit",  {"item_id": "lst_001"},   "tc_2"),
            _make_tool_response("create_fit_card", {},                        "tc_3"),
            _make_final_response("All done!"),
        ]
        return run_agent("vintage jeans", SAMPLE_WARDROBE)

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.create_fit_card")
    @patch("agent.suggest_outfit")
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_tool_observations_have_role_tool(self, MockGroq, mock_sl, mock_so, mock_fc):
        session = self._run_happy(MockGroq, mock_sl, mock_so, mock_fc)
        tool_msgs = [m for m in session["messages"] if m.get("role") == "tool"]
        assert len(tool_msgs) >= 3

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.create_fit_card")
    @patch("agent.suggest_outfit")
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_tool_observations_have_tool_call_id(self, MockGroq, mock_sl, mock_so, mock_fc):
        session = self._run_happy(MockGroq, mock_sl, mock_so, mock_fc)
        tool_msgs = [m for m in session["messages"] if m.get("role") == "tool"]
        for msg in tool_msgs:
            assert "tool_call_id" in msg

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.create_fit_card")
    @patch("agent.suggest_outfit")
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_assistant_messages_with_tools_include_tool_calls_key(
        self, MockGroq, mock_sl, mock_so, mock_fc
    ):
        session = self._run_happy(MockGroq, mock_sl, mock_so, mock_fc)
        assistant_msgs_with_tools = [
            m for m in session["messages"]
            if m.get("role") == "assistant" and m.get("tool_calls")
        ]
        assert len(assistant_msgs_with_tools) >= 3

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.create_fit_card")
    @patch("agent.suggest_outfit")
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_tool_call_ids_match_between_assistant_and_tool_messages(
        self, MockGroq, mock_sl, mock_so, mock_fc
    ):
        session = self._run_happy(MockGroq, mock_sl, mock_so, mock_fc)
        issued_ids = set()
        for msg in session["messages"]:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls", []):
                    issued_ids.add(tc["id"])
        observed_ids = {
            m["tool_call_id"]
            for m in session["messages"]
            if m.get("role") == "tool"
        }
        assert issued_ids == observed_ids

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.create_fit_card")
    @patch("agent.suggest_outfit")
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_observation_content_is_string(self, MockGroq, mock_sl, mock_so, mock_fc):
        session = self._run_happy(MockGroq, mock_sl, mock_so, mock_fc)
        tool_msgs = [m for m in session["messages"] if m.get("role") == "tool"]
        for msg in tool_msgs:
            assert isinstance(msg["content"], str)

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("agent.create_fit_card")
    @patch("agent.suggest_outfit")
    @patch("agent.search_listings")
    @patch("agent.Groq")
    def test_search_observation_is_valid_json(self, MockGroq, mock_sl, mock_so, mock_fc):
        session = self._run_happy(MockGroq, mock_sl, mock_so, mock_fc)
        # The first tool message should be the search_listings observation
        tool_msgs = [m for m in session["messages"] if m.get("role") == "tool"]
        first_obs = tool_msgs[0]["content"]
        # Should parse as JSON (slim preview)
        json.loads(first_obs)   # raises if invalid


# ─────────────────────────────────────────────────────────────────────────────
# M. TOOL_SCHEMAS constants
# ─────────────────────────────────────────────────────────────────────────────

class TestToolSchemas:
    def test_tool_schemas_is_list_of_three(self):
        assert isinstance(TOOL_SCHEMAS, list)
        assert len(TOOL_SCHEMAS) == 3

    def test_all_schemas_have_type_function(self):
        for schema in TOOL_SCHEMAS:
            assert schema["type"] == "function"

    def test_tool_names_are_correct(self):
        names = [s["function"]["name"] for s in TOOL_SCHEMAS]
        assert names == ["search_listings", "suggest_outfit", "create_fit_card"]

    def test_search_listings_schema_requires_description(self):
        sl = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "search_listings")
        assert "description" in sl["function"]["parameters"]["required"]

    def test_search_listings_schema_has_size_and_max_price_as_optional(self):
        sl = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "search_listings")
        props = sl["function"]["parameters"]["properties"]
        required = sl["function"]["parameters"]["required"]
        assert "size" in props
        assert "max_price" in props
        assert "size" not in required
        assert "max_price" not in required

    def test_suggest_outfit_schema_requires_item_id(self):
        so = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "suggest_outfit")
        assert "item_id" in so["function"]["parameters"]["required"]

    def test_create_fit_card_schema_requires_no_parameters(self):
        fc = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "create_fit_card")
        assert fc["function"]["parameters"]["required"] == []

    def test_all_schemas_have_non_empty_descriptions(self):
        for schema in TOOL_SCHEMAS:
            desc = schema["function"]["description"]
            assert isinstance(desc, str) and len(desc) > 10

    def test_schema_is_valid_groq_function_call_format(self):
        for schema in TOOL_SCHEMAS:
            assert "type" in schema
            assert "function" in schema
            fn = schema["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn
            params = fn["parameters"]
            assert params["type"] == "object"
            assert "properties" in params
            assert "required" in params
