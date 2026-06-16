# FitFindr — planning.md

> Complete this document before writing any implementation code.
> Your spec and agent diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Your planning.md will be reviewed as part of your submission.
> Update it before starting any stretch features.

---

## Tools

List every tool your agent will use. For each tool, fill in all four fields.
You must have at least 3 tools. The three required tools are listed — add any additional tools below them.

### Tool 1: search_listings

**What it does:**
<!-- Describe what this tool does in 1–2 sentences -->
Searches the mock clothing database for available secondhand items that match the user's requested style, size, and budget. It acts as the crucial first step in the agent's pipeline by attempting to secure a concrete, available item before the agent is allowed to move on to outfit planning.

**Input parameters:**
<!-- List each parameter, its type, and what it represents -->
- `description` (str): The specific type or style of the item the user wants (e.g., "vintage Levi's 501s", "chunky knit sweater").
- `size` (str): The alphanumeric size requested by the user (e.g., "M", "32", "US 8").
- `max_price` (float): The upper limit of the user's budget in dollars.

**What it returns:**
<!-- Describe the return value — what fields does a result contain? -->
A list of dictionaries representing the matching items. Each dictionary must contain the necessary metadata for downstream tools to function. For example: [{"item_id": "101", "title": "Vintage Levi's 501s", "price": 45.00, "size": "32", "color": "blue", "condition": "good"}]

**What happens if it fails or returns nothing:**
<!-- What should the agent do if no listings match? -->
If the database query returns an empty list [], the tool execution ends, and the agent's planning loop must intercept this state. The agent must formulate a response telling the user that no items were found and explicitly ask them to relax a constraint (e.g., "I couldn't find any Levi's 501s in size 32 under $50. Would you like me to increase the budget or check for a size 34?"). The agent should not pass an empty result forward to suggest_outfit tool. 

---

### Tool 2: suggest_outfit

**What it does:**
<!-- Describe what this tool does in 1–2 sentences -->
Evaluates a newly sourced clothing item against the user's existing closet inventory to generate a cohesive outfit. It acts as the synthesis step, bridging the gap between what the user just bought (or wants to buy) and what they already own to prove the new item's wearability.

**Input parameters:**
<!-- List each parameter, its type, and what it represents -->
- `new_item` (dict): A single dictionary representing the specific item successfully retrieved by search_listings. It must contain the item's defining attributes (e.g., {"item_id": "101", "type": "jeans", "color": "blue", "style": "vintage"}).
- `wardrobe` (dict): A dictionary representing the user's current closet inventory, strictly formatted as a list of item dictionaries under an items key (e.g., {"items": [{"type": "t-shirt", "color": "white"}, {"type": "sneakers", "color": "black"}]}).

**What it returns:**
<!-- Describe the return value -->
A dictionary containing the finalized outfit composition. It should return a list of the selected items (including the new_item) and a brief mechanical note on why they pair well. Example: {"outfit": [{"type": "jeans", "id": "101"}, {"type": "t-shirt", "color": "white"}], "reasoning": "Classic blue denim pairs neutrally with a basic white top."}.

**What happens if it fails or returns nothing:**
<!-- What should the agent do if the wardrobe is empty or no outfit can be suggested? -->
If the tool receives an empty wardrobe dictionary ({"items": []}), or if it cannot mathematically find a color/style match, it must return an explicit failure signal (e.g., {"outfit": [], "error": "No matching wardrobe items"}). The agent's planning loop must intercept this and execute a fallback strategy: inform the user that their closet lacks a match, and suggest generic staple pieces to complete the look (e.g., "Since you don't have items saved yet, I suggest pairing these jeans with a basic white tee and canvas sneakers."). The agent must never invent or hallucinate items into the user's wardrobe. It should not even attempt to call Tool 3 because it does not have a real outfit to pass forward.

---

### Tool 3: create_fit_card

**What it does:**
<!-- Describe what this tool does in 1–2 sentences -->
Translates the structured, mechanical outfit data from the previous step into a creative, human-readable social media caption. It serves as the presentation layer of your agent, turning raw database items and pairing logic into an engaging summary.

**Input parameters:**
<!-- List each parameter, its type, and what it represents -->
- `outfit` (list): A list of dictionaries representing the full outfit generated by Tool 2 (e.g., [{"type": "jeans", "id": "101"}, {"type": "t-shirt", "color": "white"}]).
- `new_item` (dict): The original dictionary of the newly thrifted item from Tool 1.

**What it returns:**
<!-- Describe the return value -->
A dictionary containing the generated copy and metadata. Example: {"caption": "Snagged these vintage 501s and paired them with a crisp white tee for an effortless weekend look.", "hashtags": ["#thrifted", "#vintagelevis", "#fitcheck"]}.

**What happens if it fails or returns nothing:**
<!-- What should the agent do if the outfit data is incomplete? -->
If the tool receives incomplete data (e.g., an empty outfit list) or if the LLM generation fails (e.g., network timeout, output formatting error), the agent must catch the exception to prevent a system crash. It must bypass the LLM entirely and execute a fallback strategy by outputting a hardcoded string using the safely stored new_item data (e.g., "Check out my new [insert new_item title]!"). This ensures the application degrades gracefully and always returns a final response to the user.

---

### Additional Tools (if any)

<!-- Copy the block above for any tools beyond the required three -->

---

## Planning Loop

**How does your agent decide which tool to call next?**
<!-- Describe the logic your planning loop uses. What does it look at? What conditions change its behavior? How does it know when it's done? -->

The agent operates on a reactive, state-driven loop. At each iteration, it evaluates the current state of its context window (the user's initial request plus the outputs of any previously called tools) against a strict sequence of rules defined in the system prompt.

***What it looks at***: 

It examines the exact data payload returned by the most recent tool. It does not guess; it reads the dictionary output.

Conditions that change behavior: The loop is completely dictated by data validation.

***Success Path***: 

If a tool returns populated data (e.g., search_listings finds a match), the loop immediately routes that specific data payload into the next tool in the sequence (suggest_outfit).

***Failure Path***: 

If a tool returns an empty list, an error flag, or fails to find a match, the loop explicitly aborts the primary sequence. It intercepts the error, triggers the designated fallback response for that specific tool, and shifts from "tool calling mode" back to "conversational mode" to ask the user for clarification.

***How it knows when it is done***: 

The loop terminates when it reaches a definitive endpoint. This happens either when create_fit_card successfully returns a final caption, or when a tool's fallback strategy has been executed. Once a terminal state is reached, the agent stops triggering functions, outputs a final natural language message to the user, and waits for a new prompt.

---

## State Management

**How does information from one tool get passed to the next?**
<!-- Describe how your agent stores and accesses state within a session. What data is tracked? How is it passed between tool calls? -->
FitFindr manages state using a hybrid architecture: a continuous Conversational Transcript for dynamic, real-time data, and Session Variables for static background data.

***What data is tracked & stored***: 

The agent tracks the chronological history of the session—user prompts, the LLM's routing decisions, and the exact data payloads returned by tools—within a continuous Python list of dictionaries (the messages array). Conversely, static baseline data, such as the example_wardrobe, is tracked independently by loading it into a local Python session variable at runtime rather than flooding the conversation transcript.

***How it is passed between tool calls***: 

Data moves sequentially between tools by using the LLM's context window as a router. When Tool 1 (search_listings) returns a thrifted item, the Python backend explicitly appends that result to the transcript array. On the next iteration of the planning loop, the LLM reads the updated transcript, recognizes the new item data, and mechanically extracts those details to construct the exact input arguments for Tool 2 (suggest_outfit). As the LLM triggers Tool 2, the Python backend intercepts the call and seamlessly injects the locally stored wardrobe session variable alongside the LLM's extracted item data.


---

## Error Handling

For each tool, describe the specific failure mode you're handling and what the agent does in response.
| Tool | Failure Mode | Agent Response |
|------|-------------|----------------|
| **search_listings** | No listings match the user's search criteria. | The agent detects the empty result set, terminates the workflow, and asks the user to adjust specific constraints such as budget, size, or location instead of generating nonexistent listings. |
| **suggest_outfit** | No compatible items are found in the user's wardrobe. | The agent halts the recommendation process, informs the user that no matching outfit can be created from their current closet, and suggests versatile staple items that could complete the look. |
| **create_fit_card** | Required outfit information is missing or incomplete. | The agent catches the validation error, skips the fit-card generation step, and returns a predefined fallback message using available item information (e.g., "Check out my new [item]!") to prevent workflow failure. |

---

## Architecture

<!-- Draw a diagram of your agent showing how the components connect:
     User input → Planning Loop → Tools (search_listings, suggest_outfit, create_fit_card)
                                                                          ↕
                                                                   State / Session
     Show what triggers each tool, how state flows between them, and where error paths branch off.
     ASCII art, a Mermaid diagram (https://mermaid.js.org/syntax/flowchart.html), or an embedded
     sketch are all fine. You'll share this diagram with an AI tool when asking it to implement
     the planning loop and each individual tool. -->

![FitFindr Architecture Flowchart](architecture_diagram.png)

---

## AI Tool Plan

<!-- For each part of the implementation below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, your agent diagram)
     - What you expect it to produce
     - How you'll verify the output matches your spec before moving on

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Tool 1 spec (inputs, return value, failure mode) and ask it to implement
     search_listings() using load_listings() from the data loader — then test it against 3 queries
     before trusting it" is a plan. -->

**Milestone 3 — Individual tool implementations:**

- **AI Tool:** Claude 4.6 Sonnet 

- **Input:** I will give Claude one tool specification at a time from this document, including the inputs, return values, error-handling requirements, and the mock JSON database schema. I'll instruct it: *"Write this as a standalone Python function. Do not build the agent loop."*

- **Expected Output:** Three separate Python functions:
  - `search_listings`
  - `suggest_outfit`
  - `create_fit_card`

  Each function should include error handling for the failure cases defined in the Error Handling table.

- **Verification:** Before building the planning loop, I'll test each function manually. I'll run normal test cases to confirm the expected behavior and also provide invalid inputs (such as an empty wardrobe or a search for an item that doesn't exist) to verify that the fallback logic works correctly and the program doesn't crash.

**Milestone 4 — Planning loop and state management:**
- **AI Tool:** Claude 4.6 Sonnet 

- **Input:** I will provide Claude Code with the State Management section, the completed Mermaid architecture diagram, and the three tested Python functions. My prompt will be:

  > "Build the main loop that manages the conversation history. Always append tool outputs to the messages array to preserve context, and only inject the static `example_wardrobe` variable when calling Tool 2."

- **Expected Output:** An `app.py` script that:
  - Handles the LLM's tool-calling decisions
  - Routes requests to the correct local Python functions
  - Updates the conversation history
  - Sends the updated transcript back to the API

- **Verification:** I'll temporarily add `print(messages)` at the end of the loop and run a full conversation. I'll verify that the message history grows correctly after each turn and that the LLM extracts and passes the outfit data correctly to Tool 3.

---

## A Complete Interaction (Step by Step)

Write out what a full user interaction looks like from start to finish — tool call by tool call. Use a specific example query.

**Example user query:** "I'm looking for a vintage graphic tee under $30. I mostly wear baggy jeans and chunky sneakers. What's out there and how would I style it?"

**Step 1:**
<!-- What does the agent do first? Which tool is called? With what input? -->

The LLM reads the user's prompt in the transcript, identifies the core search parameters, and determines the first action is to find the item.

- **Tool called:**  
  `search_listings`

- **Input payload:**

```json
{
  "description": "vintage graphic tee",
  "max_price": 30.0
}
```

- **Result:**  
  The Python backend queries the mock database and returns:

```json
{
  "id": "204",
  "title": "90s Harley Davidson Tee",
  "price": 25.00,
  "source": "thrift"
}
```

The backend appends this result to the conversation transcript.

**Step 2:**
<!-- What happens next? What was returned from step 1? What tool is called now? -->
The LLM reads the updated transcript, sees the Harley tee, and recognizes the user's request for styling suggestions.

- **Tool called:**  
  `suggest_outfit`

- **Input payload:**

```json
{
  "id": "204",
  "title": "90s Harley Davidson Tee",
  "price": 25.00,
  "source": "thrift"
}
```

> Note: The Python backend intercepts this call and injects the local `example_wardrobe` session variable before executing the tool.

- **Result:**  
  The tool selects complementary wardrobe items based on the user's preference for baggy jeans and chunky sneakers. It returns an outfit array containing three item dictionaries.

The backend appends the outfit array to the transcript.

**Step 3:**
<!-- Continue until the full interaction is complete -->
The LLM reads the transcript, sees the completed outfit, and determines that the final step is to generate a social media-style fit card.

- **Tool called:**  
  `create_fit_card`

- **Input payload:**  
  The full outfit array generated in Step 2.

- **Result:**  
  The third-party API returns:

```json
{
  "caption": "Just scored this faded 90s Harley tee for $25. Throwing it on with the baggy denim and chunky sneakers for that effortless weekend uniform.",
  "hashtags": [
    "#VintageFinds",
    "#Streetwear",
    "#Thrifted"
  ]
}
```

The backend appends this result to the transcript.

**Final output to user:**
<!-- What does the user actually see at the end? -->
The LLM reads the complete transcript, sees that all required tools have returned successfully, exits tool-calling mode, and generates a natural-language response.

- **What the user sees:**

> I found a great option for you: a 90s Harley Davidson Tee for $25.
>
> To match your style, I paired it with your light-wash baggy jeans and white chunky sneakers from your wardrobe.
>
> Here's your Fit Card:
>
> *Just scored this faded 90s Harley tee for $25. Throwing it on with the baggy denim and chunky sneakers for that effortless weekend uniform. *
>
> #VintageFinds #Streetwear #Thrifted
