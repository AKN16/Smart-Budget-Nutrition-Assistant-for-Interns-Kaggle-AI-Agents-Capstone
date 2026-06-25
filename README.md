# Smart Budget Nutrition Assistant for Interns

An agentic Python tool that stops interns from eating instant noodles for every meal. It detects where you are, looks up **real, current** local grocery prices using Gemini's Google Search grounding, and generates a structured 7-day meal plan that actually fits your weekly budget.

```
GEMINI_API_KEY=... python agent.py --budget 500000 --currency VND
```

---

## 1. Project Architecture

The agent runs a four-stage reasoning loop rather than a single prompt-and-pray call. Each stage exists to remove a specific failure mode (bad location, hallucinated prices, malformed output, over-budget plans).

```
 ┌──────────────┐    ┌────────────────────┐    ┌───────────────────────┐    ┌────────────────────┐
 │   1. LOCATE  │ -> │   2. GROUND        │ -> │   3. PLAN             │ -> │  4. SELF-CORRECT    │
 │  IP geo-     │    │  Gemini + Google   │    │  Gemini + JSON Schema │    │  Recompute totals,  │
 │  location    │    │  Search grounding  │    │  (Pydantic model)     │    │  re-prompt if over  │
 │  lookup      │    │  tool -> live      │    │  -> structured 7-day  │    │  budget (up to 2x)  │
 │              │    │  market prices     │    │  WeeklyBudgetPlan     │    │                     │
 └──────────────┘    └────────────────────┘    └───────────────────────┘    └────────────────────┘
```

**Stage 1 -- Locate.** `detect_location()` calls a free IP geolocation API to resolve the user's city and country with no input required. If every endpoint fails (no network, rate-limited, offline demo), the agent degrades gracefully to a fixed default location and the pipeline keeps running rather than crashing -- a `--location "City,Country"` flag also lets the user override detection entirely.

**Stage 2 -- Ground.** `research_local_prices()` sends a single `generate_content` call with the `google_search` tool enabled. The model decides what to search for, executes the queries itself, and returns prose grounded in live web results, plus structured `grounding_metadata` (the actual search queries used and the source URLs). This is the step that prevents the model from inventing prices from its training data.

**Stage 3 -- Plan.** `generate_meal_plan()` takes the grounded price text from Stage 2 and feeds it back into a *second*, separate `generate_content` call -- this time with `response_schema=WeeklyBudgetPlan` (a nested Pydantic model) instead of the search tool. Gemini returns JSON that is guaranteed to match the schema: 7 days, 3 meals per day, a consolidated shopping list, money-saving tips, and a nutrition summary.

**Stage 4 -- Self-correct.** `_validate_and_correct_budget()` doesn't just trust the model's own arithmetic. It recomputes the weekly total from the individual daily costs the model returned and compares it against the stated budget. If the plan is more than 5% over budget, the agent re-prompts Gemini with the exact overage and asks for a revised plan -- up to two correction rounds -- before falling back to the best available plan. This turns a single LLM call into a small closed-loop optimization process.

---

## 2. Technical Highlight: Why Grounding Solves the "Hallucinated Prices" Problem

Two facts about the Gemini API drove a key architectural decision in this project:

1. **LLMs are unreliable on numbers that change weekly.** A model's training data has a fixed cutoff, so it has no idea that eggs went up 8% last month in a given city, or that a particular noodle brand is on promotion this week. Asking it directly for "current grocery prices in X" produces a plausible-sounding but unverifiable number.
2. **Grounding with Google Search fixes this by injecting live retrieval into the same API call.** When the `google_search` tool is attached, Gemini decides whether a query needs real-world data, generates its own search queries, executes them, and synthesizes an answer from the actual search results -- returning both the answer and the source URLs (`grounding_metadata.grounding_chunks`) used to produce it. This is the same mechanism Google demonstrated with time-sensitive questions like award-show winners, where an un-grounded model answers from stale memory and a grounded one returns the verified, current answer with citations.

**The non-obvious constraint this project had to design around:** for the Gemini model family used here, *Google Search grounding and strict JSON schema / structured output cannot be requested in the same API call* -- the API rejects a request that tries to combine `tools=[google_search]` with `response_schema`. So this agent deliberately splits the work into two calls instead of one:

- Call 1 (Stage 2): grounding tool *on*, schema *off* -> free-text, citation-backed price research.
- Call 2 (Stage 3): grounding tool *off*, schema *on* -> the price research from Call 1 is pasted into the prompt as plain-text context, and the model is constrained to return valid `WeeklyBudgetPlan` JSON.

This two-call pattern is the practical fix for "I need both live facts and a guaranteed-parseable structure," and it's a pattern worth reusing any time grounded research needs to flow into a strictly-typed downstream object.

---

## 3. Setup & Execution

### Prerequisites
- Python 3.10+
- A free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

### Install

```bash
git clone <your-repo-url>
cd smart-budget-nutrition-assistant
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt`:
```
google-genai>=1.0.0
pydantic>=2.6.0
requests>=2.31.0
```

### Set your API key

```bash
# macOS / Linux
export GEMINI_API_KEY="your-key-here"

# Windows (Command Prompt)
setx GEMINI_API_KEY "your-key-here"

# Windows (PowerShell, current session)
$env:GEMINI_API_KEY = "your-key-here"
```

### Run it

```bash
# Auto-detect location, plan for a 500,000 VND weekly budget
python agent.py --budget 500000 --currency VND

# Override location instead of auto-detecting
python agent.py --budget 50 --currency USD --location "Manila,Philippines"

# Use a different Gemini model or output folder
python agent.py --budget 800000 --currency VND --model gemini-2.5-pro --output-dir my_plans
```

### Output

Each run writes three files into `output/` (or your `--output-dir`):

| File | Contents |
|---|---|
| `meal_plan_<timestamp>.json` | The full structured plan, machine-readable |
| `meal_plan_<timestamp>.md` | A human-readable Markdown version with a shopping list table and sources |
| `price_research_<timestamp>.txt` | The raw grounded search output and the queries Gemini actually ran |

The plan is also printed to the terminal as formatted JSON for quick inspection.

---

## 4. Project Structure

```
smart-budget-nutrition-assistant/
├── agent.py            # Full agent implementation (this is the deliverable)
├── requirements.txt
├── README.md
└── output/             # Generated at runtime
```

## 5. Limitations & Honest Caveats

- IP-based geolocation is approximate (city-level at best) and can be wrong on VPNs/corporate networks -- always overridable via `--location`.
- Grounded prices reflect what's indexed by Google Search at request time; for very small or hyperlocal markets (e.g. a specific wet market stall), treat them as a realistic ballpark, not a guaranteed receipt total.
- Calorie and price estimates are AI-generated approximations for planning purposes, not medical or financial advice.
