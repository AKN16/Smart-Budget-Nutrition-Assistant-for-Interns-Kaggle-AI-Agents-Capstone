# Smart Budget Nutrition Assistant for Interns -- a grounded-Gemini agent that plans real, affordable meals

**TL;DR:** I built an agent that detects your location, uses Gemini's Google Search grounding to pull *live* grocery prices instead of hallucinated ones, then generates a structured 7-day meal plan that's automatically checked and corrected against your real budget. Full code + writeup below.

---

### The problem

If you've ever been a broke student or an intern, you know the pattern: instant noodles for dinner, again, because actually planning a week of affordable, balanced meals takes more time and money-tracking than anyone in that position has to spare. Generic "healthy meal plan" content online almost never accounts for *your* local prices, *your* currency, or *your* actual weekly budget.

### The idea

Instead of asking an LLM to "just write me a cheap meal plan" (which quietly hallucinates prices from outdated training data), I split the problem into stages an agent can actually reason through:

1. **Locate** -- detect the user's city/country automatically via IP geolocation, no manual input needed.
2. **Ground** -- call Gemini with the built-in `google_search` tool to fetch current, real grocery prices for that specific location, with citations.
3. **Plan** -- feed that grounded price data back into a second Gemini call constrained by a strict JSON schema (via Pydantic), producing a fully structured 7-day plan: meals, calories, costs, a shopping list, tips.
4. **Self-correct** -- recompute the plan's real total cost in Python and, if it overshoots the stated budget by more than 5%, automatically re-prompt Gemini with the exact overage so it revises itself. Up to two correction rounds before settling on a best-effort plan.

The most interesting engineering constraint: **Google Search grounding and structured JSON output can't be requested in the same Gemini call** for this model family. So the architecture deliberately uses two separate calls -- one grounded and freeform, one schema-constrained and tool-free -- and bridges them by passing the grounded text as context into the second prompt. It's a small detail, but it's the difference between a demo that "mostly works" and one that reliably returns parseable, budget-aware output every time.

### Tech stack

- **`google-genai`** -- the unified Python SDK for the Gemini API
- **Google Search grounding tool** -- for live, citation-backed price data
- **Pydantic** -- defines the `WeeklyBudgetPlan` schema that Gemini's structured output is constrained to
- **Free IP geolocation API** -- zero-config location detection with graceful fallback
- Plain **Python 3** standard library for orchestration, retries, and Markdown/JSON export

### What it outputs

Running `python agent.py --budget 500000 --currency VND` produces:
- a JSON file with the full structured plan (ready to feed into any frontend),
- a clean Markdown version with a shopping-list table and the actual sources Gemini grounded on,
- the raw grounded price research, so you can audit exactly what the model searched for.

### Why this matters beyond meal planning

The real takeaway isn't the noodles -- it's the pattern: **grounding for facts, schemas for structure, and a cheap Python-side validation loop for correctness.** That three-part pattern generalizes to a lot of "agent that needs to be right, not just fluent" use cases: budget travel planners, price-comparison tools, local-market research bots, and so on.

### Repo

Full source (`agent.py`), `requirements.txt`, and setup instructions are linked below -- it's a single self-contained script with no placeholders, ready to run with just a Gemini API key.

---

*Built while working through Google's Gen AI learning material -- happy to answer questions about the grounding + structured-output split, the budget self-correction loop, or extending it to other countries/currencies in the comments!*
