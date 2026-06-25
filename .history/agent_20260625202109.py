#!/usr/bin/env python3
"""
Smart Budget Nutrition Assistant for Interns
=============================================

A production-grade agent that helps cash-strapped interns eat well without
living on instant noodles. It runs a three-stage agentic pipeline:

    1. LOCATE   -> detect the user's approximate city/country via IP geolocation
    2. GROUND   -> use Gemini's Google Search grounding tool to fetch *current*,
                   real-world grocery prices for that location (no hallucinated
                   numbers)
    3. PLAN     -> feed the grounded price data back into Gemini with a strict
                   JSON schema to generate a structured, budget-aware 7-day
                   meal plan, then self-validate the plan against the stated
                   budget and auto-correct if it overshoots

SDK: google-genai (the unified Google Gen AI Python SDK)
Docs: https://ai.google.dev/gemini-api/docs

Usage:
    export GEMINI_API_KEY="your-key-here"
    python agent.py --budget 500000 --currency VND
    python agent.py --budget 50 --currency USD --location "Manila,Philippines"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests
from pydantic import BaseModel, Field, ValidationError

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SmartBudgetNutritionAgent")

# --------------------------------------------------------------------------- #
# Configuration constants
# --------------------------------------------------------------------------- #

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
MAX_API_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0
BUDGET_TOLERANCE = 0.05  # allow plan to be at most 5% over budget before re-asking
MAX_PLAN_CORRECTION_ROUNDS = 2

# Free, no-API-key-required IP geolocation endpoints, tried in order.
IP_GEOLOCATION_ENDPOINTS = [
    "https://ipinfo.io/json",
    "https://ipapi.co/json/",
]

# Fallback if every geolocation lookup fails (e.g. no network egress allowed).
FALLBACK_CITY = "Ha Noi"
FALLBACK_COUNTRY = "Vietnam"

# Minimal country -> currency map for sensible defaults. Extend as needed.
COUNTRY_CURRENCY_MAP = {
    "vietnam": "VND",
    "united states": "USD",
    "philippines": "PHP",
    "indonesia": "IDR",
    "india": "INR",
    "thailand": "THB",
    "malaysia": "MYR",
    "singapore": "SGD",
    "japan": "JPY",
    "south korea": "KRW",
    "united kingdom": "GBP",
    "germany": "EUR",
    "france": "EUR",
    "australia": "AUD",
    "canada": "CAD",
}
DEFAULT_CURRENCY = "USD"
NO_DECIMAL_CURRENCIES = {"VND", "JPY", "KRW", "IDR"}

# Staple grocery items the grounding search will price out. Kept broad and
# culturally flexible so it works reasonably well across countries.
STAPLE_INGREDIENTS = [
    "white rice", "eggs", "instant noodles", "chicken thigh or breast",
    "pork", "tofu", "leafy green vegetables", "bananas or seasonal fruit",
    "cooking oil", "soy sauce or fish sauce", "instant coffee or tea",
    "bread or buns",
]


# --------------------------------------------------------------------------- #
# Plain data containers for the location + grounding stages
# --------------------------------------------------------------------------- #

@dataclass
class LocationInfo:
    city: str
    country: str
    source: str
    raw: dict = field(default_factory=dict)

    def label(self) -> str:
        return f"{self.city}, {self.country}"


@dataclass
class GroundedPriceResearch:
    raw_text: str
    search_queries: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    grounded: bool = False


# --------------------------------------------------------------------------- #
# Structured output schema for the final meal plan (Pydantic -> JSON Schema)
# --------------------------------------------------------------------------- #

class ShoppingListItem(BaseModel):
    ingredient: str = Field(description="Name of the ingredient to buy")
    quantity: str = Field(description="Human-readable total quantity for the week, e.g. '3 kg', '1 dozen'")
    estimated_price: float = Field(description="Estimated total cost for this quantity, in the target currency")
    note: Optional[str] = Field(default=None, description="Where to buy it cheaply, or a substitution tip")


class Meal(BaseModel):
    meal_type: str = Field(description="One of: breakfast, lunch, dinner")
    dish_name: str = Field(description="Short, appetizing name of the dish")
    ingredients: list[str] = Field(description="Key ingredients used in this dish")
    estimated_cost: float = Field(description="Estimated cost of this single meal, in the target currency")
    estimated_calories: int = Field(description="Rough calorie estimate for this meal")


class DayPlan(BaseModel):
    day_number: int = Field(description="1 through 7")
    day_label: str = Field(description="e.g. 'Monday'")
    meals: list[Meal] = Field(description="Exactly three meals: breakfast, lunch, dinner")
    daily_total_cost: float = Field(description="Sum of estimated_cost across the day's meals")
    daily_total_calories: int = Field(description="Sum of estimated_calories across the day's meals")


class WeeklyBudgetPlan(BaseModel):
    location_city: str
    location_country: str
    currency: str
    weekly_budget: float
    estimated_weekly_total_cost: float = Field(description="Sum of all daily_total_cost values across the 7 days")
    estimated_savings: float = Field(description="weekly_budget minus estimated_weekly_total_cost; can be negative")
    days: list[DayPlan] = Field(description="Exactly 7 entries, day_number 1 to 7")
    shopping_list: list[ShoppingListItem] = Field(description="Consolidated weekly shopping list, ingredients merged across days")
    money_saving_tips: list[str] = Field(description="3 to 6 concrete, location-specific budgeting tips")
    nutrition_summary: str = Field(description="2-4 sentence summary of how the plan balances protein, carbs, vegetables and variety")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def format_money(amount: float, currency: str) -> str:
    """Format a numeric amount with sensible decimal precision per currency."""
    currency = currency.upper()
    if currency in NO_DECIMAL_CURRENCIES:
        return f"{amount:,.0f} {currency}"
    return f"{amount:,.2f} {currency}"


def guess_currency_for_country(country: str) -> str:
    return COUNTRY_CURRENCY_MAP.get(country.strip().lower(), DEFAULT_CURRENCY)


def _retry(fn, *args, max_retries: int = MAX_API_RETRIES, **kwargs):
    """Call fn with exponential backoff on transient Gemini API errors."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except genai_errors.APIError as exc:
            last_exc = exc
            status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            transient = status in (429, 500, 502, 503, 504) or status is None
            if not transient or attempt == max_retries:
                raise
            wait = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "Gemini API call failed (attempt %s/%s, status=%s). Retrying in %.1fs...",
                attempt, max_retries, status, wait,
            )
            time.sleep(wait)
    if last_exc:
        raise last_exc


# --------------------------------------------------------------------------- #
# Main agent
# --------------------------------------------------------------------------- #

class SmartBudgetNutritionAgent:
    """Orchestrates location detection, grounded price research, and
    structured budget meal-plan generation using the Gemini API."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY environment variable is not set. "
                "Get a key at https://aistudio.google.com/apikey and run:\n"
                "  export GEMINI_API_KEY='your-key-here'   (macOS/Linux)\n"
                "  setx GEMINI_API_KEY \"your-key-here\"      (Windows)"
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model or DEFAULT_MODEL
        logger.info("Initialized SmartBudgetNutritionAgent with model='%s'", self.model)

    # ----------------------------- Stage 1: LOCATE ----------------------------- #

    def detect_location(self) -> LocationInfo:
        """Detect the user's approximate city/country via free IP geolocation
        services. Falls back to a fixed default if every lookup fails."""
        for url in IP_GEOLOCATION_ENDPOINTS:
            try:
                resp = requests.get(url, timeout=5)
                resp.raise_for_status()
                data = resp.json()
                city = data.get("city") or data.get("city_name")
                country = data.get("country_name") or data.get("country")
                if country and len(country) <= 3:
                    # Some providers return an ISO code (e.g. 'VN'); fall back
                    # to a friendlier full name when possible.
                    country = data.get("country_name", country)
                if city and country:
                    logger.info("Location detected via %s: %s, %s", url, city, country)
                    return LocationInfo(city=city, country=country, source=url, raw=data)
            except (requests.RequestException, ValueError) as exc:
                logger.warning("Geolocation lookup failed for %s: %s", url, exc)
                continue
        logger.warning(
            "All geolocation lookups failed. Falling back to default location: %s, %s",
            FALLBACK_CITY, FALLBACK_COUNTRY,
        )
        return LocationInfo(city=FALLBACK_CITY, country=FALLBACK_COUNTRY, source="fallback_default")

    # -------------------------- Stage 2: GROUND (search) ------------------------ #

    def research_local_prices(self, location: LocationInfo, currency: str) -> GroundedPriceResearch:
        """Use Gemini's Google Search grounding tool to fetch current,
        real-world grocery prices for the detected location. This is the
        anti-hallucination step: the model is forced to cite live web data
        instead of guessing prices from its training data."""
        ingredients_str = ", ".join(STAPLE_INGREDIENTS)
        prompt = (
            f"You are a local market-research assistant. The user is a university "
            f"intern living in {location.city}, {location.country} on a tight budget.\n\n"
            f"Use Google Search to find current, realistic average retail prices "
            f"(in {currency}) for the following everyday grocery staples, as sold in "
            f"local wet markets, supermarkets, or convenience stores in {location.city}: "
            f"{ingredients_str}.\n\n"
            f"For each item report: the item name, a realistic price or price range, "
            f"the unit it is typically sold in (e.g. per kg, per dozen, per pack), and "
            f"which type of store it is commonly found in. Be concise and use a simple "
            f"list format. Prioritize the most recent information you can find."
        )

        google_search_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(tools=[google_search_tool])

        logger.info("Running grounded price search for %s...", location.label())
        response = _retry(
            self.client.models.generate_content,
            model=self.model,
            contents=prompt,
            config=config,
        )

        raw_text = (response.text or "").strip()
        citations: list[dict] = []
        search_queries: list[str] = []
        grounded = False

        candidate = response.candidates[0] if response.candidates else None
        metadata = getattr(candidate, "grounding_metadata", None) if candidate else None
        if metadata:
            grounded = True
            search_queries = list(getattr(metadata, "web_search_queries", None) or [])
            for chunk in getattr(metadata, "grounding_chunks", None) or []:
                web = getattr(chunk, "web", None)
                if web:
                    citations.append({"title": getattr(web, "title", ""), "url": getattr(web, "uri", "")})

        logger.info(
            "Grounded search complete. grounded=%s, queries=%s, citations=%d",
            grounded, search_queries, len(citations),
        )
        return GroundedPriceResearch(
            raw_text=raw_text, search_queries=search_queries, citations=citations, grounded=grounded
        )

    # -------------------------- Stage 3: PLAN (structured) ---------------------- #

    def _request_structured_plan(self, prompt: str) -> WeeklyBudgetPlan:
        """Single call to Gemini requesting a JSON response that conforms to
        WeeklyBudgetPlan. NOTE: Google Search grounding and JSON/schema mode
        cannot be used together in a single request for this model family, so
        this call deliberately omits the search tool -- the grounded price
        data was already gathered in `research_local_prices` and is injected
        into the prompt as plain-text context instead."""
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=WeeklyBudgetPlan,
            temperature=0.4,
        )
        response = _retry(
            self.client.models.generate_content,
            model=self.model,
            contents=prompt,
            config=config,
        )
        try:
            return WeeklyBudgetPlan.model_validate_json(response.text)
        except ValidationError as exc:
            logger.error("Structured output failed schema validation: %s", exc)
            raise

    def generate_meal_plan(
        self,
        location: LocationInfo,
        price_research: GroundedPriceResearch,
        weekly_budget: float,
        currency: str,
    ) -> WeeklyBudgetPlan:
        """Generate a 7-day budget meal plan as a validated WeeklyBudgetPlan,
        then run a self-correction loop if the model overshoots the budget."""
        base_prompt = (
            f"You are a registered-dietitian-minded budget meal planner helping a "
            f"university intern in {location.city}, {location.country}.\n\n"
            f"Their total grocery budget for the week is {weekly_budget} {currency}. "
            f"Do not exceed this budget. Optimize for nutritional variety (protein, "
            f"vegetables, complex carbs) and avoid relying mainly on instant noodles -- "
            f"use them at most once across the whole week, as a backup, not a staple.\n\n"
            f"Here is current, real, web-grounded local price research to use as your "
            f"source of truth for costs (do not invent prices that contradict this):\n"
            f"---\n{price_research.raw_text or 'No grounded data available; use conservative, realistic local estimates.'}\n---\n\n"
            f"Build a 7-day plan (Monday through Sunday) with breakfast, lunch and "
            f"dinner each day, a consolidated shopping list with estimated costs, "
            f"3 to 6 concrete money-saving tips specific to this location, and a short "
            f"nutrition summary. All monetary values must be in {currency}."
        )

        plan = self._request_structured_plan(base_prompt)
        plan = self._validate_and_correct_budget(plan, base_prompt, weekly_budget, currency)
        return plan

    def _validate_and_correct_budget(
        self,
        plan: WeeklyBudgetPlan,
        base_prompt: str,
        weekly_budget: float,
        currency: str,
    ) -> WeeklyBudgetPlan:
        """Cross-check the model's reported total against the sum of daily
        totals and against the stated budget. If it overshoots by more than
        BUDGET_TOLERANCE, ask the model to revise downward, up to
        MAX_PLAN_CORRECTION_ROUNDS times."""
        for round_num in range(1, MAX_PLAN_CORRECTION_ROUNDS + 1):
            actual_total = round(sum(day.daily_total_cost for day in plan.days), 2)
            over_budget_pct = (actual_total - weekly_budget) / weekly_budget if weekly_budget else 0
            logger.info(
                "Budget check (round %d): plan total=%s, budget=%s, over_by=%.1f%%",
                round_num, format_money(actual_total, currency), format_money(weekly_budget, currency),
                over_budget_pct * 100,
            )
            if over_budget_pct <= BUDGET_TOLERANCE:
                # Keep the model's own numbers consistent with our recomputed total.
                plan.estimated_weekly_total_cost = actual_total
                plan.estimated_savings = round(weekly_budget - actual_total, 2)
                return plan

            if round_num == MAX_PLAN_CORRECTION_ROUNDS:
                logger.warning(
                    "Plan still over budget after %d correction rounds; returning best effort.",
                    MAX_PLAN_CORRECTION_ROUNDS,
                )
                plan.estimated_weekly_total_cost = actual_total
                plan.estimated_savings = round(weekly_budget - actual_total, 2)
                return plan

            correction_prompt = (
                base_prompt
                + f"\n\nIMPORTANT CORRECTION NEEDED: your previous plan cost "
                  f"{format_money(actual_total, currency)}, which is "
                  f"{over_budget_pct * 100:.1f}% over the {format_money(weekly_budget, currency)} "
                  f"budget. Revise the full plan to fit within budget by substituting "
                  f"cheaper local staples, reducing portion sizes slightly, or reusing "
                  f"ingredients across days, while keeping it nutritionally balanced. "
                  f"Return the complete corrected 7-day JSON plan."
            )
            plan = self._request_structured_plan(correction_prompt)
        return plan

    # ------------------------------- Rendering ---------------------------------- #

    @staticmethod
    def render_markdown(plan: WeeklyBudgetPlan, price_research: GroundedPriceResearch) -> str:
        lines: list[str] = []
        lines.append(f"# 7-Day Budget Meal Plan -- {plan.location_city}, {plan.location_country}")
        lines.append("")
        lines.append(f"**Weekly budget:** {format_money(plan.weekly_budget, plan.currency)}  ")
        lines.append(f"**Estimated total cost:** {format_money(plan.estimated_weekly_total_cost, plan.currency)}  ")
        savings_label = "Savings" if plan.estimated_savings >= 0 else "Over budget"
        lines.append(f"**{savings_label}:** {format_money(abs(plan.estimated_savings), plan.currency)}  ")
        lines.append("")
        lines.append(f"> {plan.nutrition_summary}")
        lines.append("")

        for day in plan.days:
            lines.append(f"## Day {day.day_number} -- {day.day_label}")
            lines.append(f"_Daily total: {format_money(day.daily_total_cost, plan.currency)} | "
                          f"~{day.daily_total_calories} kcal_")
            lines.append("")
            for meal in day.meals:
                lines.append(
                    f"- **{meal.meal_type.title()}: {meal.dish_name}** "
                    f"({format_money(meal.estimated_cost, plan.currency)}, ~{meal.estimated_calories} kcal)"
                )
                lines.append(f"  - Ingredients: {', '.join(meal.ingredients)}")
            lines.append("")

        lines.append("## Consolidated Shopping List")
        lines.append("")
        lines.append("| Ingredient | Quantity | Est. Price | Note |")
        lines.append("|---|---|---|---|")
        for item in plan.shopping_list:
            lines.append(
                f"| {item.ingredient} | {item.quantity} | "
                f"{format_money(item.estimated_price, plan.currency)} | {item.note or '-'} |"
            )
        lines.append("")

        lines.append("## Money-Saving Tips")
        for tip in plan.money_saving_tips:
            lines.append(f"- {tip}")
        lines.append("")

        if price_research.grounded:
            lines.append("## Sources (Google Search Grounding)")
            for c in price_research.citations:
                title = c.get("title") or c.get("url")
                lines.append(f"- [{title}]({c.get('url')})")
            lines.append("")

        return "\n".join(lines)

    # --------------------------------- Orchestration ----------------------------- #

    def run(
        self,
        weekly_budget: float,
        currency: Optional[str] = None,
        location_override: Optional[str] = None,
        output_dir: str = "output",
    ) -> WeeklyBudgetPlan:
        """Run the full LOCATE -> GROUND -> PLAN pipeline and persist results."""
        if location_override:
            city, _, country = location_override.partition(",")
            location = LocationInfo(city=city.strip(), country=country.strip() or "Unknown", source="user_override")
        else:
            location = self.detect_location()

        resolved_currency = (currency or guess_currency_for_country(location.country)).upper()
        logger.info(
            "Pipeline starting | location=%s | budget=%s | currency=%s",
            location.label(), weekly_budget, resolved_currency,
        )

        price_research = self.research_local_prices(location, resolved_currency)
        plan = self.generate_meal_plan(location, price_research, weekly_budget, resolved_currency)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_path = out_dir / f"meal_plan_{timestamp}.json"
        md_path = out_dir / f"meal_plan_{timestamp}.md"
        research_path = out_dir / f"price_research_{timestamp}.txt"

        json_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        md_path.write_text(self.render_markdown(plan, price_research), encoding="utf-8")
        research_path.write_text(
            f"Search queries: {price_research.search_queries}\n\n{price_research.raw_text}",
            encoding="utf-8",
        )

        logger.info("Saved plan to %s and %s", json_path, md_path)
        return plan


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smart Budget Nutrition Assistant for Interns -- generates a "
                    "real-price-grounded 7-day budget meal plan."
    )
    parser.add_argument("--budget", type=float, required=True,
                        help="Weekly grocery budget in local currency, e.g. 500000 for VND or 50 for USD")
    parser.add_argument("--currency", type=str, default=None,
                        help="ISO currency code override, e.g. VND, USD, PHP. Auto-detected from location if omitted")
    parser.add_argument("--location", type=str, default=None,
                        help="Override detected location as 'City,Country', e.g. 'Hanoi,Vietnam'")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Gemini model id override (default: {DEFAULT_MODEL})")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to save the generated plan (default: ./output)")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        agent = SmartBudgetNutritionAgent(model=args.model)
        plan = agent.run(
            weekly_budget=args.budget,
            currency=args.currency,
            location_override=args.location,
            output_dir=args.output_dir,
        )
    except EnvironmentError as exc:
        print(f"\nConfiguration error: {exc}\n", file=sys.stderr)
        return 1
    except genai_errors.APIError as exc:
        print(f"\nGemini API error: {exc}\n", file=sys.stderr)
        return 1
    except ValidationError as exc:
        print(f"\nThe model's response did not match the expected schema: {exc}\n", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130

    print("\n" + "=" * 70)
    print(f"  PLAN READY: {plan.location_city}, {plan.location_country}")
    print(f"  Budget: {format_money(plan.weekly_budget, plan.currency)}  |  "
          f"Estimated cost: {format_money(plan.estimated_weekly_total_cost, plan.currency)}")
    print("=" * 70 + "\n")
    print(json.dumps(plan.model_dump(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
