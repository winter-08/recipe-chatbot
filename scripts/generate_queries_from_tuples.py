from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path to allow absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

"""Generate queries from tuples CSV file using LLM.

Reads a CSV file containing recipe-related tuples and generates natural
language queries using the agent to understand the context.
"""

import argparse
import csv
import datetime as dt
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from backend.utils import get_agent_response
import weave


weave.init("recipe-chatbot-queries")
# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------

DEFAULT_CSV: Path = Path("data/tuples.csv")
OUTPUT_DIR: Path = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_WORKERS = 8  # Fewer workers since we're making LLM calls

# Dimension descriptions for context
DIMENSION_DESCRIPTIONS = """
Recipe Chatbot User Dimensions:

1. Cooking Approach: The primary cooking method or style (e.g., Quick & Easy, Grilling, Instant Pot, Sous-vide, Baking, Everyday Cooking, Slow Cooker, Sautéing, Gourmet & Intricate, Minimal/No-Cook, Air Fryer, Roasting, Steaming)

2. Persona: The user's identity or role (e.g., College Student, Busy Parent, Health Enthusiast, Amateur Foodie, Comfort-Food Lover, Budget-Conscious Cook, Adventure Seeker)

3. Dietary Requirements: Specific dietary needs or restrictions (e.g., Vegan, Gluten-Free, High-Protein, Keto, Nut-Free, Low-Sodium, Halal, Dairy-Free, Paleo, Vegetarian, Low-Sugar, Kosher, Seafood-Free, Diabetic-Friendly, Egg-Free, Calorie-Conscious)

4. Cooks For: Who they're preparing meals for (e.g., Individual, Family-friendly, Meal Prep & Batch Cooking, Couple/Date Night, Large Groups/Parties)

5. Skill Level: Cooking expertise (e.g., Absolute Beginner, Comfortable Home Cook, Culinary Expert)

6. Cuisine: Type of cuisine preference (e.g., Mexican, Mediterranean, Indian, French, American, Italian, Middle Eastern, Japanese, Thai, Korean, Chinese, Southern, Traditional/Authentic, Fusion & Experimental)

7. Preparation Constraints: Limitations or requirements for cooking (e.g., Pantry Staples, Quick Meals, Seasonal Ingredients, Fully Equipped Kitchen, Leisurely Cooking, Limited Gear, Locally Available)

8. Scenario: The context or goal for cooking (e.g., Budget-Conscious, Health & Wellness Goals, Taste-Driven, Splurge-Worthy, Occasion-Based Cooking, Eco & Ethical Considerations, Detailed Guidance, Quick Summaries, Interactive Guidance)
"""

# -----------------------------------------------------------------------------
# Core logic
# -----------------------------------------------------------------------------


def generate_query_with_llm(row: Dict[str, str], query_id: int) -> Dict[str, str]:
    """Generate a natural language query using the LLM."""

    prompt = f"""{DIMENSION_DESCRIPTIONS}

Given the following user profile, generate a natural, conversational query that someone with these characteristics would ask a recipe chatbot. The query should feel authentic and incorporate 2-4 of the most relevant dimensions naturally.

User Profile:
- Cooking Approach: {row["Cooking Approach"]}
- Persona: {row["Persona"]}
- Dietary Requirements: {row["Dietary Requirements"]}
- Cooks For: {row["Cooks For"]}
- Skill Level: {row["Skill Level"]}
- Cuisine: {row["Cuisine"]}
- Preparation Constraints: {row["Preparation Constraints"]}
- Scenario: {row["Scenario"]}

Here are some example queries for inspiration (vary the style and tone):
- what's a stir-fry i can make with chipolatas and frozen veg
- microwave-only vegan dinner ideas please
- I've got 20 minutes and a half-bag of spinach, help
- need a cheap lunch for two days, no pasta, i'm sick of pasta
- How do I sear scallops so they don't turn to rubber
- slow-cooker recipe that won't bore me to death
- kid-friendly meal with chicken thighs, nothing spicy
- batch-cook five high-protein lunches for under a tenner
- can i air-fry churros without blowing my sugar budget
- Simple ramen upgrade with things I can get at Lidl
- date-night pasta, looks fancy, idiot-proof steps
- one-pan vegetarian dinner, minimal washing-up
- gluten-free brownies that still taste like brownies
- instant-pot stew, throw everything in and forget it
- i've got tofu, broccoli, and zero patience, suggestions
- Party snacks for eight, budget is tight
- something spicy with chickpeas, done in 15
- sous-vide steak finish, cast-iron only, show me
- make-ahead breakfast burritos that won't go soggy
- dairy-free mac and cheese that doesn't suck

Generate ONLY the query itself, no explanation or additional text. Make it sound like a real person asking for help with cooking."""

    messages = [{"role": "user", "content": prompt}]

    # Use get_agent_response to generate the query
    response_history = get_agent_response(messages, {})

    # Extract the generated query
    query = ""
    if response_history and response_history[-1]["role"] == "assistant":
        query = response_history[-1]["content"].strip()

    # Create result with all data
    result = {"id": str(query_id), "query": query}

    # Add all tuple fields as additional columns
    for key, value in row.items():
        result[key] = value

    return result


def generate_queries(csv_path: Path) -> None:
    """Main entry point for query generation."""
    console = Console()

    # Read the tuples CSV
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        tuples_data = list(reader)

    if not tuples_data:
        raise ValueError("No data found in the provided CSV file.")

    console.print(
        f"[bold blue]Generating queries from {len(tuples_data)} tuples using LLM...[/bold blue]"
    )

    results_data: List[Dict[str, str]] = []

    # Use ThreadPoolExecutor for concurrent LLM calls
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_row = {
            executor.submit(generate_query_with_llm, row, idx): (row, idx)
            for idx, row in enumerate(tuples_data, 1)
        }

        console.print(
            f"[bold blue]Submitting {len(tuples_data)} queries to LLM...[/bold blue]"
        )

        for i, future in enumerate(as_completed(future_to_row)):
            row, idx = future_to_row[future]
            try:
                query_data = future.result()
                results_data.append(query_data)

                # Display the generated query
                panel_content = Text()
                panel_content.append(f"ID: {query_data['id']}\n", style="bold magenta")
                panel_content.append("Query:\n", style="bold yellow")
                panel_content.append(f"{query_data['query']}\n\n")
                panel_content.append(
                    f"Scenario: {query_data.get('Scenario', 'N/A')}\n", style="dim"
                )
                panel_content.append(
                    f"Persona: {query_data.get('Persona', 'N/A')}", style="dim"
                )

                console.print(
                    Panel(
                        panel_content,
                        title=f"Generated Query {i + 1}/{len(tuples_data)}",
                        border_style="cyan",
                    )
                )

            except Exception as exc:
                console.print(
                    Panel(
                        f"[bold red]Exception for tuple {idx}:[/bold red]\n{exc}",
                        title=f"Error in Query {i + 1}/{len(tuples_data)}",
                        border_style="red",
                    )
                )

    console.print("[bold blue]All queries generated.[/bold blue]")

    # Save to output CSV
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"generated_queries_{timestamp}.csv"

    # Get all unique keys from results
    all_keys = set()
    for q in results_data:
        all_keys.update(q.keys())

    # Ensure 'id' and 'query' come first
    fieldnames = ["id", "query"] + sorted(
        [k for k in all_keys if k not in ["id", "query"]]
    )

    with out_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_data)

    console.print(
        f"[bold green]Generated {len(results_data)} queries and saved to {str(out_path)}[/bold green]"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate queries from recipe tuples using LLM"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to CSV file containing tuples.",
    )
    args = parser.parse_args()
    generate_queries(args.csv)
