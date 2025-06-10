from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path to allow imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

"""Execute dietary queries with Arize Phoenix tracing.

Reads dietary queries from CSV, executes them with OpenAI 4o mini,
and logs full traces using Arize Phoenix.
"""

import argparse
import csv
import datetime as dt
import os
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Phoenix imports
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
import phoenix as px

# OpenAI and other imports
import openai
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

# Load environment variables
load_dotenv()

# Configuration
DEFAULT_CSV: Path = Path("data/dietary_queries.csv")
RESULTS_DIR: Path = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
MAX_WORKERS = 8  # Reduced for API rate limits
MODEL_NAME = "gpt-4o-mini"

# Set up Phoenix to use file-based storage
storage_path = Path.home() / ".phoenix" / "datasets"
storage_path.mkdir(parents=True, exist_ok=True)
os.environ["PHOENIX_WORKING_DIR"] = str(storage_path)

# Initialize Phoenix with persistent storage
px.launch_app()

# Register tracer and instrument OpenAI
tracer_provider = register()
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

# Get OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")


def build_system_prompt(dietary_preference: str) -> str:
    """Build system prompt based on dietary preference."""
    return  f"""
You are a polite, helpful, and attentive culinary assistant. Your primary role is to recommend recipes tailored specifically to the user's individual preferences, incorporating the following personal information unless the user explicitly provides conflicting instructions:

<personal_information>
  <dietary_preferences>{dietary_preference}</dietary_preferences>
  <date_of_request>Not given</date_of_request>
  <typically_makes_for>Not given</typically_makes_for>
  <skill_level>Not given</skill_level>
  <location>Not given</location>
</personal_information>

You will use this personal information in the following ways, unless explicitly directed otherwise by the user:
- Avoiding allergens completely in all recommended recipes (this must NEVER be compromised).
- Matching dietary preferences such as vegan, vegetarian, gluten-free, keto, etc.
- Suggesting recipes suitable for the date or season of the request.
- Adjusting recipes based on typical serving size requirements.
- Selecting recipes appropriate to the user's skill level.
- Considering the availability of ingredients in the user's geographic location.

You will NEVER suggest recipes containing ingredients to which the user is allergic, regardless of user instructions.

Each recipe recommendation must clearly follow the structure demonstrated below:

## RECIPE TITLE (ALL CAPS)

**Serves:** Number of servings
**Prep Time:** Preparation time
**Cook Time:** Cooking time

**Ingredients:**

- Main ingredients listed first, quantities included, clearly segmented by components (e.g., sauce, marinade, stir fry, etc.)
- Sub-components clearly labeled (e.g., **sauce**, **slurry**, **broccoli**)

**Steps:**

1. Step-by-step instructions clearly divided into separate tasks, each describing specific actions for each component.
2. Instructions should be concise, clear, and easy to follow.
3. Additional cooking tips, substitutes, or serving suggestions provided clearly as needed.

EXAMPLE:

## BEEF WITH BROCCOLI

**Serves:** 4
**Prep Time:** 15 min
**Cook Time:** 15 min

**Ingredients:**

- beef
- 1 lb beef skirt, flank, hanger, or flap, sliced for stir-fries (450 g)
- ½ tsp baking soda (2 g)
- ½ tsp kosher salt (1.5 g)
- 1 tsp light soy sauce or shoyu (5 ml)
- 1 tsp Shaoxing wine or dry sherry (5 ml)
- ½ tsp sugar (2 g)
- 1 tsp roasted sesame oil (5 ml)
- 1 tsp cornstarch (1.5 g)
- **sauce**
  - 1 Tbsp light soy sauce or shoyu (15 ml)
  - 1 Tbsp dark soy sauce (15 ml)
  - 3 Tbsp oyster sauce (45 ml)
  - 1 Tbsp sugar (12 g)
  - 2 Tbsp Shaoxing wine (30 ml)
- **slurry**
  - 2 tsp cornstarch (6 g)
  - 1 Tbsp water (15 ml)
- **broccoli**
  - 12 oz broccoli or broccolini, heads cut into bite-sized florets, stems peeled and cut on a bias into 1 1/2- to 2-inch segments
- **stir fry**
  - ¼ cups peanut, rice bran, or other neutral oil (60 ml)
  - 2 medium garlic cloves, minced
  - 2 tsp minced fresh ginger (5 g/about 1/2-inch segment)

**Steps:**

(Provide clear, concise, and numbered steps for preparation and cooking, exactly following the structured style demonstrated in the Beef with Broccoli example.)
"""

def process_query_with_tracing(
    query_id: str, query: str, dietary_preference: str
) -> Tuple[str, str, str, str]:
    """Process a single query with Phoenix tracing enabled."""
    try:
        # Create messages for OpenAI
        messages = [
            {"role": "system", "content": build_system_prompt(dietary_preference)},
            {"role": "user", "content": query}
        ]
        
        # Call OpenAI with tracing
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7,
            max_tokens=1500
        )
        
        # Extract response
        assistant_reply = response.choices[0].message.content.strip()
        
        return query_id, query, dietary_preference, assistant_reply
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        return query_id, query, dietary_preference, error_msg


def run_dietary_queries_with_tracing(csv_path: Path) -> None:
    """Execute dietary queries with Phoenix tracing."""
    console = Console()
    
    # Read CSV file
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        queries_data = []
        
        for idx, row in enumerate(reader, 1):
            if row.get("query") and row.get("dietary_preference"):
                queries_data.append({
                    "id": str(idx),
                    "query": row["query"],
                    "dietary_preference": row["dietary_preference"]
                })
    
    if not queries_data:
        raise ValueError("No valid queries found in CSV file")
    
    console.print(f"[bold blue]Found {len(queries_data)} queries to process[/bold blue]")
    console.print(f"[yellow]Phoenix UI available at: http://localhost:6006[/yellow]")
    
    results_data = []
    
    # Process queries with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(
            f"Processing {len(queries_data)} queries...", 
            total=len(queries_data)
        )
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_data = {
                executor.submit(
                    process_query_with_tracing,
                    item["id"],
                    item["query"],
                    item["dietary_preference"]
                ): item
                for item in queries_data
            }
            
            for future in as_completed(future_to_data):
                item_data = future_to_data[future]
                try:
                    query_id, query, dietary_pref, response = future.result()
                    results_data.append((query_id, query, dietary_pref, response))
                    
                    # Update progress
                    progress.update(task, advance=1)
                    
                    # Display result
                    panel_content = Text()
                    panel_content.append(f"ID: {query_id}\n", style="bold magenta")
                    panel_content.append(f"Dietary Preference: {dietary_pref}\n", style="bold green")
                    panel_content.append("Query: ", style="bold yellow")
                    panel_content.append(f"{query}\n\n")
                    
                    console.print(
                        Panel(
                            panel_content,
                            title=f"Query {query_id} - {dietary_pref}",
                            border_style="cyan"
                        )
                    )
                    
                except Exception as e:
                    console.print(
                        f"[red]Error processing query {item_data['id']}: {str(e)}[/red]"
                    )
                    results_data.append(
                        (item_data["id"], item_data["query"], 
                         item_data["dietary_preference"], f"Error: {str(e)}")
                    )
                    progress.update(task, advance=1)
    
    # Save results
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"dietary_queries_results_{timestamp}.csv"
    
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "query", "dietary_preference", "response"])
        writer.writerows(results_data)
    
    console.print(f"\n[bold green]Results saved to: {output_path}[/bold green]")
    console.print(f"[yellow]View traces at: http://localhost:6006[/yellow]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute dietary queries with Phoenix tracing"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to CSV file containing dietary queries"
    )
    args = parser.parse_args()
    
    run_dietary_queries_with_tracing(args.csv)
