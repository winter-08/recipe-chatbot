from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path to allow absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

"""Generate queries with dietary preferences using OpenAI's structured output.

This script generates natural language queries paired with dietary preferences
from a predefined list, consolidating all vegetarian types into one group.
"""

import json
import os
from typing import List, Dict, Any
from pydantic import BaseModel
import pandas as pd
from litellm import completion
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm
import random

load_dotenv()

# --- Pydantic Models for Structured Output ---
class DietaryQuery(BaseModel):
    query: str
    dietary_preference: str

class QueriesList(BaseModel):
    queries: List[DietaryQuery]

# --- Configuration ---
MODEL_NAME = "gpt-4o-mini"
NUM_QUERIES_PER_PREFERENCE = 5  # Number of queries to generate per dietary preference
OUTPUT_CSV_PATH = Path(__file__).parent.parent / "data" / "dietary_queries.csv"
MAX_WORKERS = 5  # Number of parallel LLM calls

# Dietary preferences list - consolidating vegetarians
DIETARY_PREFERENCES = [
    "Vegan",
    "Vegetarian",  # Consolidates: Lacto-ovo, Lacto, Ovo, and Jain Vegetarian
    "Pescetarian",
    "Halal",
    "Kosher",
    "Gluten-Free",
    "Wheat-Free",
    "Dairy-Free / Lactose-Free",
    "Nut-Free (Tree Nuts)",
    "Peanut Allergy",
    "Soy Allergy",
    "Fish-Free",
    "Shellfish Allergy",
    "Sesame Allergy",
    "Allium-Free",
    "Ketogenic (Keto)",
    "Mediterranean Diet",
    "Diabetic / Low-Carb",
    "Low-Sodium"
]

def call_llm(messages: List[Dict[str, str]], response_format: Any) -> Any:
    """Make a single LLM call with retries."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = completion(
                model=MODEL_NAME,
                messages=messages,
                response_format=response_format
            )
            return response_format(**json.loads(response.choices[0].message.content))
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(1)  # Wait before retry

def generate_queries_for_preference(dietary_preference: str) -> List[DietaryQuery]:
    """Generate natural language queries for a given dietary preference."""
    prompt = f"""Generate {NUM_QUERIES_PER_PREFERENCE} different natural language queries that someone with a "{dietary_preference}" dietary preference might ask a recipe chatbot.

The queries should:
1. Sound like real users asking for recipe help
2. Naturally incorporate the dietary preference (but not always explicitly mention it)
3. Vary in style, complexity, and detail level
4. Be realistic and practical
5. Include natural variations in typing style, such as:
   - Some queries in all lowercase
   - Some with casual language
   - Some with common typos
   - Some formal, some informal
   - Some with emojis or text speak
   - Some asking for specific meals (breakfast, lunch, dinner, snacks)
   - Some mentioning ingredients to use or avoid
   - Some mentioning time constraints
   - Some asking for cuisine types

Examples for a Vegan preference:
- "need a quick vegan dinner recipe"
- "What can I make with chickpeas and quinoa?"
- "plant based breakfast ideas pls!! 🌱"
- "Easy meal prep recipes for the week"
- "thai curry recipe without fish sauce"

Examples for a Gluten-Free preference:
- "gluten free pizza dough recipe"
- "What can I bake without wheat flour?"
- "need GF pasta dishes for dinner party"
- "is rice safe for celiac?"
- "Quick breakfast ideas no gluten"

Generate {NUM_QUERIES_PER_PREFERENCE} unique queries that someone with "{dietary_preference}" dietary needs might ask. Make them varied and natural."""

    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = call_llm(messages, QueriesList)
        # Ensure all queries have the correct dietary preference
        for query in response.queries:
            query.dietary_preference = dietary_preference
        return response.queries
    except Exception as e:
        print(f"Error generating queries for {dietary_preference}: {e}")
        return []

def generate_all_queries() -> List[DietaryQuery]:
    """Generate queries for all dietary preferences in parallel."""
    all_queries = []
    
    print(f"Generating {NUM_QUERIES_PER_PREFERENCE} queries each for {len(DIETARY_PREFERENCES)} dietary preferences...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all query generation tasks
        future_to_pref = {
            executor.submit(generate_queries_for_preference, pref): pref 
            for pref in DIETARY_PREFERENCES
        }
        
        # Process completed generations as they finish
        with tqdm(total=len(DIETARY_PREFERENCES), desc="Generating Queries") as pbar:
            for future in as_completed(future_to_pref):
                pref = future_to_pref[future]
                try:
                    queries = future.result()
                    if queries:
                        all_queries.extend(queries)
                    pbar.update(1)
                except Exception as e:
                    print(f"Preference '{pref}' generated an exception: {e}")
                    pbar.update(1)
    
    # Shuffle queries to mix dietary preferences
    random.shuffle(all_queries)
    
    return all_queries

def save_queries_to_csv(queries: List[DietaryQuery]):
    """Save generated queries to CSV."""
    if not queries:
        print("No queries to save.")
        return

    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'query': q.query,
            'dietary_preference': q.dietary_preference
        }
        for q in queries
    ])
    
    # Create output directory if it doesn't exist
    OUTPUT_CSV_PATH.parent.mkdir(exist_ok=True)
    
    # Save to CSV
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved {len(queries)} queries to {OUTPUT_CSV_PATH}")

def main():
    """Main function to generate and save queries."""
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it using: export OPENAI_API_KEY='your-api-key'")
        return

    start_time = time.time()
    
    # Generate queries for all dietary preferences
    queries = generate_all_queries()
    
    if queries:
        save_queries_to_csv(queries)
        elapsed_time = time.time() - start_time
        print(f"\nQuery generation completed successfully in {elapsed_time:.2f} seconds.")
        print(f"Generated {len(queries)} total queries across {len(DIETARY_PREFERENCES)} dietary preferences.")
        print(f"Average: {len(queries) / len(DIETARY_PREFERENCES):.1f} queries per preference")
    else:
        print("Failed to generate any queries.")

if __name__ == "__main__":
    main()
