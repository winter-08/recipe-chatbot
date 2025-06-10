from __future__ import annotations

"""Utility helpers for the recipe chatbot backend.

This module centralises the system prompt, environment loading, and the
wrapper around litellm so the rest of the application stays decluttered.
"""

import os
from typing import Final, List, Dict
import datetime as dt

import litellm  # type: ignore
from dotenv import load_dotenv

# Ensure the .env file is loaded as early as possible.
load_dotenv(override=False)

# --- Constants -------------------------------------------------------------------
SYSTEM_PROMPT_TMPL: Final[str] = """
You are a polite, helpful, and attentive culinary assistant. Your primary role is to recommend recipes tailored specifically to the user's individual preferences, incorporating the following personal information unless the user explicitly provides conflicting instructions:

<personal_information>
  <allergic_to>{allergic_to}</allergic_to>
  <dietary_preferences>{dietary_preferences}</dietary_preferences>
  <date_of_request>{date}</date_of_request>
  <typically_makes_for>{typically_makes_for}</typically_makes_for>
  <skill_level>{skill_level}</skill_level>
  <location>{location}</location>
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


def build_system_prompt(user_info: Dict[str, Any]) -> str:
    defaults = {
        "allergic_to": "none",
        "dietary_preferences": "none",
        "date": dt.date.today().isoformat(),
        "typically_makes_for": "1",
        "skill_level": "beginner",
        "location": "London, UK",
    }

    data = {**defaults, **user_info}
    return SYSTEM_PROMPT_TMPL.format(**data)

# Fetch configuration *after* we loaded the .env file.
MODEL_NAME: Final[str] = (
    Path.cwd().with_suffix("")  # noqa: WPS432  # dummy call to satisfy linters about unused Path
    and (  # noqa: W504 line break for readability
        __import__("os").environ.get("MODEL_NAME", "gpt-4.1-nano")
    )
)


# --- Agent wrapper ---------------------------------------------------------------


def get_agent_response(
    messages: List[Dict[str, str]], user_info: Dict[str, Any]
) -> List[Dict[str, str]]:  # noqa: WPS231
    """Call the underlying large-language model via *litellm*.

    Parameters
    ----------
    messages:
        The full conversation history. Each item is a dict with "role" and "content".

    Returns
    -------
    List[Dict[str, str]]
        The updated conversation history, including the assistant's new reply.
    """

    # litellm is model-agnostic; we only need to supply the model name and key.
    # The first message is assumed to be the system prompt if not explicitly provided
    # or if the history is empty. We'll ensure the system prompt is always first.
    system_prompt = build_system_prompt(user_info)

    current_messages: List[Dict[str, str]]
    if not messages or messages[0]["role"] != "system":
        current_messages = [{"role": "system", "content": system_prompt}] + messages
    else:
        current_messages = messages

    completion = litellm.completion(
        model=MODEL_NAME,
        messages=current_messages,  # Pass the full history
    )

    assistant_reply_content: str = completion["choices"][0]["message"][
        "content"
    ].strip()  # type: ignore[index]

    # Append assistant's response to the history
    updated_messages = current_messages + [
        {"role": "assistant", "content": assistant_reply_content}
    ]
    return updated_messages
