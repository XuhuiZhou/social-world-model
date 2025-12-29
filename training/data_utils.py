"""Data processing utilities for fine-tuning socialized context generation.

CRITICAL: This module aligns with the actual `socialize_context()` implementation
in social_world_model.py (lines 187-208). It uses the EXACT template format and
data structures used during inference.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rich.progress import track
from sotopia.generation_utils import PydanticOutputParser
from social_world_model.database import SocializedContextForModel
from social_world_model.task_modules.tomi import TOMI_SOCIALIZED_CONTEXT_PROMPT


def load_socialized_contexts(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load all JSON files from the socialized context directory.

    Args:
        data_dir: Directory containing JSON files with socialized contexts

    Returns:
        List of valid records with story, question, answer, and socialized_context
    """
    data_path = Path(data_dir)
    records = []
    errors = []

    json_files = list(data_path.glob("*.json"))
    print(f"Found {len(json_files)} JSON files in {data_dir}")

    for json_file in track(json_files, description="Loading JSON files"):
        try:
            with open(json_file) as f:
                record = json.load(f)

            # Validate required keys
            required_keys = ["story", "question", "answer", "socialized_context"]
            if all(key in record for key in required_keys):
                records.append(record)
            else:
                missing = [k for k in required_keys if k not in record]
                errors.append(f"{json_file.name}: Missing keys {missing}")

        except json.JSONDecodeError as e:
            errors.append(f"{json_file.name}: JSON decode error - {e}")
        except Exception as e:
            errors.append(f"{json_file.name}: {e}")

    if errors:
        print(f"\nWarning: {len(errors)} files had errors (skipped):")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    print(f"\nSuccessfully loaded {len(records)} valid records")
    return records


def convert_dict_to_list_format(socialized_context_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert SocializedContext (dict format) to SocializedContextForModel (list format).

    This is the REVERSE of dictlize() in social_world_model/engine/utils.py.
    The model generates list format, which is then converted to dict format.
    For fine-tuning, we need to train on list format.

    Args:
        socialized_context_dict: SocializedContext with observations/actions as dicts

    Returns:
        SocializedContextForModel with observations/actions as lists

    Example:
        Input (dict format):
            {"observations": {"Mia": "text1", "Chloe": "text2"}}
        Output (list format):
            {"observations": ["Mia: text1", "Chloe: text2"]}
    """
    result = {
        "agents_names": socialized_context_dict["agents_names"],
        "socialized_context": []
    }

    for timestep in socialized_context_dict["socialized_context"]:
        # Convert observations dict → list
        observations_list = [
            f"{agent}: {obs}"
            for agent, obs in timestep["observations"].items()
        ]

        # Convert actions dict → list
        actions_list = [
            f"{agent}: {action}"
            for agent, action in timestep["actions"].items()
        ]

        converted_timestep = {
            "timestep": timestep["timestep"],
            "state": timestep["state"],
            "observations": observations_list,
            "actions": actions_list
        }

        result["socialized_context"].append(converted_timestep)

    return result


def create_training_format(record: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    """
    Convert a record to training format matching socialize_context() template.

    This function replicates the EXACT template used in social_world_model.py:165-208.
    The fine-tuned model must receive the same input format during training as it
    will during inference.

    Args:
        record: Dict with story, question, answer, and socialized_context

    Returns:
        Dict with 'messages' key containing user/assistant messages (NO system message)
    """
    # Parse story from string list format
    try:
        story = " ".join(eval(record["story"]))
    except Exception:
        # If eval fails, assume it's already a string
        story = record["story"]

    # Get format instructions from PydanticOutputParser
    # This returns the JSON schema that the model must follow
    parser = PydanticOutputParser(pydantic_object=SocializedContextForModel)
    format_instructions = parser.get_format_instructions()

    # Build template EXACTLY matching socialize_context() lines 187-208
    template = (
        "Please analyze the following narrative/context.\n\n"
        "#### Context: {context}\n\n"
    )

    # Add task-specific instructions (TOMI_SOCIALIZED_CONTEXT_PROMPT)
    template += "#### Task specific instructions: {task_specific_instructions}\n\n"

    # Add format instructions (JSON schema)
    template += "Follow these format instructions:\n{format_instructions}"

    # Fill template with actual values
    user_message = template.format(
        context=story,
        task_specific_instructions=TOMI_SOCIALIZED_CONTEXT_PROMPT,
        format_instructions=format_instructions
    )

    # Extract socialized_context and convert to list format
    # (SocializedContextForModel format - what the model generates)
    context_dict = {
        "agents_names": record["socialized_context"]["agents_names"],
        "socialized_context": record["socialized_context"]["socialized_context"]
    }

    # Convert dict format → list format (reverse of dictlize())
    context_list_format = convert_dict_to_list_format(context_dict)

    # Build messages: user (template) + assistant (list-format JSON)
    # NO system message - this matches actual agenerate() behavior
    messages = [
        {
            "role": "user",
            "content": user_message
        },
        {
            "role": "assistant",
            "content": json.dumps(context_list_format, indent=2)
        }
    ]

    return {"messages": messages}


def train_val_split(
    records: List[Dict[str, Any]],
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split records into train and validation sets.

    Args:
        records: List of data records
        val_ratio: Fraction of data to use for validation (default: 0.1)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (train_records, val_records)
    """
    # Shuffle with fixed seed for reproducibility
    rng = random.Random(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)

    # Calculate split point
    val_size = int(len(shuffled) * val_ratio)
    train_size = len(shuffled) - val_size

    train_records = shuffled[:train_size]
    val_records = shuffled[train_size:]

    print(f"Split: {len(train_records)} train, {len(val_records)} validation")
    return train_records, val_records


def save_to_jsonl(records: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Save records to JSONL format (one JSON per line).

    Args:
        records: List of data records to save
        output_path: Path to output JSONL file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for record in track(records, description=f"Writing {output_path.name}"):
            f.write(json.dumps(record) + '\n')

    print(f"Saved {len(records)} records to {output_path}")
