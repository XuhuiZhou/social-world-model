from typing import Any, Optional
from social_world_model.social_world_model import SocialWorldModel

COBRA_FRAMES_SOCIALIZED_CONTEXT_PROMPT = """"""


def create_cobra_frames_result(
    parsed_result: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    """Create COBRA frames result dictionary."""
    targeted_entries = [
        "index",
        "scenario",
        "listener",
        "speaker",
        "statement",
        "offensive_allapp",
        "reasoning",
        "answer",
        "correct",
    ]
    parsed_result["offensive_allapp"] = row["offensive_allapp"]
    if "answer" in parsed_result:
        parsed_result = evaluate_cobra_frames(parsed_result)
    result = {}
    for entry in targeted_entries:
        if entry in parsed_result:
            result[entry] = parsed_result[entry]
        elif entry in row:
            result[entry] = row[entry]
        else:
            continue
    return result


def prepare_cobra_frames_vanilla(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    template = """
## Context
{context}

## Extra Info
(to help you better understand the scenario)
{extra_info}

## Task
Analyze whether the statement is offensive in this context
Format your response with:
1. Reasoning in <reasoning>...</reasoning> tags
2. Final answer (0 for not offensive, 1 for offensive) in <answer>...</answer> tags
"""
    input_values = {"context": row["context"], "extra_info": row["statement"]}
    return template, input_values


def evaluate_cobra_frames(row: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the COBRA frames prediction against the ground truth."""
    predicted = int(row["answer"])
    ground_truth = int(row["offensive_allapp"])
    row["correct"] = 1 if predicted == ground_truth else 0
    return row


def cobra_frames_evaluation_report(results: list[dict[str, Any]]) -> None:
    """Generate evaluation report for COBRA frames task."""
    total = len(results)
    correct = sum(result["correct"] for result in results)
    accuracy = correct / total if total > 0 else 0

    print("\nCOBRA Frames Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

    # Additional metrics could be added here
    # For example: precision, recall, F1 score for offensive vs non-offensive classification


async def cobra_frames_simulation(
    row: dict[str, Any], engine: Optional[SocialWorldModel] = None
) -> dict[str, Any]:
    """Run simulation for COBRA frames task."""
    assert engine is not None, "Engine must be provided"
    return row
