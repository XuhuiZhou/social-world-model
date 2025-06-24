import math
import re
from typing import Any, Optional, Union

from social_world_model.social_world_model import SocialWorldModel

DIAMONDS_SOCIALIZED_CONTEXT_PROMPT = """For the DIAMONDs dataset, analyze the conversation. Note that if someone left the conversation, they would not be able to observe the utterances that happened after they left until they return. (Use none for the observations of the agents that left the conversation)"""


def create_diamonds_result(
    parsed_result: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    """Create DIAMONDs result dictionary."""
    # Extract the answer from the parsed result
    answer = parsed_result.get("answer", "")

    # Try to convert the answer to a float for evaluation
    try:
        pred_ans = float(answer)
    except ValueError:
        pred_ans = 0.0

    # Evaluate the answer
    true_ans = row.get("answer", None)
    
    try:
        accuracy = (
            1
            if (
                isinstance(pred_ans, (int, float))
                and isinstance(true_ans, (int, float))
                and math.isclose(true_ans, pred_ans, rel_tol=2e-2)
            )
            else 0
        )
    except (TypeError, ValueError):
        accuracy = 0

    # Create the result dictionary
    targeted_entries = [
        "id",
        "qa_type",
        "context",
        "final_question",
        "answer",
        "conv_access_grp",
        "participant",
        "reasoning",
        "socialized_context",
        "extra_info",
    ]

    result = {
        "pred_ans": pred_ans,
        "true_ans": true_ans,
        "accuracy": accuracy,
    }

    for entry in targeted_entries:
        if entry in parsed_result:
            result[entry] = parsed_result[entry]
        elif entry in row:
            result[entry] = row[entry]

    return result


def prepare_diamonds_vanilla(
    row: dict[str, Any], pure_context: bool = False
) -> tuple[str, dict[str, Any]]:
    """Prepare DIAMONDs data for vanilla mode."""
    extra_info = row.get("extra_info", "")

    # Handle both old segmented format and new flattened format
    if isinstance(row["context"], dict):
        conversation_text = format_conversation(row["context"])
    else:
        conversation_text = row["context"]
    

    # Determine context based on pure_context flag
    if extra_info and pure_context:
        context = extra_info
        extra_info = ""
    else:
        context = conversation_text

    # Create the template
    template = """
You are an assistant helping to answer questions based on a conversation.

## Conversation
{context}

## Extra Info
(to help you better understand the conversation)
{extra_info}

## Task
{final_question}

Format your response as follows:
<reasoning>
Provide your step-by-step reasoning here.
</reasoning>

<answer>
Your final numerical answer here (only numbers that can be converted to float).
</answer>
"""

    input_values = {
        "context": context,
        "extra_info": extra_info,
        "final_question": row["final_question"],
        "participant": row["participant"],
    }

    return template, input_values


def format_conversation(conversation_data: Union[dict[str, Any], str]) -> str:
    """Format the conversation data into a readable text format."""
    # If conversation_data is already a string (flattened), return it directly
    if isinstance(conversation_data, str):
        return conversation_data
    
    # Handle legacy format with segments (for backward compatibility)
    formatted_text = ""
    if "conversation" in conversation_data:
        for utterance_list in conversation_data["conversation"]:
            for utterance in utterance_list:
                for speaker, message in utterance.items():
                    formatted_text += f"{speaker}: {message}\n"

    return formatted_text


def diamonds_evaluation_report(results: list[dict[str, Any]]) -> None:
    """Generate evaluation report for DIAMONDs dataset."""
    total = len(results)
    correct = sum(result["accuracy"] for result in results)

    # Calculate accuracy by dataset type
    dataset_types = {}
    for result in results:
        dataset_type = (
            result.get("id", "").split("_")[0]
            if "_" in result.get("id", "")
            else "base"
        )
        if dataset_type not in dataset_types:
            dataset_types[dataset_type] = {"total": 0, "correct": 0}

        dataset_types[dataset_type]["total"] += 1
        dataset_types[dataset_type]["correct"] += result["accuracy"]

    # Calculate answerability accuracy
    answerability_correct = sum(
        1
        for result in results
        if result.get("pred_ans_type") == result.get("true_ans_type")
    )

    # Print the report
    print("\nDIAMONDs Evaluation Results:")
    print(f"Overall Accuracy: {correct/total:.4f} ({correct}/{total})")

    print("\nAccuracy by Dataset Type:")
    for dataset_type, counts in dataset_types.items():
        print(
            f"  {dataset_type}: {counts['correct']/counts['total']:.4f} ({counts['correct']}/{counts['total']})"
        )

    print(
        f"\nAnswerability Accuracy: {answerability_correct/total:.4f} ({answerability_correct}/{total})"
    )


async def diamonds_simulation(
    row: dict[str, Any], engine: Optional[SocialWorldModel] = None
) -> dict[str, Any]:
    """Run DIAMONDs simulation."""
    assert engine is not None, "Engine must be provided"

    # For now, we'll just return the row as is
    # In a more complex implementation, we could simulate the conversation
    return row
