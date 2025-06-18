import math
from typing import Any, Optional, Union

from social_world_model.social_world_model import SocialWorldModel

DIAMONDS_SOCIALIZED_CONTEXT_PROMPT = """For the DIAMONDs dataset, analyze the conversation to understand what information each participant has access to. 
Pay special attention to:
1. Who is present in each part of the conversation
2. What numerical information is shared
3. How information changes throughout the conversation (e.g., price increases, quantity changes)
4. What calculations each participant would be able to perform based on their access to information"""


def create_diamonds_result(
    parsed_result: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    """Create DIAMONDs result dictionary."""
    # Extract the answer from the parsed result
    answer = parsed_result.get("answer", "")

    # Try to convert the answer to a float for evaluation
    pred_ans: Union[float, str]
    try:
        if answer.lower() == "na":
            pred_ans = "NA"
        else:
            # Extract numeric value from the answer if it contains text
            import re

            numeric_match = re.search(r"[-+]?\d*\.\d+|\d+", answer)
            if numeric_match:
                pred_ans = float(numeric_match.group())
            else:
                pred_ans = float(answer)
    except (ValueError, AttributeError):
        pred_ans = "NA"  # Default to NA if conversion fails

    # Evaluate the answer
    true_ans = row.get("answer", None)
    if true_ans == "NA":
        accuracy = 1 if pred_ans == "NA" else 0
        pred_ans_type = "NA"
        true_ans_type = "NA"
    elif pred_ans == "NA":
        accuracy = 0
        pred_ans_type = "NA"
        true_ans_type = "answerable"
    else:
        # Use the 2% tolerance as mentioned in the README
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
        pred_ans_type = "answerable"
        true_ans_type = "answerable"

    # Create the result dictionary
    targeted_entries = [
        "id",
        "qa_type",
        "conversation",
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
        "pred_ans_type": pred_ans_type,
        "true_ans_type": true_ans_type,
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

    # Get the conversation from the row
    if isinstance(row["conversation"], dict):
        conversation_text = format_conversation(row["conversation"])
    else:
        conversation_text = row["conversation"]

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
You need to answer the following question based on the conversation, from the perspective of {participant}.
{final_question}

Remember that {participant} only has access to the information they were present for in the conversation.

Format your response as follows:
<reasoning>
Provide your step-by-step reasoning here. Consider what information {participant} has access to.
</reasoning>

<answer>
Your final numerical answer here. If the question cannot be answered based on the information available to {participant}, respond with "NA".
</answer>
"""

    input_values = {
        "context": context,
        "extra_info": extra_info,
        "final_question": row["final_question"],
        "participant": row["participant"],
    }

    return template, input_values


def format_conversation(conversation_data: dict[str, Any]) -> str:
    """Format the conversation data into a readable text format."""
    formatted_text = ""

    if "conversation" in conversation_data:
        for i, segment in enumerate(conversation_data["conversation"]):
            formatted_text += f"Segment {i+1}:\n"
            for utterance in segment:
                for speaker, message in utterance.items():
                    formatted_text += f"{speaker}: {message}\n"
            formatted_text += "\n"

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
