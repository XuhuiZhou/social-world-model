from social_world_model.social_world_model import SocialWorldModel
from typing import Any, Optional

TOMI_SOCIALIZED_CONTEXT_PROMPT = """You are dissecting the TOMI scenarios. The assumptions are that the characters can perceive every scene in their location but not scenes occurring elsewhere. If the agent leaves the location, they cannot perceive the scene in that location anymore. In the agent's observation, remember to include the objects' locations if the agents are in the same location as the object."""


def prepare_tomi_vanilla(
    row: dict[str, Any],
    pure_context: bool = False,
    with_reasoning: bool = True,
) -> tuple[str, dict[str, Any]]:
    try:
        story = " ".join(eval(row["story"]))
    except Exception as e:
        print(f"Error parsing story for index {row['index']}: {e}")
        story = row["story"]
    extra_info = row.get("extra_info", "")
    if extra_info:
        if pure_context:
            story = extra_info
            extra_info = ""

    question = row["question"]
    template = """Imagine that you are an observer in the scenario. Assume that the characters can perceive every scene in their location but not scenes occurring elsewhere. If something is being moved, that means it is not in its original location anymore. You should majorly focus on where the object has been moved to, and answer the question with the **most detailed position possible** e.g., the object is in A and A is in B, then you should answer 'A'. Provide your reasoning within the <reasoning></reasoning>tag. For the answer, use <answer>(put your answer here)</answer> and only include the most **detailed** location but not other information.

Below is the story and question (and optional extra information):
## Story
{story}

## Extra Information
(to help you better understand and answer the question)
{extra_info}

## Question
{question}"""

    if not with_reasoning:
        template = """Imagine that you are an observer in the scenario. Assume that the characters can perceive every scene in their location but not scenes occurring elsewhere. If something is being moved, that means it is not in its original location anymore. You should majorly focus on where the object has been moved to, and answer the question with the **most detailed position possible** e.g., the object is in A and A is in B, then you should answer 'A'. Use <answer>(put your answer here)</answer> and only include the most **detailed** location but not other information.

Below is the story and question (and optional extra information):
## Story
{story}

## Extra Information
(to help you better understand and answer the question)
{extra_info}

## Question
{question}"""

    if row["cands"]:
        template += "\n\nPossible answers: {candidates}"

    input_values = {
        "story": story,
        "question": question,
        "extra_info": extra_info,
        "candidates": ", ".join(eval(row["cands"])) if row["cands"] else "",
    }

    return template, input_values


def create_tomi_result(
    parsed_result: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    """Create ToMi result dictionary."""
    targeted_entries = [
        "story",
        "question",
        "reasoning",
        "answer",
        "correct_answer",
        "is_correct",
        "socialized_context",
        "extra_info",
    ]
    result = {}
    correct_answer = row["correct_answer"]
    answer = parsed_result["answer"]
    parsed_result["is_correct"] = correct_answer.lower() in answer.lower()
    for entry in targeted_entries:
        if entry in parsed_result:
            result[entry] = parsed_result[entry]
        elif entry in row:
            result[entry] = row[entry]
        else:
            continue
    return result


def tomi_evaluation_report(results: list[dict[str, Any]]) -> None:
    """Evaluate ToMi result."""
    correct_count = 0
    for result in results:
        if result["is_correct"]:
            correct_count += 1
    print(
        f"Current accuracy: {correct_count}/{len(results)} = {correct_count/len(results):.2%}"
    )


async def tomi_simulation(
    row: dict[str, Any], engine: Optional[SocialWorldModel] = None
) -> dict[str, Any]:
    assert engine is not None, "Engine must be provided"
    result = {
        "question": row["question"],
        "reasoning": "",
        "answer": "",
        "correct_answer": "",
        "is_correct": "",
        "socialized_context": "",
        "transformed_question": "",
        "memory": "",
        "agents": "",
    }
    row["socialized_context"] = ""
    row["extra_info"] = ""
    return result
