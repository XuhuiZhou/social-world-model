from social_world_model.social_world_model import SocialWorldModel
from sotopia.generation_utils import StrOutputParser, agenerate
from typing import Any, Optional


EXPLOREToM_SOCIALIZED_CONTEXT_PROMPT = """You are tasked with generating a detailed socialized context for the provided story, clearly indicating each agent's observations, private or public interactions, internal mental states, and movements between locations."""


def prepare_exploretom_vanilla(
    row: dict[str, Any], pure_context: bool = False
) -> tuple[str, dict[str, Any]]:
    row["extra_info"] = row.get("extra_info", "")
    if row["extra_info"]:
        if pure_context:
            row["story"] = row["extra_info"]
            row["extra_info"] = ""
        else:
            pass

    template = """Imagine you are an observer in the scenario. Answer the question based solely on the story. First give step-by-step analysis about the question. Then output the answer. Provide your reasoning within the <reasoning></reasoning>tag. For the answer, use <answer>(put your answer here)</answer>.

Below is the story and question (and optional extra information):
## Story
{story}

## Extra Information
(to help you better understand the memory and answer the question)
{extra_info}

## Question
{question}"""

    input_values = {
        "story": row["story"],
        "extra_info": row["extra_info"],
        "question": row["question"],
    }
    return template, input_values


async def evaluate_response(
    result: dict[str, Any], model_name: str = "gpt-4.1-2025-04-14"
) -> dict[str, Any]:
    """Evaluate the response using LLM-as-a-judge."""
    if result["answer"].strip().lower() == result["correct_answer"].strip().lower():
        result["is_correct"] = True
        return result

    template = """
    Evaluate whether the model's answer aligns with the correct answer. Respond with "yes" or "no" only.

    Question:
    {question}

    Model's Answer:
    {answer}

    Correct Answer:
    {correct_answer}

    Only respond with "yes" or "no". Do not include any explanations or additional text.
    """

    input_values = {
        "question": result["question"],
        "answer": result["answer"],
        "correct_answer": result["correct_answer"],
    }

    response = await agenerate(
        model_name=model_name,
        template=template,
        input_values=input_values,
        temperature=0.0,
        output_parser=StrOutputParser(),
        structured_output=False,
    )
    result["is_correct"] = "yes" in response.strip().lower()
    return result


async def create_exploretom_result(
    parsed_result: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    """Create ExploreToM result dictionary."""

    targeted_entries = [
        "set_id",
        "index",
        "story",
        "question",
        "reasoning",
        "answer",
        "is_correct",
        "socialized_context",
        "correct_answer",
        "extra_info",
    ]
    if not parsed_result:
        return {}
    result = {}
    for entry in targeted_entries:
        if entry in parsed_result:
            result[entry] = parsed_result[entry]
        elif entry in row:
            result[entry] = row[entry]
    result = await evaluate_response(result)
    return result


def exploretom_evaluation_report(results: list[dict[str, Any]]) -> None:
    """Evaluate ExploreToM results."""
    correct_count = 0
    for result in results:
        if result["is_correct"]:
            correct_count += 1
    print(
        f"Current accuracy: {correct_count}/{len(results)} = {correct_count/len(results):.2%}"
    )


def exploretom_simulation(
    row: dict[str, Any], engine: Optional[SocialWorldModel] = None
) -> dict[str, Any]:
    pass
