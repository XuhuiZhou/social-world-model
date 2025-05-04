import re
import pandas as pd
from social_world_model.social_world_model import SocialWorldModel
from typing import Any, Optional

HITOM_SOCIALIZED_CONTEXT_PROMPT = """You are tasked with generating socialized contexts for HITOM scenarios. Follow these assumptions closely: (1) Each agent observes all actions and object movements occurring in their current location until they exit it. After exiting, they no longer perceive events there. (2) An agent (A) can infer another agent's (B) mental state only if they have previously shared a location or have engaged in private or public interactions. (3) Agents frequently lie; therefore, an agent's true belief remains unaffected by what they tell others. (4) Agents know that private communications are not overheard by others, while public statements can be heard by anyone present. When generating observations for an agent, explicitly include the locations of objects if the agent shares the same location as those objects. Clearly distinguish internal thoughts, inferences, or beliefs using the <mental_state> tag."""


def reformat_hitom_data(data_list: dict[str, Any]) -> pd.DataFrame:
    # sample_id -> index
    # answer -> correct_answer
    data_list = data_list["data"]
    data = pd.DataFrame(data_list)
    data = data[data["prompting_type"] == "CoTP"]
    data["set_id"] = data.groupby("story", sort=False).ngroup()
    return data.rename(columns={"sample_id": "index", "answer": "correct_answer"})


def prepare_hitom_vanilla(
    row: dict[str, Any], pure_context: bool = False
) -> tuple[str, dict[str, Any]]:
    story = row["story"]
    extra_info = row.get("extra_info", "")
    if extra_info:
        if pure_context:
            story = extra_info
            extra_info = ""
        else:
            story = story + "\n" + extra_info

    question = row["question"] + "\n" + row["choices"]
    template = """You are analysing a social interaction and need to answer a question about it. The following story happens in chronological order. You will be given a multiple-choice question and a note at the end. You should assume the following: (1) An agent witnesses everything and every movements before exiting a location. (2) An agent A can infer another agent B's mental state only if A and B have been in the same location, or have private or public interactions. (3) Note that every agent tend to lie. What a character tells others doesn't affect his actual belief. (4) Agents in private communications know that others won't hear them, but they know that anyone can hear any public claims. First give step-by-step analysis about the question. Then output the answer. Provide your reasoning within the <reasoning></reasoning>tag. For the answer, use <answer>(put your answer here)</answer> and include only the letter corresponding to your choice but not other information.

Below is the story and question:
## Story
{story}
## Extra Information
(to help you better understand and answer the question)
{extra_info}
## Question
{question}"""

    input_values = {
        "story": story,
        "question": question,
    }

    return template, input_values


def evaluate_response(result: dict[str, Any]) -> dict[str, Any]:
    # dict(re.findall(r'([A-Z])\. ([^,]+)', choice_str))
    choices = result["choices"]
    answer = result["answer"].strip().capitalize()

    answer_dict = dict(re.findall(r"([A-Z])\. ([^,]+)", choices))
    answer_list = list(answer_dict.keys())

    if answer not in answer_list:
        result["is_correct"] = False
    else:
        if answer_dict[answer] == result["correct_answer"]:
            result["is_correct"] = True
        else:
            result["is_correct"] = False

    return result


def create_hitom_result(
    parsed_result: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    """Create ToMi result dictionary."""
    targeted_entries = [
        "set_id",
        "index",
        "deception",
        "story_length",
        "question_order",
        "story",
        "question",
        "reasoning",
        "answer",
        "correct_answer",
        "is_correct",
        "socialized_context",
        "extra_info",
        "choices",
    ]
    if not parsed_result:
        return {}
    result = {}
    for entry in targeted_entries:
        if entry in parsed_result:
            result[entry] = parsed_result[entry]
        elif entry in row:
            result[entry] = row[entry]
        else:
            continue
    result = evaluate_response(result)
    return result


def hitom_evaluation_report(results: list[dict[str, Any]]) -> None:
    """Evaluate ToMi result."""

    correct_count = 0
    for result in results:
        if result["is_correct"]:
            correct_count += 1

    print(
        f"Current accuracy: {correct_count}/{len(results)} = {correct_count/len(results):.2%}"
    )


def get_question_agent_names(question: str) -> list[str]:
    if question.startswith("Where is"):
        return []
    else:
        question = question.replace("Where does ", "")

        pattern = r"thinks?"
        parts = re.split(pattern, question)
        names = []
        for part in parts:
            name_match = re.search(r"[A-Z][a-z]+", part)
            if name_match:
                names.append(name_match.group())

        return names


def extract_integer(text: str) -> str:
    match = re.search(r"\d+", text)
    if match:
        return match.group()
    return "0"


def process_timestep(timestep: str) -> str:
    try:
        int(timestep)
        return str(timestep)
    except ValueError:
        extracted = extract_integer(timestep)
        return str(extracted)


async def hitom_simulation(
    row: dict[str, Any], engine: Optional[SocialWorldModel] = None
) -> dict[str, Any]:
    """Run experiment in simulation mode (using ToM engine for memory tracking)."""
    assert engine is not None, "Engine must be provided"
    result = {
        "question": row["question"],
        "reasoning": "",
        "answer": "",
        "correct_answer": row["correct_answer"],
        "socialized_context": "",
        "transformed_question": "",
        "memory": "",
        "agents": "",
    }
    row["socialized_context"] = ""
    row["extra_info"] = ""
    return result
