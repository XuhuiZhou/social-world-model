from social_world_model.social_world_model import SocialWorldModel
from typing import Any, Optional

LIFECHOICES_SOCIALIZED_CONTEXT_PROMPT = """You are dissecting the LifeChoices scenarios. You should assume the following: (1) An agent witnesses everything and every movements before exiting a location. (2) An agent A can infer another agent B's mental state only if A and B have been in the same location, or have private or public interactions. (3) Note that every agent tend to lie. What a character tells others doesn't affect his actual belief. (4) Agents in private communications know that others won't hear them, but they know that anyone can hear any public claims. In the agent's observation, remember to include the objects' locations if the agents are in the same location as the object."""


def reformat_lifechoices_data(data_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result_data = []
    index = 0
    set_id = 0
    for book_data in data_list:
        for character_data in book_data:
            character_data["set_id"] = set_id
            character_data["index"] = index

            result_data.append(character_data)
            index += 1

        set_id += 1
    return result_data


def prepare_lifechoices_vanilla(
    row: dict[str, Any], pure_context: bool = False
) -> tuple[str, dict[str, Any]]:
    row["extra_info"] = row.get("extra_info", "")
    if row["extra_info"]:
        if pure_context:
            row["input_text"] = row["extra_info"]
            row["extra_info"] = ""
        else:
            pass

    template = """Please play the role of {character_name} based on the Profile and make your life choice under the Scenario regarding Question. Return the option letter (A, B, C, or D) that your character should most appropriately choose in the current scenario. The Profile consists of Description and Memory, where Description is an overall description of the character, and Memory consists of specific events the character has experienced. First give step-by-step analysis about the question. Then output the answer. Provide your reasoning within the <reasoning></reasoning>tag. For the answer, use <answer>(put your answer here)</answer> and include only the letter corresponding to your choice but not other information.

# Inputs
## Profile
### Description
{character_name}

### Memory
{input_text}

## Scenario
{scenario}

## Extra Information
(to help you better understand the memory and answer the question)
{extra_info}

## Question
{question}

## Options
A. {options0}
B. {options1}
C. {options2}
D. {options3}

"""

    input_values = {
        "character_name": row["character_name"],
        "input_text": row["input_text"],
        "scenario": row["Multiple Choice Question"]["Scenario"],
        "extra_info": row["extra_info"],
        "question": row["Multiple Choice Question"]["Question"],
        "options0": row["Multiple Choice Question"]["Options"][0],
        "options1": row["Multiple Choice Question"]["Options"][1],
        "options2": row["Multiple Choice Question"]["Options"][2],
        "options3": row["Multiple Choice Question"]["Options"][3],
    }
    return template, input_values


def evaluate_response(result: dict[str, Any]) -> dict[str, Any]:
    answer = result["answer"].strip().capitalize()
    correct_answer = result["correct_answer"].strip().capitalize()
    if answer == correct_answer:
        result["is_correct"] = True
    else:
        result["is_correct"] = False
    return result


def create_lifechoices_result(
    parsed_result: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    """Create LifeChoices result dictionary."""
    targeted_entries = [
        "set_id",
        "index",
        "character_name",
        "input_text",
        "reasoning",
        "answer",
        "scenario",
        "extra_info",
        "question",
        "options0",
        "options1",
        "options2",
        "options3",
        "correct_answer",
        "is_correct",
        "socialized_context",
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
        else:
            if entry == "scenario":
                result[entry] = row["Multiple Choice Question"]["Scenario"]
            elif entry == "question":
                result[entry] = row["Multiple Choice Question"]["Question"]
            elif entry == "options0":
                result[entry] = row["Multiple Choice Question"]["Options"][0]
            elif entry == "options1":
                result[entry] = row["Multiple Choice Question"]["Options"][1]
            elif entry == "options2":
                result[entry] = row["Multiple Choice Question"]["Options"][2]
            elif entry == "options3":
                result[entry] = row["Multiple Choice Question"]["Options"][3]
            elif entry == "correct_answer":
                result[entry] = row["Multiple Choice Question"]["Correct Answer"]
            else:
                continue
    result = evaluate_response(result)
    return result


def lifechoices_evaluation_report(results: list[dict[str, Any]]) -> None:
    """Evaluate LifeChoices result."""
    correct_count = 0
    for result in results:
        if result["is_correct"]:
            correct_count += 1
    print(
        f"Current accuracy: {correct_count}/{len(results)} = {correct_count/len(results):.2%}"
    )


async def lifechoices_simulation(
    row: dict[str, Any], engine: Optional[SocialWorldModel] = None
) -> dict[str, Any]:
    pass
