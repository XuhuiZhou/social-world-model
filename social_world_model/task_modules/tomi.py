from social_world_model.social_world_model import SocialWorldModel
from typing import Any, Optional

TOMI_SOCIALIZED_CONTEXT_PROMPT = """You are dissecting the TOMI scenarios. The assumptions are that the characters can perceive every scene in their location but not scenes occurring elsewhere. If the agent leaves the location, they cannot perceive the scene in that location anymore. In the agent's observation, remember to include the objects' locations if the agents are in the same location as the object."""


def prepare_tomi_vanilla(
    row: dict[str, Any], pure_context: bool = False
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
    template = """Imagine that you are an observer in the scenario. Assume that the characters can perceive every scene in their location but not scenes occurring elsewhere. If something is being moved, that means it is not in its original location anymore. You should majorly focus on where the object has been moved to, and answer the question with the most detailed position possible e.g., the object is in A and A is in B, then you should answer 'A'. Provide your reasoning within the <reasoning></reasoning>tag. For the answer, use <answer>(put your answer here)</answer> and only include the most detailed location but not other information.

Below is the story and question:
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
    """Run experiment in simulation mode (using ToM engine for memory tracking)."""
    assert engine is not None, "Engine must be provided"
    socialized_context = engine.existing_socialized_contexts[str(row["index"])]
    # Extract char1 and char2 if they don't exist
    if not row["char1"] or str(row["char1"]) == "nan":
        question = row["question"].lower()
        if "where will" in question:
            # Type: "Where will Ella look for the celery?"
            row["char1"] = question.split("where will")[1].split("look")[0].strip()
            row["char2"] = ""
            row["char1"] = row["char1"].capitalize()
        elif "where does" in question and "think that" in question:
            # Type: "Where does Owen think that Logan searches for the grapes?"
            row["char1"] = question.split("where does")[1].split("think")[0].strip()
            row["char2"] = question.split("that")[1].split("searches")[0].strip()
            row["char1"] = row["char1"].capitalize()
            row["char2"] = row["char2"].capitalize()
        else:
            # Type: "Where was the xx at the beginning?"
            row["char1"] = ""
            row["char2"] = ""
    engine.set_agent_prompt(
        "You will be asking some questions about your beliefs. The previous history of the interaction below is your memory (i.e., you perceive the entire history of the interaction). Assume that you can perceive every scene in your location but not scenes occurring elsewhere. If something is being moved, that means it is not in its original location anymore.\
You need to first reason about the question (majorly focusing where the object has been moved to, and answer the most detailed position possible e.g., the object is in A and A is in B, then you should answer 'A') and then respond to the question with the following format:<reasoning>(reasoning)</reasoning> <answer>(answer; the answer should be just the position and nothing else)</answer>"
    )
    agent_names = socialized_context.agents_names
    socialized_events = socialized_context.socialized_context
    if row["char2"] and str(row["char2"]) != "nan":
        imagined_socialized_events = []
        for index, event in enumerate(socialized_events):
            if event.observations[row["char1"]] == "none":
                if index > 0:
                    socialized_events[index - 1].actions[row["char2"]] = "none"
            else:
                imagined_socialized_events.append(event)
        socialized_context.socialized_context = imagined_socialized_events
    await engine.initialize_simulation_from_socialized_context(socialized_context)

    if row["char2"] and str(row["char2"]) != "nan":
        object = row["question"].split("searches for")[1].split("?")[0]
        restructure_question = f"Where will you look for the {object}?"
        reasoning, answer = await engine.reason_about_belief(
            restructure_question, agent_names, target_agent=row["char2"]
        )
    elif row["char1"] and str(row["char1"]) != "nan":
        restructure_question = row["question"].replace(row["char1"], "you")
        reasoning, answer = await engine.reason_about_belief(
            restructure_question, agent_names, target_agent=row["char1"]
        )
    else:
        return {}
    correct_answer = row["correct_answer"]
    assert isinstance(answer, str)
    is_correct = correct_answer in answer

    simulation = engine.get_simulation()
    simulation_dict = simulation.dict()
    result = {
        "question": row["question"],
        "reasoning": reasoning,
        "answer": answer,
        "correct_answer": correct_answer,
        "is_correct": is_correct,
        "socialized_context": socialized_context,
        "transformed_question": simulation_dict["question"],
        "memory": simulation_dict["agent_memories"],
        "agents": simulation_dict["agents"],
    }
    row["socialized_context"] = socialized_context
    row["extra_info"] = socialized_context.to_natural_language()
    return result
