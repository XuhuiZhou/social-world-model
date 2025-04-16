import re
import pandas as pd
from social_world_model.social_world_model import SocialWorldModel
from typing import Any, Optional
from copy import deepcopy

HITOM_SOCIALIZED_CONTEXT_PROMPT = """You are dissecting the HITOM scenarios. You should assume the following: (1) An agent witnesses everything and every movements before exiting a location. (2) An agent A can infer another agent B's mental state only if A and B have been in the same location, or have private or public interactions. (3) Note that every agent tend to lie. What a character tells others doesn't affect his actual belief. (4) Agents in private communications know that others won't hear them, but they know that anyone can hear any public claims. In the agent's observation, remember to include the objects' locations if the agents are in the same location as the object."""


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
            story = story

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
        "extra_info": extra_info,
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
    # Get the socialized context from the engine
    socialized_context = deepcopy(engine.existing_socialized_contexts[row["set_id"]])
    agent_names = socialized_context.agents_names
    socialized_events = socialized_context.socialized_context
    # Extract character information from the question
    question = row["question"]
    agent_names_in_question = get_question_agent_names(question)

    # Step 0: Make sure the timestep can be converted to integers
    for event in socialized_events:
        event.timestep = process_timestep(event.timestep)

    # Step 1: Infer whether the question is about an agent's belief
    if len(agent_names_in_question) == 0:
        prompt = """You are analysing a social interaction and need to answer a question about it. The following story happens in chronological order. You will be given a multiple-choice question and a note at the end. First give step-by-step analysis about the question. Then output the answer. Provide your reasoning within the <reasoning></reasoning>tag. For the answer, use <answer>(put your answer here)</answer> and include only the letter corresponding to your choice but not other information."""
    else:
        prompt = """You are analysing a social interaction and need to answer a question about it. The following story happens in chronological order. You will be given a multiple-choice question and a note at the end. You should assume the following: (1) You witness everything and every movement before exiting a location. (2) You can infer another agent's mental state only if you and that agent have been in the same location, or have had private or public interactions. (3) Note that every agent tend to lie. What a character tells others doesn't affect his actual belief. (4) When you engage in private communication, you know others won't hear it, but you are aware that anyone can hear any public claims. First give step-by-step analysis about the question. Then output the answer. Provide your reasoning within the <reasoning></reasoning>tag. For the answer, use <answer>(put your answer here)</answer> and include only the letter corresponding to your choice but not other information. Note that observations in social contexts may be inaccurate, particularly when an agent has already left the location of an object, yet the observation remains as <same_as_state />."""
    engine.set_agent_prompt(prompt)
    for idx, agent_name in enumerate(agent_names_in_question):
        if idx < len(agent_names_in_question) - 1:
            agent1, agent2 = agent_name, agent_names_in_question[idx + 1]

            imagined_socialized_events = []
            for index, event in enumerate(socialized_events):
                if event.observations[agent1] == "none":
                    if index > 0:
                        socialized_events[index - 1].actions[agent2] = "none"
                else:
                    imagined_socialized_events.append(event)
            socialized_events = imagined_socialized_events

    socialized_context.socialized_context = socialized_events
    await engine.initialize_simulation_from_socialized_context(socialized_context)

    # what does A think <obj> is?
    # what does A think B thinks <obj> is?
    # reformat question to be second-person narrative
    # if len(agent_names_in_question) > 0:
    #     question = question.replace(agent_names_in_question[0], "you")
    #     question = question.replace("Where does", "where do")

    question = question + "\n" + row["choices"]

    if len(agent_names_in_question) > 0:
        reasoning, answer = await engine.reason_about_belief(
            question,
            agent_names,
            target_agent=agent_names_in_question[0],
        )
    else:
        return {}

    simulation = engine.get_simulation()
    simulation_dict = simulation.dict()
    result = {
        "question": row["question"],
        "reasoning": reasoning,
        "answer": answer,
        "correct_answer": row["correct_answer"],
        "socialized_context": socialized_context,
        "transformed_question": simulation_dict["question"],
        "memory": simulation_dict["agent_memories"],
        "agents": simulation_dict["agents"],
    }
    row["socialized_context"] = socialized_context
    row["extra_info"] = socialized_context.to_natural_language()
    return result
