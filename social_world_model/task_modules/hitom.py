import re
import pandas as pd
from pathlib import Path
from social_world_model.social_world_model import SocialWorldModel
from social_world_model.database import SocializedContext
import json
from typing import Any, Optional

HITOM_SOCIALIZED_CONTEXT_PROMPT = """You are dissecting the HITOM scenarios. You should assume the following: (1) An agent witnesses everything and every movements before exiting a location. (2) An agent A can infer another agent B's mental state only if A and B have been in the same location, or have private or public interactions. (3) Note that every agent tend to lie. What a character tells others doesn't affect his actual belief. An agent tend to trust a agent that exited the room later than himself. The exit order is known to all agents. (4) Agents in private communications know that others won't hear them, but they know that anyone can hear any public claims. In the agent's observation, remember to include the objects' locations if the agents are in the same location as the object."""

def reformat_hitom_data(data_list: dict[str, Any]):
    # sample_id -> index
    # answer -> correct_answer
    data_list = data_list['data']
    data = pd.DataFrame(data_list)
    data = data[data["prompting_type"] == "CoTP"]
    data["set_id"] = data.groupby("story", sort=False).ngroup()
    return data.rename(columns={"sample_id": "index", "answer": "correct_answer"})
    

def prepare_hitom_vanilla(row: dict[str, Any], pure_context: bool = False) -> tuple[str, dict[str, Any]]:

    story = row["story"]
    extra_info = row.get("extra_info", "")
    if extra_info:
        if pure_context:
            story = extra_info
            extra_info = ""
        else:
            story = story + "\n" + extra_info

    question = row["question"] + "\n" + row["choices"]
    template = """You are analysing a social interaction and need to answer a question about it. The following story happens in chronological order. You will be given a multiple-choice question and a note at the end. You should assume the following: (1) An agent witnesses everything and every movements before exiting a location. (2) An agent A can infer another agent B's mental state only if A and B have been in the same location, or have private or public interactions. (3) Note that every agent tend to lie. What a character tells others doesn't affect his actual belief. An agent tend to trust a agent that exited the room later than himself. The exit order is known to all agents. (4) Agents in private communications know that others won't hear them, but they know that anyone can hear any public claims. First give step-by-step analysis about the question. Then output the answer. Provide your reasoning within the <reasoning></reasoning>tag. For the answer, use <answer>(put your answer here)</answer> and include only the letter corresponding to your choice but not other information.

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
    
    answer_dict = dict(re.findall(r'([A-Z])\. ([^,]+)', choices))
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
    targeted_entries = ["set_id", "index", "deception", "story_length", "question_order", "story", "question", "reasoning", "answer", "correct_answer", "is_correct", "socialized_context", "extra_info", "choices"]
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

async def hitom_simulation(row: dict[str, Any], engine: Optional[SocialWorldModel] = None) -> dict[str, Any]:
    # TODO
    """Run experiment in simulation mode (using ToM engine for memory tracking)."""
    assert engine is not None, "Engine must be provided"
    # Get the socialized context from the engine
    socialized_context = engine.existing_socialized_contexts[str(row['set_id'])]
    agent_names = socialized_context.agents_names
    # Extract character information from the question
    question = row['question']
    engine.set_agent_prompt(
        "You are trying to figure out a theory of mind question based on a conversation you participated in. "
        "At the end of the conversation, you will be asked a question about the conversation. "
        "You may or may not join the whole conversation as indicated in your memory shown below. "
        f"Here's the full conversation for your reference: {row['context']} "
        "You can compare your memory below with the full conversation above to help you better answer the question. "
        "First reason about the question and then respond with the following format: "
        "<reasoning>(your step-by-step reasoning)</reasoning> <answer>(your final answer)</answer>"
    )
    all_reasoning = []
    all_answers = []
    # Initialize the simulation with the socialized context
    await engine.initialize_simulation_from_socialized_context(socialized_context)
    
    for agent_name in agent_names:
        reasoning, answer = await engine.reason_about_belief(
            question, 
            agent_names, 
            target_agent=agent_name,
        )
        all_reasoning.append(f"{agent_name}'s reasoning: {reasoning}")
        all_answers.append(f"{agent_name}'s answer: {answer}")

    combined_reasoning = "\n\n".join(all_reasoning)
    combined_answer = "\n\n".join(all_answers)
    
    # Create a summary of the reasoning process with agent attribution
    reasoning = f"Based on individual questioning of each agent, the following agents indicated they know the answer to '{question}': None\n\n{combined_reasoning}"
    answer = combined_answer
    engine.simulation.reasoning = reasoning
    engine.simulation.answer = answer
    # Get the simulation state
    simulation = engine.get_simulation()
    simulation_dict = simulation.dict()
    # Prepare the result
    result = {
       "extra_info": socialized_context.to_natural_language() + "\n\n" + f"### Reasoning and answers from each agent participating in the conversation:\n{combined_reasoning}\n\n{combined_answer}. They are simulated agents with partial memory of the whole conversation (induced from the socialized context above), so the answers are subjective and not always correct. Please use them as extra information to help you answer the question.",
       "memory": simulation_dict["agent_memories"],
       "agents": simulation_dict["agents"],
       "socialized_context": socialized_context
    }
    row["extra_info"] = result["extra_info"]
    row["memory"] = result["memory"]
    row["agents"] = result["agents"]
    row["socialized_context"] = result["socialized_context"]
    return result