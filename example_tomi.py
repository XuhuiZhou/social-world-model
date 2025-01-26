from rich import print
from rich.logging import RichHandler
from social_world_model.tom_engine import ToMEngine
from pathlib import Path
import pandas as pd
import logging
from datetime import datetime
import json
from typing import Any
import os
from pandas import Series
from sotopia.generation_utils import StrOutputParser, agenerate

FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(
    level=15,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[
        RichHandler()
    ],
)


async def run_single_experiment_vanilla(row, model_name: str = "gpt-4-mini", save_result: bool = False) -> tuple[bool, dict[str, Any]]: # type: ignore
    """A simplified version of run_single_experiment that just uses direct LLM generation.
    
    Args:
        row: A row from the Percept-ToMi dataset
        model_name: Name of the model to use
        
    Returns:
        Tuple of (is_correct, result_dict)
    """
    # Create prompt with story and question
    story = " ".join(eval(row['story']))
    history = f"Story: {story}\nQuestion: {row['question']}"
    
    response = await agenerate(
        model_name=model_name,
        template="Imagine that you are an observer in the scenario. Assume that the characters can perceive every scene in their location but not scenes occurring elsewhere. If something is being moved, that means it is not in its original location anymore.\nYou need to first reason about the question (majorly focusing where the object has been moved to, and focus on the most detailed position possible e.g., the object is in A and A is in B, then you should focus on 'A') and then answer the question with the following format:<reasoning>(reasoning)</reasoning> <answer>(answer)</answer>\n\nBelow is the story and question:\n{message_history}\n\nPossible answers: {candidates}",
        input_values={
            "message_history": history,
            "agent_name": "observer",
            "candidates": ", ".join(eval(row['cands'])),
        },
        temperature=0.7,
        output_parser=StrOutputParser(),
    )
    
    # Extract reasoning and answer from response
    try:
        reasoning = response.split("<reasoning>")[1].split("</reasoning>")[0].strip()
        answer = response.split("<answer>")[1].split("</answer>")[0].strip()
    except IndexError:
        reasoning = "Failed to parse reasoning"
        answer = response  # Use full response as answer if parsing fails
    
    correct_answer = row['answer']
    assert isinstance(answer, str)
    is_correct = correct_answer in answer
    
    result = {
        "question": row['question'],
        "reasoning": reasoning,
        "answer": answer,
        "correct_answer": correct_answer,
        "is_correct": is_correct
    }
    if save_result:
        if not os.path.exists(f"data/tomi_vanilla/{model_name}"):
            os.makedirs(f"data/tomi_vanilla/{model_name}")
        with open(f"data/tomi_vanilla/{model_name}/{row['index']}.json", "w") as f:
            json.dump(result, f)
    return is_correct, result


async def run_single_experiment(row, save_simulation: bool = False, model_name: str = "gpt-4-mini") -> tuple[bool, dict[str, Any]]: # type: ignore
    observation_with_perceivers = [] 
    engine = ToMEngine(
        agent_prompt="You will be asking some questions about your beliefs. Assume that you can perceive every scene in your location but not scenes occurring elsewhere. If something is being moved, that means it is not in its original location anymore.\
You need to first reason about the question (majorly focusing where the object has been moved to, and focus on the most detailed position possible e.g., the object is in A and A is in B, then you should focus on 'A') and then answer the question with the following format:<reasoning>(reasoning)</reasoning> <answer>(answer)</answer>",
        model_name=model_name,
    )
    for story_with_perceivers in eval(row['story_with_perceivers']):
        for observation, perceivers in story_with_perceivers.items():
            assert isinstance(observation, str)
            if "entered" in observation:
                observation += f"({', '.join(perceivers)} were/was there)"
            observation_with_perceivers.append((observation, perceivers))
    
    agent_names = list(set([agent for agents in eval(row['perceivers']) for agent in agents]))
    if row['char2'] and str(row['char2'])!="nan":
        for observation, perceivers in observation_with_perceivers:
            if row['char2'] in perceivers and row['char1'] not in perceivers:
                observation_with_perceivers.remove((observation, perceivers))
    
    await engine.initialize_simulation(agent_names, observation_with_perceivers)
    
    if row['char2'] and str(row['char2'])!="nan":
        object = row["question"].split("searches for")[1].split("?")[0]
        restructure_question = f"Where will you look for the {object}?"
        reasoning, answer = await engine.reason_about_belief(restructure_question, agent_names, target_agent=row['char2'], answer_candidates=eval(row['cands']))
    else:
        restructure_question = row['question'].replace(row['char1'], "you")
        reasoning, answer = await engine.reason_about_belief(restructure_question, agent_names, target_agent=row['char1'], answer_candidates=eval(row['cands']))
        
    correct_answer = row['answer']
    assert isinstance(answer, str)
    is_correct = correct_answer in answer
    
    result = {
        "question": row['question'],
        "reasoning": reasoning,
        "answer": answer,
        "correct_answer": correct_answer,
        "is_correct": is_correct
    }
    
    if save_simulation:
        simulation = engine.get_simulation()
        simulation_dict = simulation.dict()
        if not os.path.exists(f"data/simulations_tomi/{model_name}"):
            os.makedirs(f"data/simulations_tomi/{model_name}")
        with open(f"data/simulations_tomi/{model_name}/{row['index']}.json", "w") as f:
            simulation_dict["correct_answer"] = correct_answer
            simulation_dict["original_question"] = row['question']
            json.dump(simulation_dict, f)
    
    return is_correct, result

async def run_batch(rows: pd.DataFrame, save: bool = False, model_name: str = "gpt-4-mini", mode: str = "vanilla") -> list[tuple[bool, dict[str, Any]]]:
    logging.info(f"Running batch with mode {mode}")
    if mode == "vanilla":
        tasks = [run_single_experiment_vanilla(row, model_name, save) for _, row in rows.iterrows()]
    else:
        tasks = [run_single_experiment(row, save, model_name) for _, row in rows.iterrows()]
    return await asyncio.gather(*tasks)

async def run_tomi_experiments(batch_size: int = 10, save: bool = False, model_name: str = "gpt-4-mini", mode: str = "vanilla") -> None:
    # Read the Percept-ToMi dataset
    tomi_data = pd.read_csv(Path("data/Percept-ToMi.csv"))
    print(f"Total number of experiments: {len(tomi_data)}")
    
    # Split data into batches
    num_batches = (len(tomi_data) + batch_size - 1) // batch_size
    correct_count = 0
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(tomi_data))
        batch_data = tomi_data.iloc[start_idx:end_idx]
        
        print(f"\nProcessing batch {batch_idx + 1}/{num_batches} (experiments {start_idx}-{end_idx})")
        results = await run_batch(batch_data, mode=mode, model_name=model_name, save=save)
        
        # Process results
        for is_correct, result in results:
            if is_correct:
                correct_count += 1
            print(f"\nQuestion: {result['question']}")
            print(f"Reasoning: {result['reasoning']}")
            print(f"Answer: {result['answer']}")
            print(f"Correct answer: {result['correct_answer']}")
            print(f"Current correct count: {correct_count}")

async def run_simple_experiment(story: str, question: str, model_name: str = "gpt-4-mini") -> str:
    """Run a simple ToM experiment with just a story and question.
    
    Args:
        story: The story text
        question: The question to ask about the story
        model_name: Name of the model to use
        
    Returns:
        The model's answer to the question
    """
    engine = ToMEngine(
        agent_prompt="You are answering questions about a story. First reason about the question, then provide your answer in this format: <reasoning>(reasoning)</reasoning> <answer>(answer)</answer>",
        model_name=model_name,
    )
    
    # Initialize with a single agent and the story as an observation
    await engine.initialize_simulation(["observer"], [(story, ["observer"])])
    
    # Get reasoning and answer from the model
    reasoning, answer = await engine.reason_about_belief(
        question, 
        ["observer"],
        target_agent="observer"
    )
    
    return answer

# Add to main for testing
if __name__ == "__main__":
    import asyncio
    print(f"Running experiments")
    
    # Run the original experiments
    asyncio.run(run_tomi_experiments(batch_size=4, save=True, model_name="o1-2024-12-17", mode="vanilla"))