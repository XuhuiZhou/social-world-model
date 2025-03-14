from rich import print
from rich.logging import RichHandler
from social_world_model.tom_engine import ToMEngine
from pathlib import Path
import pandas as pd
import logging
import json
from typing import Any, Optional
import os
from sotopia.generation_utils import StrOutputParser, agenerate
from social_world_model.tom_engine import SocializedContext  # type: ignore
import asyncio

# Set up logging

FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(
    level=15,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()],
)


def dictlize(d: dict[str, Any]) -> dict[str, Any]:
    """Convert a list of observations/actions into a dictionary format.

    Args:
        d: Input data that may contain lists of "key: value" strings

    Returns:
        Transformed dictionary with nested dictionaries instead of lists
    """
    socialized_context = d["socialized_context"]
    for step in socialized_context:
        for key, value in step.items():
            if key in ["observations", "actions"] and isinstance(value, list):
                step[key] = {}
                for item in value:
                    if isinstance(item, str) and ":" in item:
                        k, v = item.split(":", 1)
                        step[key][k.strip()] = v.strip()
    return d


async def run_single_experiment_vanilla(
    row: pd.Series, model_name: str = "gpt-4-mini", save_result: bool = False  # type: ignore
) -> tuple[bool, dict[str, Any]]:
    """A simplified version of run_single_experiment that just uses direct LLM generation.

    Args:
        row: A row from the Percept-ToMi dataset
        model_name: Name of the model to use

    Returns:
        Tuple of (is_correct, result_dict)
    """
    # Create prompt with story and question
    story = " ".join(eval(row["story"]))
    history = f"Story: {story}\nQuestion: {row['question']}"

    template = "Imagine that you are an observer in the scenario. Assume that the characters can perceive every scene in their location but not scenes occurring elsewhere. If something is being moved, that means it is not in its original location anymore.\nYou need to first reason about the question (majorly focusing where the object has been moved to, and focus on the most detailed position possible e.g., the object is in A and A is in B, then you should answer 'A') and then answer the question with the following format:<reasoning>(reasoning)</reasoning> <answer>(answer)</answer>\n\nBelow is the story and question:\n{message_history}"

    if row["cands"]:
        template += "\n\nPossible answers: {candidates}"

    response = await agenerate(
        model_name=model_name,
        template=template,
        input_values={
            "message_history": history,
            "agent_name": "observer",
            "candidates": ", ".join(eval(row["cands"])) if row["cands"] else "",
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

    correct_answer = row["answer"]
    assert isinstance(answer, str)
    is_correct = correct_answer in answer

    result = {
        "question": row["question"],
        "reasoning": reasoning,
        "answer": answer,
        "correct_answer": correct_answer,
        "is_correct": is_correct,
    }
    if save_result:
        if not os.path.exists(f"data/tomi_vanilla/{model_name}_rephrased"):
            os.makedirs(f"data/tomi_vanilla/{model_name}_rephrased")
        with open(
            f"data/tomi_vanilla/{model_name}_rephrased/{row['index']}.json", "w"
        ) as f:
            json.dump(result, f)
    return is_correct, result


async def run_single_experiment(
    row: pd.Series, save_simulation: bool = False, model_name: str = "gpt-4-mini"  # type: ignore
) -> tuple[bool, dict[str, Any]]:
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
    observation_with_perceivers = []
    engine = ToMEngine(
        agent_prompt="You will be asking some questions about your beliefs. The previous history of the interaction below is your memory (i.e., you perceive the entire history of the interaction). Assume that you can perceive every scene in your location but not scenes occurring elsewhere. If something is being moved, that means it is not in its original location anymore.\
You need to first reason about the question (majorly focusing where the object has been moved to, and answer the most detailed position possible e.g., the object is in A and A is in B, then you should answer 'A') and then respond to the question with the following format:<reasoning>(reasoning)</reasoning> <answer>(answer; the answer should be just the position and nothing else)</answer>",
        model_name=model_name,
    )
    if "socialized_context" in row and str(row["socialized_context"]) != "nan":
        socialized_context = dictlize(row["socialized_context"])
        agent_names = socialized_context["agents_names"]
        socialized_events = socialized_context["socialized_context"]
        if row["char2"] and str(row["char2"]) != "nan":
            imagined_socialized_events = []
            for index, event in enumerate(socialized_events):
                if event["observations"][row["char1"]] == "none":
                    if index > 0:
                        socialized_events[index - 1]["actions"][row["char2"]] = "none"
                else:
                    imagined_socialized_events.append(event)
            socialized_context["socialized_context"] = imagined_socialized_events
        await engine.initialize_simulation_from_socialized_context(socialized_context)
    else:
        for story_with_perceivers in eval(row["story_with_perceivers"]):
            for observation, perceivers in story_with_perceivers.items():
                assert isinstance(observation, str)
                if "entered" in observation:
                    observation += f"({', '.join(perceivers)} were/was there)"
                observation_with_perceivers.append((observation, perceivers))
        agent_names = list(
            set([agent for agents in eval(row["perceivers"]) for agent in agents])
        )
        if row["char2"] and str(row["char2"]) != "nan":
            for observation, perceivers in observation_with_perceivers:
                if row["char2"] in perceivers and row["char1"] not in perceivers:
                    observation_with_perceivers.remove((observation, perceivers))
        await engine.initialize_simulation(agent_names, observation_with_perceivers)
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
        is_correct, result = await run_single_experiment_vanilla(
            row, model_name, save_result=True
        )
        reasoning = result["reasoning"]
        answer = result["answer"]

    correct_answer = row["answer"]
    assert isinstance(answer, str)
    is_correct = correct_answer in answer

    result = {
        "question": row["question"],
        "reasoning": reasoning,
        "answer": answer,
        "correct_answer": correct_answer,
        "is_correct": is_correct,
    }
    if save_simulation:
        simulation = engine.get_simulation()
        simulation_dict = simulation.dict()
        if not os.path.exists(f"data/simulations_tomi/{model_name}"):
            os.makedirs(f"data/simulations_tomi/{model_name}")
        with open(f"data/simulations_tomi/{model_name}/{row['index']}.json", "w") as f:
            result["transformed_question"] = simulation_dict["question"]
            result["memory"] = simulation_dict["agent_memories"]
            result["agents"] = simulation_dict["agents"]
            json.dump(result, f)

    return is_correct, result


async def run_batch(
    rows: pd.DataFrame,
    save: bool = False,
    model_name: str = "gpt-4-mini",
    mode: str = "vanilla",
) -> list[tuple[bool, dict[str, Any]]]:
    logging.info(f"Running batch with mode {mode}")
    if mode == "vanilla":
        tasks = [
            run_single_experiment_vanilla(row, model_name, save)
            for _, row in rows.iterrows()
        ]
    else:
        tasks = [
            run_single_experiment(row, save, model_name) for _, row in rows.iterrows()
        ]
    return await asyncio.gather(*tasks)


async def run_tomi_experiments(
    dataset_path: str,
    batch_size: int = 10,
    save: bool = False,
    model_name: str = "gpt-4-mini",
    mode: str = "vanilla",
    specific_indices: Optional[list[int]] = None,
    critic_mode: bool = False,
) -> None:
    # Check if dataset_path is a folder or file
    path = Path(dataset_path)
    if path.is_dir():
        # Handle folder containing JSON files
        json_files = list(path.glob("*.json"))
        data_list = []
        for json_file in json_files:
            if specific_indices and int(json_file.stem) not in specific_indices:
                continue
            with open(json_file, "r") as f:
                data = json.load(f)
                # Convert JSON data to match DataFrame structure
                data_entry = {
                    "index": json_file.stem,
                    "question": data.get("question", ""),
                    "answer": data.get("correct_answer", ""),
                    "story": data.get("story", ""),  # Empty story for JSON cases
                    "socialized_context": data.get("socialized_context", ""),
                    "char1": data.get("char1", ""),
                    "char2": data.get("char2", ""),
                    "cands": data.get("cands", ""),  # Empty candidates for JSON cases
                }
                data_list.append(data_entry)
        tomi_data = pd.DataFrame(data_list)
        if critic_mode:
            example_analysis = json.load(open("data/social_contexts_example/tomi.json"))
            example_patterns = open(
                "data/social_contexts_example/tomi_patterns.txt"
            ).read()
            tom_engine = ToMEngine(model_name=model_name)
            # Process in batches for critique operations
            critique_batch_size = batch_size
            num_critique_batches = (
                len(tomi_data) + critique_batch_size - 1
            ) // critique_batch_size
            improved_contexts = []

            for batch_idx in range(num_critique_batches):
                start_idx = batch_idx * critique_batch_size
                end_idx = min((batch_idx + 1) * critique_batch_size, len(tomi_data))
                batch_data = tomi_data.iloc[start_idx:end_idx]
                # Create tasks for current batch of critique operations
                critique_tasks = [
                    tom_engine.critique_and_improve_context(
                        SocializedContext(**row["socialized_context"]),
                        context=row["story"],
                        example_analysis=example_analysis,
                        example_patterns=example_patterns,
                    )
                    for _, row in batch_data.iterrows()
                ]

                # Execute batch tasks concurrently
                batch_improved_contexts = await asyncio.gather(*critique_tasks)
                improved_contexts.extend(batch_improved_contexts)

                print(
                    f"Processed critique batch {batch_idx + 1}/{num_critique_batches}"
                )

            # Update the dataframe with improved contexts
            for (index, row), improved_context in zip(
                tomi_data.iterrows(), improved_contexts
            ):
                assert isinstance(improved_context, SocializedContext)
                tomi_data.at[index, "socialized_context"] = (
                    improved_context.model_dump()
                )
                with open(
                    f"data/fixed_socialized_contexts/{row['index']}.json", "w"
                ) as f:
                    json.dump(improved_context.model_dump(), f, indent=2)
    else:
        # Read CSV file as before
        tomi_data = pd.read_csv(dataset_path).fillna("")

    print(f"Total number of experiments: {len(tomi_data)}")

    # Split data into batches
    num_batches = (len(tomi_data) + batch_size - 1) // batch_size
    correct_count = 0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(tomi_data))
        batch_data = tomi_data.iloc[start_idx:end_idx]

        print(
            f"\nProcessing batch {batch_idx + 1}/{num_batches} (experiments {start_idx}-{end_idx})"
        )
        results = await run_batch(
            batch_data, mode=mode, model_name=model_name, save=save
        )

        # Process results
        for is_correct, result in results:
            if is_correct:
                correct_count += 1
            print(f"\nQuestion: {result['question']}")
            print(f"Reasoning: {result['reasoning']}")
            print(f"Answer: {result['answer']}")
            print(f"Correct answer: {result['correct_answer']}")
            print(f"Current correct count: {correct_count}")


# Add to main for testing
if __name__ == "__main__":
    import asyncio
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--specific_indices",
        nargs="+",
        type=int,
        help="List of specific experiment indices to run",
    )
    args = parser.parse_args()

    print("Running experiments")

    # Run the original experiments
    # asyncio.run(run_tomi_experiments(dataset_path="./data/rephrased_tomi_test_600.csv", batch_size=4, save=True, model_name="o1-2024-12-17", mode="vanilla"))
    asyncio.run(
        run_tomi_experiments(
            dataset_path="./data/tomi_results/simulation_o1-2024-12-17_rephrased_tomi_test_600.csv",
            batch_size=5,
            save=True,
            model_name="o1-2024-12-17",
            mode="simulation",
            specific_indices=args.specific_indices,
            critic_mode=True,
        )
    )
