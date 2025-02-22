import pandas as pd
from rich import print
import json
from pathlib import Path
import logging
from datetime import datetime
import os
from sotopia.generation_utils import StrOutputParser, agenerate
from typing import Any
from rich.logging import RichHandler
from hidden_utils.fantom_eval_utils import FantomEvalAgent
import typer
import glob
import time

FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(
    level=15,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()],
)

app = typer.Typer()

fantom_eval_agent = FantomEvalAgent("gpt-4-mini")

def load_fantom_data():
    """Load the FANToM dataset from CSV"""
    data_path = Path("data/Percept_FANToM/flattened_fantom.csv")
    df = pd.read_csv(data_path)
    return df

def analyze_perceivers(df):
    """Analyze the perceivers in the gt_perception column"""
    # Extract all unique perceivers
    all_perceivers = set()
    for perceptions in df['gt_perception']:
        try:
            perceptions_dict = json.loads(perceptions.replace("'", '"'))
            all_perceivers.update(perceptions_dict.keys())
        except:
            continue
    return sorted(list(all_perceivers))

def get_perception_stats(df):
    """Get statistics about perception types and frequencies"""
    stats = {
        'total_questions': len(df),
        'question_types': df['question_type'].value_counts().to_dict(),
        'tom_types': df['tom_type'].value_counts().to_dict(),
    }
    
    # Analyze perception patterns
    perception_patterns = []
    for perceptions in df['gt_perception']:
        try:
            perceptions_dict = json.loads(perceptions.replace("'", '"'))
            pattern = tuple(sorted(perceptions_dict.keys()))
            perception_patterns.append(pattern)
        except:
            continue
    
    # Count unique perception patterns
    pattern_counts = pd.Series(perception_patterns).value_counts().to_dict()
    stats['perception_patterns'] = pattern_counts
    
    return stats

def load_saved_results(model_name: str) -> list[dict]:
    """Load saved results from json files"""
    results_dir = Path(f"data/fantom_results/{model_name}")
    if not results_dir.exists():
        raise ValueError(f"No results found for model {model_name}")
    
    results = []
    for file_path in sorted(results_dir.glob("*.json")):
        with open(file_path) as f:
            results.append(json.load(f))
    return results

@app.command()
def run_cli(
    dataset_path: str = "data/Percept_FANToM/Percept-FANToM-flat.csv",
    batch_size: int = 1,
    save: bool = True,
    model_name: str = "o1-2024-12-17"
) -> None:
    """Wrapper for running experiments that handles async execution"""
    asyncio.run(run(dataset_path, batch_size, save, model_name))

@app.command()
def evaluate(model_name: str):
    """Evaluate saved results for a given model"""
    print(f"Loading saved results for {model_name}...")
    results = load_saved_results(model_name)
    
    print(f"Evaluating {len(results)} results...")
    fantom_eval_agent = FantomEvalAgent(model_name)
    evaluated_results = fantom_eval_agent.evaluate_response(
        results, 
        [result['answer'] for result in results]
    )
    
    results_df = pd.DataFrame(evaluated_results)
    report = fantom_eval_agent.score_and_analyze(results_df)
    print("\nEvaluation Report:")
    print(report)

async def run(
    dataset_path: str = "data/Percept_FANToM/Percept-FANToM-flat.csv",
    batch_size: int = 5,
    save: bool = True,
    model_name: str = "o1-2024-12-17"
) -> None:
    """Run experiments on the FANToM dataset."""
    fantom_eval_agent = FantomEvalAgent(model_name)
    
    # Load and prepare data
    fantom_data = pd.read_csv(dataset_path).fillna("")
    print(f"Total number of experiments: {len(fantom_data)}")

    # Get statistics about unique set_ids
    set_id_counts = fantom_data['set_id'].value_counts()
    print(f"Number of unique set_ids: {len(set_id_counts)}")
    print(f"Average samples per set_id: {set_id_counts.mean():.2f}")
    print(f"Distribution of samples per set_id:")
    print(set_id_counts.describe())
    
    # Sample a subset of set_ids and get all their corresponding entries
    sampled_set_ids = set_id_counts.sample(n=10, random_state=42).index
    fantom_data = fantom_data[fantom_data['set_id'].isin(sampled_set_ids)]
    print(f"\nSampled {len(sampled_set_ids)} set_ids")
    print(f"Total number of experiments after sampling: {len(fantom_data)}")

    # Split data into batches
    num_batches = (len(fantom_data) + batch_size - 1) // batch_size
    all_results = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(fantom_data))
        batch_data = fantom_data.iloc[start_idx:end_idx]
        print(f"\nProcessing batch {batch_idx + 1}/{num_batches} (experiments {start_idx}-{end_idx})")
        results = await run_batch(batch_data, save=save, model_name=model_name)
        all_results.extend(results)

    all_results_df = pd.DataFrame(all_results)
    report = fantom_eval_agent.score_and_analyze(all_results_df)
    print("\nEvaluation Report:")
    print(report)

async def run_single_experiment_fantom(row, model_name: str = "gpt-4-mini", save_result: bool = False) -> tuple[bool, dict[str, Any]]:
    """Run a single FANToM experiment using LLM generation.
    
    Args:
        row: A row from the FANToM dataset
        model_name: Name of the model to use
        save_result: Whether to save the result to disk
        
    Returns:
        Tuple of (is_correct, result_dict)
    """

    # check if the result already exists
    result_path = Path(f"data/fantom_results/{model_name}/{row['index']}.json")
    if result_path.exists():
        with open(result_path) as f:
            return json.load(f)

    # Create prompt with context and question
    context = row['context']
    complete_question = row['complete_question']
    
    template = """
You are analyzing a social conversation and need to answer a question about it. You should assume the characters do not know any other information than what is provided in the conversation.

Context: {context}
Question: {question}
"""
    response = await agenerate(
        model_name=model_name,
        template=template,
        input_values={
            "context": context,
            "question": complete_question,
        },
        temperature=0.7,
        output_parser=StrOutputParser(),
        structured_output=False
    )

    # Extract reasoning and answer
    try:
        reasoning = response.split("<think>")[1].split("</think>")[0].strip()
        answer = response.split("</think>")[1].strip()
    except IndexError:
        reasoning = "No reasoning provided"
        answer = response
    
    result = {
        "wrong_answer": row['wrong_answer'],
        "missed_info_accessibility": row['missed_info_accessibility'],
        "set_id": row['set_id'],
        "part_id": row['part_id'],
        "question_type": row['question_type'],
        "tom_type": row['tom_type'],
        "context": row['context'],
        "gt_perception": row['gt_perception'],
        "question": complete_question,
        "reasoning": reasoning,
        "answer": answer,
        "correct_answer": row['correct_answer'],
    }


    if save_result:
        save_dir = Path(f"data/fantom_results/{model_name}")
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / f"{row['index']}.json", "w") as f:
            json.dump(result, f, indent=2)

    # sleep to avoid rate limiting
    time.sleep(4)

    return result

async def run_batch(rows: pd.DataFrame, save: bool = False, model_name: str = "gpt-4-mini") -> list[tuple[bool, dict[str, Any]]]:
    """Run a batch of FANToM experiments."""
    logging.info(f"Running batch with model {model_name}")
    tasks = [run_single_experiment_fantom(row, model_name, save) for _, row in rows.iterrows()]
    results = await asyncio.gather(*tasks)
    fantom_eval_agent = FantomEvalAgent(model_name)
    evaluated_output = fantom_eval_agent.evaluate_response(results, [result['answer'] for result in results])
    return evaluated_output

if __name__ == "__main__":
    import asyncio
    # run the app
    # uv run python example_fantom.py run-cli o1-2024-12-17
    # or with optional parameters:
    # uv run python example_fantom.py run-cli o1-2024-12-17 --batch-size 5 --save

    # evaluate the app
    # uv run python example_fantom.py evaluate o1-2024-12-17
    app()