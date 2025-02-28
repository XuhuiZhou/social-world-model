# Run the benchmarks for the TOM dataset
import os
import pandas as pd
from rich import print
import json
from pathlib import Path
import logging
from datetime import datetime
import asyncio
from sotopia.generation_utils import StrOutputParser, agenerate
from typing import Any, Literal
from rich.logging import RichHandler
from hidden_utils.fantom_eval_utils import FantomEvalAgent
from social_world_model.tom_engine import ToMEngine
from social_world_model.database import SocializedContext
import typer
import time


# Configure logging
FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(
    level=15,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()],
)

app = typer.Typer(pretty_exceptions_enable=False)

# Add these constants at the top level
BENCHMARK_TYPES = ["tomi", "fantom"]
MODES = ["vanilla", "simulation", "pure_simulation"]
CONTINUE_MODES = ["new", "continue"]

class ToMBenchmarkRunner:
    def __init__(self, model_name: str = "gpt-4-mini", dataset_name: str = "tomi"):
        self.model_name = model_name
        self.fantom_eval_agent = FantomEvalAgent(model_name)
        self.dataset_name = dataset_name
    
    async def run_single_experiment(
        self, 
        row: pd.Series, 
        benchmark_type: Literal["tomi", "fantom"],
        save_result: bool = False,
        mode: Literal["vanilla", "simulation", "pure_simulation"] = "vanilla",
        continue_mode: Literal["new", "continue"] = "new",
        example_analysis_file: str = ""
    ) -> dict[str, Any]:
        """Run a single experiment for either ToMi or FANToM benchmark."""
        # Define result path regardless of continue mode
        save_dir = Path(f"data/{benchmark_type}_results/{mode}_{self.model_name}_{self.dataset_name}")
        result_path = save_dir / f"{row['index']}.json"
        
        # Check for cached results if in continue mode
        if continue_mode == "continue" and result_path.exists():
            print(f"Loading cached result for index {row['index']}")
            with open(result_path) as f:
                return json.load(f)

        # If no cached result or in new mode, run the experiment
        if mode == "vanilla":
            result = await self._run_vanilla(row, benchmark_type)
        elif mode == "pure_simulation":
            result = await self._run_simulation(row, benchmark_type, example_analysis_file, pure_simulation=True)
        else:
            result = await self._run_simulation(row, benchmark_type, example_analysis_file)
        if save_result:
            result['socialized_context'] = result['socialized_context'].model_dump()
            self._save_result(result, result_path)

        return result

    async def _run_vanilla(self, row: pd.Series, benchmark_type: str, pure_simulation: bool = False) -> dict[str, Any]:
        """Run experiment in vanilla mode (direct LLM generation)."""
        # Prepare context and question based on benchmark type
        if 'socialized_context' in row:
            if isinstance(row['socialized_context'], SocializedContext):
                analysis = row['socialized_context'].to_natural_language()
            else:
                analysis_object = SocializedContext(**row['socialized_context'])
                analysis = analysis_object.to_natural_language()
            socialized_context = "\nHere's the analysis of the scenario:\n" + analysis + "And here's the json schema explanation to help you understand the analysis:\n" + json.dumps(SocializedContext.model_json_schema())
        else:
            socialized_context = ""
        if benchmark_type == "tomi":
            try:
                story = " ".join(eval(row['story']))
            except:
                story = row['story']
            if socialized_context:
                if pure_simulation:
                    story = socialized_context
                else:
                    story = story + "\n" + socialized_context
            
            question = row['question']
            template = """Imagine that you are an observer in the scenario. Assume that the characters can perceive every scene in their location but not scenes occurring elsewhere. If something is being moved, that means it is not in its original location anymore. You should majorly focus on where the object has been moved to, and answer the question with the most detailed position possible e.g., the object is in A and A is in B, then you should answer 'A'. For the answer, use <answer>Your answer here</answer> and only include the most detailed location but not other information.

Below is the story and question:
Story: {story}      
Question: {question}"""
            
            if row['cands']:
                template += "\n\nPossible answers: {candidates}"
                
            input_values = {
                "story": story,
                "question": question,
                "candidates": ", ".join(eval(row['cands'])) if row['cands'] else "",
            }
        elif benchmark_type == "fantom":  # fantom
            if socialized_context:
                if pure_simulation:
                    context = socialized_context
                else:
                    context = row['context'] + "\n" + socialized_context
            template = """
You are analyzing a social conversation and need to answer a question about it. You should assume the characters do not know any other information than what is provided in the conversation.

Context: {context}
Question: {question}
"""
            input_values = {
                "context": context,
                "question": row['complete_question'],
            }

        # Generate response
        response = await agenerate(
            model_name=self.model_name,
            template=template,
            input_values=input_values,
            temperature=0.0,
            output_parser=StrOutputParser(),
            structured_output=False
        )

        # Parse response and create result
        if benchmark_type == "tomi":
            result = self._parse_tomi_response(response, row)
        elif benchmark_type == "fantom":
            result = self._create_fantom_result(response, row)
            
        return result

    async def _run_simulation(self, row: pd.Series, benchmark_type: str, example_analysis_file: str = "", pure_simulation: bool = False) -> dict[str, Any]:
        """Run experiment in simulation mode (using ToM engine for memory tracking)."""
        # Check if there's an existing simulation result
        existing_simulation_path = Path(f"data/{benchmark_type}_results/simulation_o1-2024-12-17_{self.dataset_name}/{row['index']}.json")
        
        if existing_simulation_path.exists():
            # Load existing simulation
            with open(existing_simulation_path) as f:
                existing_result = json.load(f)
                if 'socialized_context' in existing_result:
                    # Convert the loaded dict back to SocializedContext object
                    row['socialized_context'] = SocializedContext(**existing_result['socialized_context'])
                    return await self._run_vanilla(row, benchmark_type)
        
        # If no existing simulation found, run new simulation
        engine = ToMEngine(
            agent_prompt="You will be asking some questions about your beliefs. Assume that you can perceive every scene in your location but not scenes occurring elsewhere. If something is being moved, that means it is not in its original location anymore.",
            model_name=self.model_name,
        )
        
        if benchmark_type == "tomi":
            context = " ".join(eval(row['story']))
            engine.set_task_specific_instructions("You are dissecting the TOMI scenarios. The assumptions are that the characters can perceive every scene in their location but not scenes occurring elsewhere. If the agent leaves the location, they cannot perceive the scene in that location anymore. In the agent's observation, remember to include the objects' locations if the agents are in the same location as the object.")
            question = row['question']
        elif benchmark_type == "fantom":
            # Process FANToM-specific observation structure
            context = row['context']
            engine.set_task_specific_instructions("You are analyzing a social conversation and need to answer a question about it. When the agents leave the conversation, they cannot perceive the conversation anymore untill they join the conversation again.")
            question = row['complete_question']

        if example_analysis_file:
            example_analysis = json.load(open(example_analysis_file))
        else:
            example_analysis = ""
        socialized_context = await engine.socialize_context(context, example_analysis)
        row['socialized_context'] = socialized_context
        breakpoint()
        result = await self._run_vanilla(row, benchmark_type, pure_simulation=pure_simulation)
        return result

    def _parse_tomi_response(self, response: str, row: pd.Series) -> dict[str, Any]:
        """Parse ToMi response and create result dictionary."""
        try:
            reasoning = response.split("<think>")[1].split("</think>")[0].strip()
            answer = response.split("<answer>")[1].split("</answer>")[0].strip()
        except IndexError:
            reasoning = "Failed to parse reasoning"
            answer = response
        
        return {
            "story": row['story'],
            "question": row['question'],
            "reasoning": reasoning,
            "answer": answer,
            "correct_answer": row['answer'],
            "is_correct": row['answer'].lower() in answer.lower(),
            "socialized_context": row['socialized_context'] if 'socialized_context' in row else ""
        }

    def _create_fantom_result(self, response: str, row: pd.Series) -> dict[str, Any]:
        """Create FANToM result dictionary."""
        try:
            reasoning = response.split("<think>")[1].split("</think>")[0].strip()
            answer = response.split("</think>")[1].strip()
        except IndexError:
            reasoning = "No reasoning provided"
            answer = response
        
        return {
            "wrong_answer": row['wrong_answer'],
            "missed_info_accessibility": row['missed_info_accessibility'],
            "set_id": row['set_id'],
            "part_id": row['part_id'],
            "question_type": row['question_type'],
            "tom_type": row['tom_type'],
            "context": row['context'],
            "gt_perception": row['gt_perception'],
            "question": row['complete_question'],
            "reasoning": reasoning,
            "answer": answer,
            "correct_answer": row['correct_answer'],
            "socialized_context": row['socialized_context'] if 'socialized_context' in row else ""
        }

    def _save_result(self, result: dict, result_path: Path):
        """Save experiment result to file."""
        save_dir = result_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

def validate_benchmark_type(value: str) -> str:
    """Validate benchmark type."""
    if value not in BENCHMARK_TYPES:
        raise typer.BadParameter(f"Benchmark type must be one of {BENCHMARK_TYPES}")
    return value

def validate_mode(value: str) -> str:
    """Validate mode."""
    if value not in MODES:
        raise typer.BadParameter(f"Mode must be one of {MODES}")
    return value

def validate_continue_mode(value: str) -> str:
    """Validate continue mode."""
    if value not in CONTINUE_MODES:
        raise typer.BadParameter(f"Continue mode must be one of {CONTINUE_MODES}")
    return value

@app.command()
def run_benchmark(
    benchmark_type: str = typer.Argument(
        ..., 
        help="Type of benchmark to run (tomi/fantom)",
        callback=validate_benchmark_type
    ),
    dataset_path: str = None,
    batch_size: int = 4,
    save: bool = True,
    model_name: str = "o1-2024-12-17",
    mode: str = typer.Option(
        "vanilla",
        help="Mode to run in (vanilla/simulation)",
        callback=validate_mode
    ),
    continue_mode: str = typer.Option(
        "new",
        help="Whether to continue from existing results (new/continue)",
        callback=validate_continue_mode
    ),
    example_analysis_file: str = typer.Option(
        "",
        help="Path to the example analysis file"
    )
) -> None:
    """Run benchmark experiments."""
    if dataset_path is None:
        dataset_path = {
            "tomi": "./data/rephrased_tomi_test_600.csv",
            "fantom": "data/Percept_FANToM/Percept-FANToM-flat.csv"
        }[benchmark_type]
    
    asyncio.run(_run_benchmark(
        benchmark_type=benchmark_type,
        dataset_path=dataset_path,
        batch_size=batch_size,
        save=save,
        model_name=model_name,
        mode=mode,
        continue_mode=continue_mode,
        example_analysis_file=example_analysis_file
    ))

async def _run_benchmark(
    benchmark_type: str,
    dataset_path: str,
    batch_size: int,
    save: bool,
    model_name: str,
    mode: str,
    continue_mode: str = "new",
    example_analysis_file: str = ""
):
    """Async implementation of benchmark runner."""
    runner = ToMBenchmarkRunner(model_name, dataset_name=dataset_path.split("/")[-1])
    data = pd.read_csv(dataset_path).fillna("")
    
    if benchmark_type == "fantom":
        # Sample a subset of set_ids for FANToM
        set_id_counts = data['set_id'].value_counts()
        sampled_set_ids = set_id_counts.sample(n=10, random_state=42).index
        data = data[data['set_id'].isin(sampled_set_ids)]
    
    print(f"Running {benchmark_type.upper()} benchmark with {len(data)} examples")
    
    all_results = []
    correct_count = 0
    
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(data) + batch_size - 1)//batch_size}")
        
        tasks = [
            runner.run_single_experiment(
                row, 
                benchmark_type=benchmark_type,
                save_result=save,
                mode=mode,
                continue_mode=continue_mode,
                example_analysis_file=example_analysis_file
            ) 
            for _, row in batch.iterrows()
        ]
        results = await asyncio.gather(*tasks)
        all_results.extend(results)
        
        # Print results
        for result in results:
            if benchmark_type == "tomi":
                if result['is_correct']:
                    correct_count += 1
                print(f"\nQuestion: {result['question']}")
                print(f"Reasoning: {result['reasoning']}")
                print(f"Answer: {result['answer']}")
                print(f"Correct answer: {result['correct_answer']}")
                print(f"Current accuracy: {correct_count}/{i + len(results)} = {correct_count/(i + len(results)):.2%}")
    
    # Final evaluation
    if benchmark_type == "fantom":
        evaluated_results = runner.fantom_eval_agent.evaluate_response(
            all_results, 
            [result['answer'] for result in all_results]
        )
        results_df = pd.DataFrame(evaluated_results)
        report = runner.fantom_eval_agent.score_and_analyze(results_df)
        print("\nEvaluation Report:")
        print(report)
    else:
        print(f"\nFinal accuracy: {correct_count}/{len(data)} = {correct_count/len(data):.2%}")

@app.command()
def evaluate_results(
    benchmark_type: str = typer.Argument(
        ..., 
        help="Type of benchmark to evaluate (tomi/fantom)",
        callback=validate_benchmark_type
    ),
    model_name: str = typer.Argument(..., help="Name of the model to evaluate"),
    dataset_name: str = typer.Argument(..., help="Name of the dataset to evaluate"),
    mode: str = typer.Option(
        "vanilla",
        help="Mode to evaluate (vanilla/simulation)",
        callback=validate_mode
    )
) -> None:
    """Evaluate saved results for a given model and benchmark."""
    # Determine results directory
    results_dir = Path(f"data/{benchmark_type}_results/{mode}_{model_name}_{dataset_name}")
    if not results_dir.exists():
        print(f"No results found at {results_dir}")
        return
    
    # Load all results
    results = []
    for file_path in sorted(results_dir.glob("*.json")):
        with open(file_path) as f:
            result = json.load(f)
            result['file_name'] = file_path.name
            results.append(result)
    
    print(f"\nLoaded {len(results)} results from {results_dir}")
    
    if benchmark_type == "tomi":
        # Calculate ToMi metrics
        correct_count = sum(1 for r in results if r['is_correct'])
        accuracy = correct_count / len(results)
        
        # Group results by story type if available
        story_metrics = {}
        for result in results:
            story = result.get('story')
            if story:
                story_type = "unknown"
                if "entered" in str(story):
                    story_type = "location_change"
                elif "moved" in str(story):
                    story_type = "object_movement"
                elif "saw" in str(story):
                    story_type = "observation"
                
                if story_type not in story_metrics:
                    story_metrics[story_type] = {"correct": 0, "total": 0}
                story_metrics[story_type]["total"] += 1
                if result['is_correct']:
                    story_metrics[story_type]["correct"] += 1
        
        # Print results
        print("\nOverall Results:")
        print(f"Total examples: {len(results)}")
        print(f"Correct answers: {correct_count}")
        print(f"Accuracy: {accuracy:.2%}")
        
        if story_metrics:
            print("\nResults by story type:")
            for story_type, metrics in story_metrics.items():
                type_accuracy = metrics["correct"] / metrics["total"]
                print(f"{story_type}: {metrics['correct']}/{metrics['total']} = {type_accuracy:.2%}")
                
        # Sample of incorrect predictions
        print("\nIncorrect predictions:")
        incorrect_results = [r for r in results if not r['is_correct']]
        for result in incorrect_results:  # Show first 5 incorrect predictions
            print(f"\nindex: {result['file_name'].split('.')[0]}")
            
    else:  # FANToM evaluation
        runner = ToMBenchmarkRunner(model_name)
        evaluated_results = runner.fantom_eval_agent.evaluate_response(
            results, 
            [result['answer'] for result in results]
        )
        results_df = pd.DataFrame(evaluated_results)
        report = runner.fantom_eval_agent.score_and_analyze(results_df)
        print("\nEvaluation Report:")
        print(report)
        
        # Sample of predictions
        print("\nSample of predictions:")
        for result in results[:5]:  # Show first 5 predictions
            print(f"\nContext: {result['context'][:200]}...")  # Show first 200 chars of context
            print(f"Question: {result['question']}")
            print(f"Answer: {result['answer']}")
            print(f"Reasoning: {result['reasoning']}")

if __name__ == "__main__":
    app()