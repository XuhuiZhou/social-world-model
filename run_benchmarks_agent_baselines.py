# Run agent debate baseline benchmarks for ToMI dataset
import pandas as pd
from rich import print
import json
from pathlib import Path
import logging
import asyncio
from collections import Counter
from social_world_model.generation_utils import StrOutputParser, agenerate
from typing import Any, cast
from rich.logging import RichHandler
from social_world_model.task_modules import (
    prepare_tomi_vanilla,
    create_tomi_result,
    tomi_evaluation_report,
)
import typer

# Configure logging
FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(
    level=15,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()],
)

app = typer.Typer(pretty_exceptions_enable=False)

# Create type aliases
MAX_RETRIES = 10
NUM_DEBATE_AGENTS = 3
NUM_DEBATE_ROUNDS = 2


class AgentBaselineRunner:
    def __init__(
        self,
        model_name: str = "gpt-4-mini",
        dataset_name: str = "tomi",
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name

    def _parse_response(self, response: str, row: dict[str, Any]) -> dict[str, Any]:
        """Parse ToMi response and create result dictionary."""
        try:
            reasoning = (
                response.split("<reasoning>")[1].split("</reasoning>")[0].strip()
            )
            answer = response.split("<answer>")[1].split("</answer>")[0].strip()
        except Exception as e:
            print(f"Failed to parse response: {e}")
            reasoning = "Failed to parse reasoning"
            answer = response

        return {
            "reasoning": reasoning,
            "answer": answer,
        }

    def _save_result(self, result: dict[str, Any], result_path: Path) -> None:
        """Save experiment result to file."""
        save_dir = result_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

    async def _run_debate(
        self,
        row: dict[str, Any],
        with_reasoning: bool = True,
    ) -> dict[str, Any]:
        """Run experiment in debate mode (multi-agent debate with consensus)."""
        # Prepare template and input values for tomi (same as vanilla for initial answers)
        template, input_values = prepare_tomi_vanilla(
            row, pure_context=False, with_reasoning=with_reasoning
        )
        
        debate_history: list[dict[str, Any]] = []
        
        # Step 1: Generate initial answers from 3 agents independently
        print(f"Generating initial answers from {NUM_DEBATE_AGENTS} agents...")
        initial_answers = []
        for agent_idx in range(NUM_DEBATE_AGENTS):
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    response = await agenerate(
                        model_name=self.model_name,
                        template=template,
                        input_values=input_values,
                        temperature=0.7,  # Use some temperature for diversity
                        output_parser=StrOutputParser(),
                        structured_output=False,
                    )
                    parsed = self._parse_response(response, row)
                    initial_answers.append({
                        "agent": agent_idx,
                        "round": 0,
                        "reasoning": parsed["reasoning"],
                        "answer": parsed["answer"],
                        "full_response": response,
                    })
                    break
                except Exception as e:
                    print(
                        f"Error generating initial answer for agent {agent_idx} on attempt {attempt}: {e}"
                    )
                    if attempt == MAX_RETRIES:
                        initial_answers.append({
                            "agent": agent_idx,
                            "round": 0,
                            "reasoning": "Failed to generate",
                            "answer": "",
                            "full_response": "",
                        })
        
        debate_history.append({"round": 0, "answers": initial_answers})
        
        # Step 2: Debate rounds - agents refine their answers based on others' responses
        current_answers = initial_answers
        for round_num in range(1, NUM_DEBATE_ROUNDS + 1):
            print(f"Debate round {round_num}/{NUM_DEBATE_ROUNDS}...")
            refined_answers = []
            
            for agent_idx in range(NUM_DEBATE_AGENTS):
                # Get other agents' answers from previous round
                other_answers = [
                    ans for ans in current_answers if ans["agent"] != agent_idx
                ]
                
                # Create critique/refinement prompt
                critique_template = """{original_prompt}

You have already provided an initial answer. Below are the answers and reasoning from other agents analyzing the same question:

{other_agents_responses}

Please review your own reasoning and answer in light of the other agents' perspectives. You may:
1. Refine your answer if you find flaws in your reasoning
2. Strengthen your reasoning if other agents raise valid points
3. Maintain your answer if you remain confident it is correct

Provide your refined reasoning within the <reasoning></reasoning> tag. For the answer, use <answer>(put your answer here)</answer> and only include the most **detailed** location but not other information."""

                other_responses_text = "\n\n".join([
                    f"Agent {other_ans['agent'] + 1}'s reasoning:\n{other_ans['reasoning']}\n\nAgent {other_ans['agent'] + 1}'s answer: {other_ans['answer']}"
                    for other_ans in other_answers
                ])
                
                critique_input_values = {
                    **input_values,
                    "original_prompt": template.format(**input_values),
                    "other_agents_responses": other_responses_text,
                }
                
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        response = await agenerate(
                            model_name=self.model_name,
                            template=critique_template,
                            input_values=critique_input_values,
                            temperature=0.5,  # Lower temperature for refinement
                            output_parser=StrOutputParser(),
                            structured_output=False,
                        )
                        parsed = self._parse_response(response, row)
                        refined_answers.append({
                            "agent": agent_idx,
                            "round": round_num,
                            "reasoning": parsed["reasoning"],
                            "answer": parsed["answer"],
                            "full_response": response,
                        })
                        break
                    except Exception as e:
                        print(
                            f"Error refining answer for agent {agent_idx} in round {round_num} on attempt {attempt}: {e}"
                        )
                        if attempt == MAX_RETRIES:
                            # Fall back to previous answer if refinement fails
                            prev_answer = next(
                                (ans for ans in current_answers if ans["agent"] == agent_idx),
                                current_answers[agent_idx] if agent_idx < len(current_answers) else initial_answers[agent_idx]
                            )
                            refined_answers.append({
                                "agent": agent_idx,
                                "round": round_num,
                                "reasoning": prev_answer["reasoning"],
                                "answer": prev_answer["answer"],
                                "full_response": prev_answer["full_response"],
                            })
            
            current_answers = refined_answers
            debate_history.append({"round": round_num, "answers": refined_answers})
        
        # Step 3: Generate consensus answer from final debate answers
        print("Generating consensus answer...")
        consensus_template = """{original_prompt}

After multiple rounds of discussion and debate, {num_agents} agents have analyzed this question. Below are their final answers and reasoning:

{final_debate_responses}

Based on all the agents' reasoning and answers, please provide a final consensus answer that synthesizes the best insights from the debate.

Provide your consensus reasoning within the <reasoning></reasoning> tag. For the answer, use <answer>(put your answer here)</answer> and only include the most **detailed** location but not other information."""

        final_responses_text = "\n\n".join([
            f"Agent {ans['agent'] + 1}'s final reasoning:\n{ans['reasoning']}\n\nAgent {ans['agent'] + 1}'s final answer: {ans['answer']}"
            for ans in current_answers
        ])
        
        consensus_input_values = {
            **input_values,
            "original_prompt": template.format(**input_values),
            "num_agents": NUM_DEBATE_AGENTS,
            "final_debate_responses": final_responses_text,
        }
        
        consensus_response = ""
        consensus_generated = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                consensus_response = await agenerate(
                    model_name=self.model_name,
                    template=consensus_template,
                    input_values=consensus_input_values,
                    temperature=0.0,  # Deterministic for consensus
                    output_parser=StrOutputParser(),
                    structured_output=False,
                )
                consensus_generated = True
                break
            except Exception as e:
                print(
                    f"Error generating consensus on attempt {attempt}/{MAX_RETRIES}: {e}"
                )
                if attempt == MAX_RETRIES:
                    # Fallback: use the most common answer from final debate round
                    print("Consensus generation failed after all retries, falling back to majority vote from debate answers...")
                    answers = [ans["answer"].strip().lower() for ans in current_answers]
                    answer_counts = Counter(answers)
                    most_common_answer = answer_counts.most_common(1)[0][0]
                    # Find reasoning from an agent with the most common answer
                    most_common_reasoning = next(
                        (ans["reasoning"] for ans in current_answers if ans["answer"].strip().lower() == most_common_answer),
                        "Consensus generation failed. Using majority answer from debate rounds."
                    )
                    consensus_response = f"<reasoning>{most_common_reasoning}</reasoning>\n<answer>{most_common_answer}</answer>"
        
        # Parse consensus response
        if not consensus_generated and consensus_response:
            print(f"Using fallback consensus: {consensus_response[:100]}...")
        parsed_consensus = self._parse_response(consensus_response, row)
        
        # Create result with debate history
        result = create_tomi_result(parsed_consensus, row)
        result["debate_history"] = debate_history
        result["consensus_reasoning"] = parsed_consensus["reasoning"]
        result["consensus_answer"] = parsed_consensus["answer"]
        
        return result

    async def run_single_experiment(
        self,
        row: dict[str, Any],
        save_result: bool = False,
        continue_mode: str = "new",
    ) -> dict[str, Any]:
        """Run a single experiment in debate mode."""
        # Determine result path
        result_path = Path(
            f"data/tomi_results/debate_{self.model_name.replace('/', '_').replace('.', '_')}/{self.dataset_name}/{row['index']}.json"
        )
        
        # Check if result already exists in continue mode
        if continue_mode == "continue" and result_path.exists():
            with open(result_path, "r") as f:
                result = dict(json.load(f))
                return result
        
        # Run debate experiment
        result = await self._run_debate(row)
        
        # Save result if requested
        if save_result:
            # Ensure the main answer fields are set correctly for evaluation
            result["reasoning"] = result.get("consensus_reasoning", result.get("reasoning", ""))
            result["answer"] = result.get("consensus_answer", result.get("answer", ""))
            self._save_result(result, result_path)
        
        return result


@app.command()
def run_baseline(
    dataset_path: str = typer.Option(
        "./data/rephrased_tomi_test_600.csv",
        help="Path to the ToMI dataset CSV file",
    ),
    model_name: str = typer.Option(
        "o3-2025-04-16",
        help="Model name to use",
    ),
    batch_size: int = typer.Option(
        4,
        help="Batch size for processing",
    ),
    save: bool = typer.Option(
        True,
        help="Whether to save results",
    ),
    continue_mode: str = typer.Option(
        "new",
        help="Whether to continue from existing results (new/continue)",
    ),
) -> None:
    """Run agent debate baseline benchmarks for ToMI dataset."""
    dataset_name = dataset_path.split("/")[-1]
    
    # Load dataset
    try:
        data = pd.read_csv(dataset_path).fillna("")
        # Ensure index is string
        if "index" in data.columns:
            data["index"] = data["index"].astype(str)
    except Exception as e:
        raise ValueError(f"Failed to load dataset: {e}")
    
    asyncio.run(
        _run_baseline_benchmark(
            dataset_name=dataset_name,
            data=data,
            batch_size=batch_size,
            save=save,
            model_name=model_name,
            continue_mode=continue_mode,
        )
    )


async def _run_baseline_benchmark(
    dataset_name: str,
    data: pd.DataFrame,
    batch_size: int,
    save: bool,
    model_name: str,
    continue_mode: str,
) -> None:
    """Async implementation of baseline benchmark runner."""
    runner = AgentBaselineRunner(
        model_name=model_name,
        dataset_name=dataset_name,
    )
    
    print(f"Running ToMI benchmark in debate mode with {len(data)} examples")
    print(f"Using model: {model_name}")
    
    all_results = []
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i : i + batch_size].to_dict("records")
        print(
            f"\nProcessing batch {i//batch_size + 1}/{(len(data) + batch_size - 1)//batch_size}"
        )
        
        tasks = [
            runner.run_single_experiment(
                cast(dict[str, Any], row),
                save_result=save,
                continue_mode=continue_mode,
            )
            for row in batch
        ]
        results = await asyncio.gather(*tasks)
        all_results.extend(results)
    
    # Final evaluation report
    tomi_evaluation_report(all_results)


if __name__ == "__main__":
    app()

