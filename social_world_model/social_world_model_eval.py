"""
Social World Model Evaluation using LLM Judge

Evaluates socialized contexts by using an LLM judge to assess whether
agents observe the right things they should observe, comparing generated
contexts against ground truth.
"""

from pydantic import BaseModel, Field
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import asyncio
from sotopia.generation_utils import agenerate, StrOutputParser


# ============================================================================
# Data Models
# ============================================================================

class EvaluationResult(BaseModel):
    """Result of evaluating a single context pair"""
    file_id: str = Field(description="Filename identifier")
    structural_score: float = Field(description="Schema compliance score (0-1)")
    observation_accuracy: float = Field(description="LLM judge score for observation accuracy (0-1)")
    overall_score: float = Field(description="Composite score (0-1)")
    judge_reasoning: str = Field(description="Judge's reasoning for the score")

    def to_markdown_row(self) -> str:
        """Convert to markdown table row"""
        return (
            f"| {self.file_id:<12} | "
            f"{self.structural_score:.3f} | "
            f"{self.observation_accuracy:.3f} | "
            f"{self.overall_score:.3f} |"
        )


# ============================================================================
# Structural Validation
# ============================================================================

def validate_structure(context_dict: dict) -> float:
    """
    Validate schema compliance of a socialized context.

    Checks:
    - Required fields present (agents_names, socialized_context)
    - Agent name consistency across timesteps
    - Timestep format and sequencing

    Args:
        context_dict: Dictionary containing socialized context

    Returns:
        Structural score between 0 and 1
    """
    score = 0.0

    # Check required fields (30% of score)
    if "agents_names" in context_dict and "socialized_context" in context_dict:
        score += 0.3
    else:
        return score  # Can't continue without these fields

    agents = set(context_dict.get("agents_names", []))
    socialized_context = context_dict.get("socialized_context", [])

    if not socialized_context:
        return score

    # Check agent consistency across timesteps (40% of score)
    agent_consistency_score = 0.0
    for step in socialized_context:
        obs_agents = set(step.get("observations", {}).keys())
        act_agents = set(step.get("actions", {}).keys())

        if obs_agents == agents and act_agents == agents:
            agent_consistency_score += 1.0

    agent_consistency_score /= len(socialized_context)
    score += 0.4 * agent_consistency_score

    # Check timestep format (30% of score)
    try:
        timesteps = [int(step.get("timestep", "0")) for step in socialized_context]
        # Check if sequential (allowing for 0-indexed or 1-indexed)
        if timesteps == list(range(len(timesteps))) or \
           timesteps == list(range(1, len(timesteps) + 1)):
            score += 0.3
        elif sorted(timesteps) == timesteps:  # At least monotonic
            score += 0.15
    except (ValueError, TypeError):
        # Timesteps aren't numeric, give partial credit if they exist
        if all("timestep" in step for step in socialized_context):
            score += 0.1

    return min(score, 1.0)


# ============================================================================
# LLM Judge Evaluation
# ============================================================================

def format_socialized_context(context_dict: dict) -> str:
    """Format socialized context as a readable string for the judge."""
    agents = context_dict.get("agents_names", [])
    steps = context_dict.get("socialized_context", [])
    
    lines = [f"Agents: {', '.join(agents)}", ""]
    
    for i, step in enumerate(steps):
        lines.append(f"Timestep {i+1}:")
        lines.append(f"  State: {step.get('state', 'N/A')}")
        lines.append("  Observations:")
        for agent, obs in step.get("observations", {}).items():
            lines.append(f"    {agent}: {obs}")
        lines.append("  Actions:")
        for agent, action in step.get("actions", {}).items():
            lines.append(f"    {agent}: {action}")
        lines.append("")
    
    return "\n".join(lines)


async def judge_observation_accuracy(
    gt_context: dict,
    gen_context: dict,
    story: str,
    question: str,
    model_name: str = "gpt-4o",
) -> tuple[float, str]:
    """
    Use LLM judge to evaluate whether agents observe the right things.

    Args:
        gt_context: Ground truth socialized context
        gen_context: Generated socialized context
        story: The story/context from the original task
        question: The question being asked
        model_name: LLM model to use as judge

    Returns:
        Tuple of (score 0-1, reasoning)
    """
    gt_formatted = format_socialized_context(gt_context)
    gen_formatted = format_socialized_context(gen_context)
    
    prompt = f"""You are evaluating a socialized context for a Theory of Mind (ToMI) task.

**Task Story:**
{story}

**Question:**
{question}

**Ground Truth Socialized Context:**
{gt_formatted}

**Generated Socialized Context to Evaluate:**
{gen_formatted}

**Your Task:**
Evaluate whether the generated socialized context correctly captures what each agent should observe at each timestep. 

**Key Evaluation Criteria:**
1. **Observation Accuracy (MOST IMPORTANT)**: Do agents observe what they should observe based on:
   - Their location (agents can only observe events in their current location)
   - What happened in previous timesteps (if they were present)
   - What they can actually see vs. what they cannot see (if they left a location)

2. **State Accuracy**: Is the world state correctly described?

3. **Action Accuracy**: Are the actions correctly attributed to agents?

**Critical Focus:**
The most important aspect is whether each agent's observations are correct. An agent should:
- Observe events they witness in their location
- NOT observe events that happen in locations they're not in
- Observe actions they see being performed
- Have correct observations based on their perspective and knowledge

**Output Format:**
Provide a JSON response with:
{{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<detailed explanation of your evaluation, focusing especially on observation accuracy>"
}}

The score should reflect:
- 1.0: Perfect - all observations are correct
- 0.8-0.9: Very good - minor issues
- 0.6-0.7: Good - some incorrect observations
- 0.4-0.5: Fair - several incorrect observations
- 0.0-0.3: Poor - many incorrect observations

Focus especially on whether agents observe things they shouldn't be able to observe, or miss things they should observe.
"""

    try:
        response = await agenerate(
            model_name=model_name,
            template=prompt,
            input_values={},
            output_parser=StrOutputParser(),
            structured_output=False,
        )
        
        # Parse JSON response
        # Try to extract JSON from response (might have markdown code blocks)
        import re
        json_match = re.search(r'\{[^{}]*"score"[^{}]*"reasoning"[^{}]*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
        else:
            # Try to parse the whole response as JSON
            result = json.loads(response)
        
        score = float(result.get("score", 0.0))
        reasoning = result.get("reasoning", "No reasoning provided")
        
        # Clamp score to [0, 1]
        score = max(0.0, min(1.0, score))
        
        return score, reasoning
    except Exception as e:
        print(f"Error in LLM judging: {e}")
        print(f"Response was: {response[:500] if 'response' in locals() else 'No response'}")
        return 0.0, f"Error: {str(e)}"


# ============================================================================
# Main Evaluation Logic
# ============================================================================

async def evaluate_contexts_async(
    gt_dict: dict,
    gen_dict: dict,
    file_id: str = "unknown",
    judge_model: str = "gpt-4o",
) -> EvaluationResult:
    """
    Evaluate a single context pair using LLM judge.

    Args:
        gt_dict: Ground truth context dictionary
        gen_dict: Generated context dictionary
        file_id: Identifier for this evaluation
        judge_model: LLM model to use as judge

    Returns:
        EvaluationResult with all metrics
    """
    # Extract socialized contexts
    gt_sc = gt_dict.get("socialized_context", {})
    gen_sc = gen_dict.get("socialized_context", {})

    # Validate structure
    structural_score = validate_structure(gen_sc)

    # Get story and question for context
    story = gt_dict.get("story", "")
    question = gt_dict.get("question", "")

    # Use LLM judge to evaluate observation accuracy
    observation_accuracy, reasoning = await judge_observation_accuracy(
        gt_sc, gen_sc, story, question, judge_model
    )

    # Overall composite score (30% structural + 70% observation accuracy)
    overall_score = 0.3 * structural_score + 0.7 * observation_accuracy

    return EvaluationResult(
        file_id=file_id,
        structural_score=structural_score,
        observation_accuracy=observation_accuracy,
        overall_score=overall_score,
        judge_reasoning=reasoning
    )


def evaluate_contexts(
    gt_dict: dict,
    gen_dict: dict,
    file_id: str = "unknown",
    judge_model: str = "gpt-4o",
) -> EvaluationResult:
    """
    Synchronous wrapper for async evaluation.
    """
    return asyncio.run(evaluate_contexts_async(gt_dict, gen_dict, file_id, judge_model))


async def evaluate_directory_async(
    gt_dir: Path,
    gen_dir: Path,
    judge_model: str = "gpt-4o",
    batch_size: int = 100,
) -> str:
    """
    Evaluate all matching JSON files in two directories using LLM judge.

    Args:
        gt_dir: Directory containing ground truth JSON files
        gen_dir: Directory containing generated JSON files
        judge_model: LLM model to use as judge
        batch_size: Number of files to process in parallel (default: 100)

    Returns:
        Markdown table string with results
    """
    gt_dir = Path(gt_dir)
    gen_dir = Path(gen_dir)

    # Find all JSON files in ground truth directory
    gt_files = sorted(gt_dir.glob("*.json"))

    if not gt_files:
        return "No JSON files found in ground truth directory."

    # Prepare file pairs
    file_pairs = []
    for gt_file in gt_files:
        file_id = gt_file.stem
        gen_file = gen_dir / f"{file_id}.json"

        if not gen_file.exists():
            print(f"Warning: No matching file for {file_id} in generated directory")
            continue

        file_pairs.append((gt_file, gen_file, file_id))

    if not file_pairs:
        return "No valid file pairs to evaluate."

    # Process in batches
    results = []
    total_batches = (len(file_pairs) + batch_size - 1) // batch_size
    
    for batch_idx in range(0, len(file_pairs), batch_size):
        batch = file_pairs[batch_idx:batch_idx + batch_size]
        batch_num = (batch_idx // batch_size) + 1
        
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} files)...")
        
        # Create async tasks for this batch
        async def evaluate_file_pair(gt_file, gen_file, file_id):
            try:
                # Load JSON files
                with open(gt_file, 'r') as f:
                    gt_dict = json.load(f)
                with open(gen_file, 'r') as f:
                    gen_dict = json.load(f)
                
                # Evaluate
                return await evaluate_contexts_async(gt_dict, gen_dict, file_id, judge_model)
            except json.JSONDecodeError as e:
                print(f"Error loading {file_id}: {e}")
                return None
            except Exception as e:
                print(f"Error evaluating {file_id}: {e}")
                return None
        
        # Run batch in parallel
        batch_results = await asyncio.gather(*[
            evaluate_file_pair(gt_file, gen_file, file_id)
            for gt_file, gen_file, file_id in batch
        ])
        
        # Filter out None results
        results.extend([r for r in batch_results if r is not None])
        
        print(f"Batch {batch_num} complete. Total results: {len(results)}")

    if not results:
        return "No valid file pairs to evaluate."

    # Build markdown table
    header = (
        "| File         | Structural | Observation Accuracy | Overall |\n"
        "|--------------|------------|----------------------|---------|"
    )

    rows = [result.to_markdown_row() for result in results]

    # Calculate mean scores
    mean_structural = np.mean([r.structural_score for r in results])
    mean_obs_accuracy = np.mean([r.observation_accuracy for r in results])
    mean_overall = np.mean([r.overall_score for r in results])

    mean_row = (
        f"| **Mean**     | "
        f"{mean_structural:.3f} | "
        f"{mean_obs_accuracy:.3f} | "
        f"{mean_overall:.3f} |"
    )

    # Final score row
    final_score_row = (
        f"| **Final Score** | - | - | **{mean_overall:.3f}** |"
    )

    # Combine all parts
    table = "\n".join([header] + rows + [mean_row, final_score_row])

    return table


def evaluate_directory(
    gt_dir: Path,
    gen_dir: Path,
    judge_model: str = "gpt-4o",
    batch_size: int = 100,
) -> str:
    """
    Synchronous wrapper for async directory evaluation.
    """
    return asyncio.run(evaluate_directory_async(gt_dir, gen_dir, judge_model, batch_size))


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Command-line interface for evaluation"""
    parser = argparse.ArgumentParser(
        description="Evaluate socialized context generation quality using LLM judge"
    )
    parser.add_argument(
        "--gt-dir",
        required=True,
        help="Directory containing ground truth JSON files"
    )
    parser.add_argument(
        "--gen-dir",
        required=True,
        help="Directory containing generated JSON files"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (optional, prints to stdout if not specified)"
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o",
        help="LLM model to use as judge (default: gpt-4o)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of files to process in parallel (default: 100)"
    )

    args = parser.parse_args()

    # Evaluate
    table = evaluate_directory(Path(args.gt_dir), Path(args.gen_dir), args.judge_model, args.batch_size)

    # Output
    if args.output:
        Path(args.output).write_text(table)
        print(f"Results saved to {args.output}")
    else:
        print(table)


if __name__ == "__main__":
    main()
