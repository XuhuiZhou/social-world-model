"""
Run socialized context evaluation from project root.

Evaluates socialized contexts by using an LLM judge to assess whether
agents observe the right things they should observe, comparing generated
contexts against ground truth.
"""

import argparse
import asyncio
import json
import numpy as np
from pathlib import Path
from social_world_model.social_world_model_eval import (
    evaluate_contexts_async,
    EvaluationResult,
)
from typing import Optional, Any


async def evaluate_file_pair(
    gt_file: Path, gen_file: Path, file_id: str, judge_model: str
) -> Optional[EvaluationResult]:
    """Evaluate a single file pair."""
    try:
        # Load JSON files
        with open(gt_file, "r") as f:
            gt_dict: dict[str, Any] = json.load(f)
        with open(gen_file, "r") as f:
            gen_dict: dict[str, Any] = json.load(f)

        # Evaluate
        return await evaluate_contexts_async(gt_dict, gen_dict, file_id, judge_model)
    except json.JSONDecodeError as e:
        print(f"Error loading {file_id}: {e}")
        return None
    except Exception as e:
        print(f"Error evaluating {file_id}: {e}")
        return None


async def _evaluate_directory_async(
    gt_dir: Path,
    gen_dir: Path,
    judge_model: str,
    batch_size: int,
    output: Optional[str] = None,
) -> None:
    """
    Async implementation of directory evaluation.
    """
    gt_dir = Path(gt_dir)
    gen_dir = Path(gen_dir)

    # Find all JSON files in ground truth directory
    gt_files = sorted(gt_dir.glob("*.json"))

    if not gt_files:
        print("No JSON files found in ground truth directory.")
        return

    # Prepare file pairs
    file_pairs = []
    gt_files = gt_files[:5]
    for gt_file in gt_files:
        file_id = gt_file.stem
        gen_file = gen_dir / f"{file_id}.json"

        if not gen_file.exists():
            print(f"Warning: No matching file for {file_id} in generated directory")
            continue

        file_pairs.append((gt_file, gen_file, file_id))

    if not file_pairs:
        print("No valid file pairs to evaluate.")
        return

    # Process in batches
    results = []
    total_batches = (len(file_pairs) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(file_pairs), batch_size):
        batch = file_pairs[batch_idx : batch_idx + batch_size]
        batch_num = (batch_idx // batch_size) + 1

        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} files)...")

        # Run batch in parallel
        tasks = [
            evaluate_file_pair(gt_file, gen_file, file_id, judge_model)
            for gt_file, gen_file, file_id in batch
        ]
        batch_results = await asyncio.gather(*tasks)

        # Filter out None results
        results.extend([r for r in batch_results if r is not None])

        print(f"Batch {batch_num} complete. Total results: {len(results)}")

    print(
        f"\nAll batches complete. Building markdown table for {len(results)} results..."
    )

    if not results:
        print("No valid results from evaluation.")
        return

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
    final_score_row = f"| **Final Score** | - | - | **{mean_overall:.3f}** |"

    # Combine all parts
    table = "\n".join([header] + rows + [mean_row, final_score_row])

    if output:
        Path(output).write_text(table)
        print(f"Results saved to {output}")
    else:
        print(table)


def main() -> None:
    """Command-line interface for evaluation"""
    parser = argparse.ArgumentParser(
        description="Evaluate socialized context generation quality using LLM judge"
    )
    parser.add_argument(
        "--gt-dir", required=True, help="Directory containing ground truth JSON files"
    )
    parser.add_argument(
        "--gen-dir", required=True, help="Directory containing generated JSON files"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (optional, prints to stdout if not specified)",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o",
        help="LLM model to use as judge (default: gpt-4o)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of files to process in parallel (default: 100)",
    )

    args = parser.parse_args()

    print("Starting evaluation...")
    print(f"Ground truth dir: {args.gt_dir}")
    print(f"Generated dir: {args.gen_dir}")
    print(f"Judge model: {args.judge_model}")
    print(f"Batch size: {args.batch_size}\n")

    asyncio.run(
        _evaluate_directory_async(
            Path(args.gt_dir),
            Path(args.gen_dir),
            args.judge_model,
            args.batch_size,
            args.output,
        )
    )


if __name__ == "__main__":
    main()
