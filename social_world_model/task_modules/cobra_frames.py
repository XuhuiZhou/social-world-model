from typing import Any, Optional
from social_world_model.social_world_model import SocialWorldModel
from social_world_model.database import SocialSimulation

COBRA_FRAMES_SOCIALIZED_CONTEXT_PROMPT = """IMPORTANT: The statement should be the first action of the speaker. And agents would be able to observe that before the action. Try to use <mental_state>...</mental_state> tags to describe the mental state of each agent.
"""

def create_cobra_frames_result(
    parsed_result: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    """Create COBRA frames result dictionary."""
    targeted_entries = [
        "index",
        "scenario",
        "listener",
        "speaker",
        "statement",
        "context",
        "offensive_allapp",
        "reasoning",
        "answer",
        "socialized_context",
        "correct",
    ]
    parsed_result["offensive_allapp"] = row["offensive_allapp"]
    if "answer" in parsed_result:
        parsed_result = evaluate_cobra_frames(parsed_result)
    result = {}
    for entry in targeted_entries:
        if entry in parsed_result:
            result[entry] = parsed_result[entry]
        elif entry in row:
            result[entry] = row[entry]
        else:
            continue
    return result


def prepare_cobra_frames_vanilla(
    row: dict[str, Any], pure_context: bool = False
) -> tuple[str, dict[str, Any]]:
    context = row.get("context", "")
    extra_info = row.get("extra_info", "")

    if extra_info:
        if pure_context:
            context = extra_info
            extra_info = ""

    template = """
## Context
{context}

## Extra Info
(to help you better understand the scenario)
{extra_info}

## Task
Analyze whether the statement is offensive in this context.
Don't just focus on literal meaning of the statement, and use the simulations in the Extra Info for your analysis.
Format your response with:
1. Reasoning in <reasoning>...</reasoning> tags
2. Final answer (0 for not offensive, 1 for offensive) in <answer>...</answer> tags
"""
    input_values = {
        "context": context,
        "extra_info": extra_info,
    }

    return template, input_values


def evaluate_cobra_frames(row: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the COBRA frames prediction against the ground truth."""
    predicted = int(row["answer"])
    ground_truth = int(row["offensive_allapp"])
    row["correct"] = 1 if predicted == ground_truth else 0
    return row


def cobra_frames_evaluation_report(results: list[dict[str, Any]]) -> None:
    """Generate evaluation report for COBRA frames task."""
    total = len(results)
    correct = sum(result["correct"] for result in results)
    accuracy = correct / total if total > 0 else 0

    print("\nCOBRA Frames Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

    # Additional metrics could be added here
    # For example: precision, recall, F1 score for offensive vs non-offensive classification


async def cobra_frames_simulation(
    row: dict[str, Any], engine: Optional[SocialWorldModel] = None, num_simulations: int = 2
) -> None:
    """Run simulation for COBRA frames task."""
    assert engine is not None, "Engine must be provided"
    social_simulation = SocialSimulation(simulations=[])
    if row["index"] in engine.existing_social_simulations and len(engine.existing_social_simulations[row["index"]].simulations) >= num_simulations:
        social_simulation = engine.existing_social_simulations[row["index"]]
    else:
        social_simulation = SocialSimulation(simulations=[])
        for _ in range(num_simulations):
            social_simulation = await engine.simulate_socialized_context(
                context=row["context"],
                social_simulation=social_simulation,
            )
    engine.existing_social_simulations[row["index"]] = social_simulation
    row["socialized_context"] = social_simulation
    row["extra_info"] = social_simulation.to_natural_language()
