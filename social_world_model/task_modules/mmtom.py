from social_world_model.social_world_model import SocialWorldModel
from typing import Any, Optional
import pandas as pd

MMTOM_SOCIALIZED_CONTEXT_PROMPT = """You are dissecting the MMTom scenarios. The assumptions are that the characters can perceive every scene in their location but not scenes occurring elsewhere. If the agent leaves the location, they cannot perceive the scene in that location anymore. In the agent's observation, remember to include the objects' locations if the agents are in the same location as the object."""


def prepare_mmtom_vanilla(
    row: dict[str, Any], pure_context: bool = False, with_reasoning: bool = True
) -> tuple[str, dict[str, Any]]:
    """Prepare the vanilla prompt for MMTom dataset."""
    context = row["context"]

    extra_info = row.get("extra_info", "")
    if extra_info:
        if pure_context:
            context = extra_info
            extra_info = ""

    question = row["question"]

    if with_reasoning:
        template = """Imagine that you are an observer in the scenario. Assume that the characters can perceive every scene in their location but not scenes occurring elsewhere. If something is being moved, that means it is not in its original location anymore. You should carefully analyze the character's actions and beliefs based on their observations. Provide your reasoning within the <reasoning></reasoning> tag. For the answer, use <answer>(put your answer here)</answer> and only include the letter (a or b) of your chosen option.

Below is the context and question (and optional extra information):
## Context
{context}
## Extra Information
(to help you better understand and answer the question)
{extra_info}
## Question
{question}"""
    else:
        template = """Imagine that you are an observer in the scenario. Assume that the characters can perceive every scene in their location but not scenes occurring elsewhere. If something is being moved, that means it is not in its original location anymore. You should carefully analyze the character's actions and beliefs based on their observations. For the answer, use <answer>(put your answer here)</answer> and only include the letter (a or b) of your chosen option.

Below is the context and question (and optional extra information):
## Context
{context}
## Extra Information
(to help you better understand and answer the question)
{extra_info}
## Question
{question}"""

    input_values = {
        "context": context,
        "question": question,
        "extra_info": extra_info,
    }

    return template, input_values


def create_mmtom_result(
    parsed_result: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    """Create MMTom result dictionary."""
    parsed_result["answer"] = (
        parsed_result["answer"].replace("<answer>", "").replace("</answer>", "").strip()
    )
    result = {
        "index": row["index"],
        "context": row["context"],
        "question": row["question"],
        "reasoning": parsed_result.get("reasoning", ""),
        "answer": parsed_result.get("answer", ""),
        "correct_answer": row["answer"],
        "is_correct": parsed_result.get("answer", "").lower() == row["answer"].lower(),
        "question_type": row.get("question_type", ""),
        "test": row.get("test", ""),
        "episode": row.get("episode", ""),
        "start_time": row.get("start_time", ""),
        "end_time": row.get("end_time", ""),
        "answer_list": row.get("answer_list", []),
    }

    # Add socialized_context and extra_info if they exist
    if "socialized_context" in row:
        result["socialized_context"] = row["socialized_context"]
    if "extra_info" in row:
        result["extra_info"] = row["extra_info"]

    return result


class MMTomEvalAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def score_and_analyze(
        self, df: pd.DataFrame, target_scenario: str = "all"
    ) -> dict[str, Any]:
        """
        Aggregates scores and performs analysis on the model responses and evaluation results.
        Args:
            df (pandas.DataFrame): The dataframe containing evaluation results
            target_scenario (str): Either 'all' or specific question type
        Returns:
            dict: Report containing various metrics and analyses
        """
        report = {}
        # Filter by question type if specified
        if target_scenario != "all":
            df = df[df["question_type"] == float(target_scenario)]

        # Overall accuracy
        report["overall_accuracy"] = [df["is_correct"].mean(), len(df)]

        # Accuracy by question type
        for qtype in df["question_type"].unique():
            type_df = df[df["question_type"] == qtype]
            report[f"type_{qtype}_accuracy"] = [
                type_df["is_correct"].mean(),
                len(type_df),
            ]

        # Accuracy by episode
        for episode in df["episode"].unique():
            episode_df = df[df["episode"] == episode]
            report[f"episode_{episode}_accuracy"] = [
                episode_df["is_correct"].mean(),
                len(episode_df),
            ]

        return report


def mmtom_evaluation_report(results: list[dict[str, Any]]) -> None:
    """Evaluate MMTom results."""
    eval_agent = MMTomEvalAgent("mmtom")
    df = pd.DataFrame(results)

    # Overall report
    overall_report = eval_agent.score_and_analyze(df)
    print("\nOverall Results:")
    print("---------------")
    for metric, (score, count) in overall_report.items():
        print(f"{metric}: {score:.2%} ({count} examples)")

    # Report by question type
    for qtype in df["question_type"].unique():
        type_report = eval_agent.score_and_analyze(df, str(qtype))
        print(f"\nResults for Question Type {qtype}:")
        print("---------------------------")
        for metric, (score, count) in type_report.items():
            print(f"{metric}: {score:.2%} ({count} examples)")


async def mmtom_simulation(
    row: dict[str, Any], engine: Optional[SocialWorldModel] = None
) -> dict[str, Any]:
    """Run MMTom simulation."""
    assert engine is not None, "Engine must be provided"
    # Run simulation
    result = {
        "question": row["question"],
        "reasoning": "",
        "answer": "",
        "correct_answer": row["answer"],
        "is_correct": False,
    }
    return result
