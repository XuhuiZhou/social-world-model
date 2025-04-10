# Run the benchmarks for the TOM dataset
import os
import pandas as pd
from rich import print
import json
from pathlib import Path
import logging
import asyncio
from sotopia.generation_utils import StrOutputParser, agenerate
from typing import Any, Literal, get_args, Optional, cast
from rich.logging import RichHandler
from social_world_model.social_world_model import SocialWorldModel
from social_world_model.database import SocializedContext, SocialSimulation
from social_world_model.task_modules import (
    tomi_simulation,
    fantom_simulation,
    flatten_fantom_data,
    confaide_simulation,
    hitom_simulation,
    prepare_tomi_vanilla,
    prepare_fantom_vanilla,
    prepare_confaide_vanilla,
    prepare_hitom_vanilla,
    create_tomi_result,
    create_fantom_result,
    create_confaide_result,
    create_hitom_result,
    tomi_evaluation_report,
    fantom_evaluation_report,
    confaide_evaluation_report,
    hitom_evaluation_report,
    TOMI_SOCIALIZED_CONTEXT_PROMPT,
    FANTOM_SOCIALIZED_CONTEXT_PROMPT,
    CONFAIDE_SOCIALIZED_CONTEXT_PROMPT,
    COBRA_FRAMES_SOCIALIZED_CONTEXT_PROMPT,
    cobra_frames_simulation,
    prepare_cobra_frames_vanilla,
    create_cobra_frames_result,
    cobra_frames_evaluation_report,
    HITOM_SOCIALIZED_CONTEXT_PROMPT,
    mmtom_simulation,
    prepare_mmtom_vanilla,
    create_mmtom_result,
    mmtom_evaluation_report,
    MMTOM_SOCIALIZED_CONTEXT_PROMPT,
)
from social_world_model.engine import load_existing_socialized_contexts
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

# Create type aliases using the constants
ModeType = Literal[
    "vanilla",
    "socialized_context",
    "pure_context",
    "simulation",
    "generate_socialized_context",
]
ContextModeType = Literal["socialized_context", "simulation"]
ContinueModeType = Literal["new", "continue"]
BenchmarkType = Literal["tomi", "fantom", "confaide", "cobra_frames", "hitom" "mmtom"]

SocializedContextPrompt = {
    "tomi": TOMI_SOCIALIZED_CONTEXT_PROMPT,
    "fantom": FANTOM_SOCIALIZED_CONTEXT_PROMPT,
    "confaide": CONFAIDE_SOCIALIZED_CONTEXT_PROMPT,
    "hitom": HITOM_SOCIALIZED_CONTEXT_PROMPT,
    "cobra_frames": COBRA_FRAMES_SOCIALIZED_CONTEXT_PROMPT,
    "mmtom": MMTOM_SOCIALIZED_CONTEXT_PROMPT,
}


class ToMBenchmarkRunner:
    def __init__(
        self,
        model_name: str = "gpt-4-mini",
        dataset_name: str = "tomi",
        existing_socialized_contexts_path: Optional[dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.existing_socialized_contexts: dict[str, SocializedContext] = {}
        self.existing_social_simulations: dict[str, SocialSimulation] = {}
        if existing_socialized_contexts_path and os.path.exists(
            existing_socialized_contexts_path["data_path"]
        ):
            self.existing_socialized_contexts, self.existing_social_simulations = (
                load_existing_socialized_contexts(
                    existing_socialized_contexts_path["data_path"],
                    existing_socialized_contexts_path["identifier_key"],
                )
            )

    async def run_single_experiment(
        self,
        row: dict[str, Any],
        benchmark_type: BenchmarkType,
        save_result: bool = False,
        mode: ModeType = "vanilla",
        continue_mode: ContinueModeType = "new",
        example_analysis_file: str = "",
    ) -> dict[str, Any]:
        """Run a single experiment for either ToMi or FANToM benchmark."""
        # Define result path regardless of continue mode
        engine = SocialWorldModel(
            model_name=self.model_name,
            existing_socialized_contexts=self.existing_socialized_contexts,
            existing_social_simulations=self.existing_social_simulations,
        )

        save_dir = Path(
            f"data/{benchmark_type}_results/{mode}_{self.model_name}_{self.dataset_name}"
        )
        result_path = save_dir / f"{row['index']}.json"

        # Check for cached results if in continue mode
        if continue_mode == "continue" and result_path.exists():
            print(f"Loading cached result for index {row['index']}")
            with open(result_path) as f:
                result = dict(json.load(f))
                return result

        # If no cached result or in new mode, run the experiment
        if mode == "vanilla":
            result = await self._run_vanilla(row, benchmark_type)
        elif mode == "pure_context":
            result = await self._run_socialized_context(
                row,
                benchmark_type,
                example_analysis_file,
                pure_context=True,
                engine=engine,
            )
        elif mode == "socialized_context":
            result = await self._run_socialized_context(
                row, benchmark_type, example_analysis_file, engine=engine
            )
        elif mode == "simulation":
            result = await self._run_simulation(row, benchmark_type, engine=engine)
        if save_result:
            if "socialized_context" in result and (
                isinstance(result["socialized_context"], SocializedContext)
                or isinstance(result["socialized_context"], SocialSimulation)
            ):
                result["socialized_context"] = result["socialized_context"].model_dump()
            self._save_result(result, result_path)
        return result

    async def _run_vanilla(
        self, row: dict[str, Any], benchmark_type: str, pure_context: bool = False
    ) -> dict[str, Any]:
        """Run experiment in vanilla mode (direct LLM generation)."""
        # Prepare context and question based on benchmark type
        if benchmark_type == "tomi":
            template, input_values = prepare_tomi_vanilla(row, pure_context)
        elif benchmark_type == "fantom":  # fantom
            template, input_values = prepare_fantom_vanilla(row, pure_context)
        elif benchmark_type == "confaide":  # confaide
            template, input_values = prepare_confaide_vanilla(row, pure_context)
        elif benchmark_type == "cobra_frames":
            template, input_values = prepare_cobra_frames_vanilla(row, pure_context)
        elif benchmark_type == "hitom":
            template, input_values = prepare_hitom_vanilla(row, pure_context)
        elif benchmark_type == "mmtom":
            template, input_values = prepare_mmtom_vanilla(row, pure_context)
        # Generate response
        response = await agenerate(
            model_name=self.model_name,
            template=template,
            input_values=input_values,
            temperature=0.0,
            output_parser=StrOutputParser(),
            structured_output=False,
        )
        # Parse response and create result
        if benchmark_type == "tomi":
            parsed_result = self._parse_response(response, row)
            result = create_tomi_result(parsed_result, row)
        elif benchmark_type == "fantom":
            parsed_result = self._parse_response(response, row)
            result = create_fantom_result(parsed_result, row)
        elif benchmark_type == "confaide":
            parsed_result = self._parse_response(response, row)
            result = create_confaide_result(parsed_result, row)
        elif benchmark_type == "cobra_frames":
            parsed_result = self._parse_response(response, row)
            result = create_cobra_frames_result(parsed_result, row)
        elif benchmark_type == "hitom":
            parsed_result = self._parse_response(response, row)
            result = create_hitom_result(parsed_result, row)
        elif benchmark_type == "mmtom":
            parsed_result = self._parse_response(response, row)
            result = create_mmtom_result(parsed_result, row)
        return result

    async def _run_socialized_context(
        self,
        row: dict[str, Any],
        benchmark_type: str,
        example_analysis_file: str = "",
        pure_context: bool = False,
        engine: Optional[SocialWorldModel] = None,
    ) -> dict[str, Any]:
        """Run experiment in socialized_context mode (using ToM engine for memory tracking)."""
        assert isinstance(
            engine, SocialWorldModel
        ), "Engine must be an instance of ToMEngine"
        if benchmark_type in []:
            critic_and_improve = True
        else:
            critic_and_improve = False

        if benchmark_type == "tomi":
            context = " ".join(eval(row["story"]))
        elif benchmark_type == "hitom":
            context = row["story"]
        else:
            context = row["context"]
        engine.set_task_specific_instructions(SocializedContextPrompt[benchmark_type])
        if example_analysis_file:
            example_analysis = json.load(open(example_analysis_file))
            example_analysis = str(example_analysis)
        else:
            example_analysis = ""
        if (
            benchmark_type
            in [
                "fantom",
                "confaide",
                "hitom",
            ]
        ):  # Both FANToM and ConFaIde have repeated set_ids, so we cache the socialized contexts
            if row["set_id"] in engine.existing_socialized_contexts:
                socialized_context = engine.existing_socialized_contexts[row["set_id"]]
            else:
                socialized_context = await engine.socialize_context(
                    context, example_analysis, critic_and_improve=critic_and_improve
                )
                engine.existing_socialized_contexts[row["set_id"]] = socialized_context
        else:
            if row["index"] in engine.existing_socialized_contexts:
                socialized_context = engine.existing_socialized_contexts[row["index"]]
            else:
                socialized_context = await engine.socialize_context(
                    context, example_analysis, critic_and_improve=critic_and_improve
                )
                engine.existing_socialized_contexts[row["index"]] = socialized_context
        row["socialized_context"] = socialized_context
        row["extra_info"] = socialized_context.to_natural_language()
        result = await self._run_vanilla(row, benchmark_type, pure_context=pure_context)
        return result

    async def _run_simulation(
        self,
        row: dict[str, Any],
        benchmark_type: str,
        engine: Optional[SocialWorldModel] = None,
    ) -> dict[str, Any]:
        """Run experiment in simulation mode (using ToM engine for memory tracking)."""
        assert isinstance(
            engine, SocialWorldModel
        ), "Engine must be an instance of ToMEngine"
        engine.set_task_specific_instructions(SocializedContextPrompt[benchmark_type])
        if benchmark_type == "tomi":
            assert (
                str(row["index"]) in engine.existing_socialized_contexts
            ), f"Socialized context for index {row['index']} not found"
            result = await tomi_simulation(row, engine)
        elif benchmark_type == "fantom":
            parsed_result = await fantom_simulation(row, engine)
            result = create_fantom_result(parsed_result, row)
        elif benchmark_type == "confaide":
            parsed_result = await confaide_simulation(row, engine)
            result = create_confaide_result(parsed_result, row)
        elif benchmark_type == "cobra_frames":
            await cobra_frames_simulation(row, engine)
            result = await self._run_vanilla(row, benchmark_type)
        elif benchmark_type == "hitom":
            parsed_result = await hitom_simulation(row, engine)
            result = create_hitom_result(parsed_result, row)
        elif benchmark_type == "mmtom":
            parsed_result = await mmtom_simulation(row, engine)
            result = create_mmtom_result(parsed_result, row)
        else:
            result = await self._run_vanilla(row, benchmark_type)
        if not result:
            result = await self._run_vanilla(row, benchmark_type)
        return result

    def _parse_response(self, response: str, row: dict[str, Any]) -> dict[str, Any]:
        """Parse ToMi response and create result dictionary."""
        try:
            reasoning = response.split("<reasoning>")[1].split("</reasoning>")[0].strip()
            answer = response.split("<answer>")[1].split("</answer>")[0].strip()
        except Exception as e:
            print(f"Failed to parse response: {e}")
            reasoning = "Failed to parse reasoning"
            # For MMTom, try to extract just the answer letter (a/b) if the full parsing fails
            if "mmtom" in str(row.get("question_type", "")):
                answer = response.strip().lower()[-1]  # Get last character
                if answer not in ["a", "b"]:
                    answer = "a"  # Default to a if not a valid answer
            else:
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


def validate_benchmark_type(value: str) -> str:
    """Validate benchmark type."""
    if value not in get_args(BenchmarkType):
        raise typer.BadParameter(
            f"Benchmark type must be one of {get_args(BenchmarkType)}"
        )
    return value


def validate_mode(value: str) -> str:
    """Validate mode."""
    if value not in get_args(ModeType):
        raise typer.BadParameter(f"Mode must be one of {get_args(ModeType)}")
    return value


def validate_continue_mode(value: str) -> str:
    """Validate continue mode."""
    if value not in get_args(ContinueModeType):
        raise typer.BadParameter(
            f"Continue mode must be one of {get_args(ContinueModeType)}"
        )
    return value


def validate_context_mode(value: str) -> str:
    """Validate context mode."""
    if value not in get_args(ContextModeType):
        raise typer.BadParameter(
            f"Context mode must be one of {get_args(ContextModeType)}"
        )
    return value


@app.command()
def run_benchmark(
    benchmark_type: str = typer.Argument(
        ...,
        help="Type of benchmark to run (tomi/fantom/confaide/hitom/mmtom)",
        callback=validate_benchmark_type,
    ),
    dataset_path: Optional[str] = None,
    batch_size: int = 4,
    save: bool = True,
    model_name: str = "o1-2024-12-17",
    mode: str = typer.Option(
        "vanilla",
        help="Mode to run in (vanilla/socialized_context/pure_context/simulation/generate_socialized_context; you need to run generate_socialized_context first to use simulation mode)",
        callback=validate_mode,
    ),
    continue_mode: str = typer.Option(
        "new",
        help="Whether to continue from existing results (new/continue)",
        callback=validate_continue_mode,
    ),
    example_analysis_file: str = typer.Option(
        "", help="Path to the example analysis file"
    ),
    context_model: str = typer.Option(
        "o1-2024-12-17",
        help="Model to use for context generation",
    ),
) -> None:
    """Run benchmark experiments."""
    if dataset_path is None:
        dataset_path = {
            "tomi": "./data/rephrased_tomi_test_600.csv",
            "fantom": "./data/fantom_data/fantom_for_tt_processed.jsonl",
            "confaide": "./data/confaide_data/confaide.jsonl",
            "cobra_frames": "./data/cobra_data/cobra_frames_adv.jsonl",
            "hitom": "./data/hitom_data/processed_hitom_data.csv",
            "mmtom": "./data/mmtom-qa/questions.jsonl",
        }[benchmark_type]

    dataset_name = dataset_path.split("/")[-1]
    try:
        data = pd.read_csv(dataset_path).fillna("")
        # Ensure index is string
        if "index" in data.columns:
            data["index"] = data["index"].astype(str)
        if "set_id" in data.columns:
            data["set_id"] = data["set_id"].astype(str)
    except Exception as e:
        # Load jsonl file for fantom, confaide, and mmtom datasets
        if dataset_path.endswith(".jsonl"):
            data_list = []
            if benchmark_type == "mmtom":
                # For MMTom, read the entire file and split on newlines
                with open(dataset_path, "r") as f:
                    content = f.read()
                    # Split on newlines and filter out empty lines
                    json_objects = [obj.strip() for obj in content.split("\n") if obj.strip()]
                    for obj in json_objects:
                        try:
                            entry = json.loads(obj)
                            # Ensure required fields are present
                            entry["index"] = str(len(data_list))
                            entry["question_type"] = entry.get("question_type", "")
                            entry["episode"] = entry.get("episode", "")
                            entry["answer"] = entry.get("answer", "")  # Ensure answer field exists
                            data_list.append(entry)
                        except json.JSONDecodeError:
                            pass  # Skip invalid JSON
            else:
                # For other datasets, assume one JSON object per line
                with open(dataset_path, "r") as f:
                    for line in f:
                        entry = json.loads(line)
                        if benchmark_type == "fantom":
                            data_list += flatten_fantom_data(entry)
                        else:
                            # For confaide, we assume the data is already flattened
                            data_list.append(entry)
            data = pd.DataFrame(data_list)
            data["index"] = [str(i) for i in range(len(data))]

        elif dataset_path.endswith(".json"):
            with open(dataset_path, "r") as f:
                data_list = json.load(f)
                data = pd.DataFrame(data_list)
        else:
            raise ValueError(f"Data set in a different format: {e}")
        if "set_id" in data.columns:
            data["set_id"] = data["set_id"].astype(str)
    if mode == "generate_socialized_context":
        # For fantom and confaide, select a subset of unique set_ids
        if benchmark_type in ["fantom", "confaide", "hitom"]:
            data = data.groupby("set_id").head(1).reset_index(drop=True)
            mode = "socialized_context"
    asyncio.run(
        _run_benchmark(
            benchmark_type=benchmark_type,
            dataset_name=dataset_name,
            data=data,
            batch_size=batch_size,
            save=save,
            model_name=model_name,
            mode=mode,
            continue_mode=continue_mode,
            example_analysis_file=example_analysis_file,
            context_model=context_model,
            load_contexts=mode != "vanilla",  # Skip loading contexts in vanilla mode
        )
    )


async def _run_benchmark(
    benchmark_type: str,
    dataset_name: str,
    data: pd.DataFrame,
    batch_size: int,
    save: bool,
    model_name: str,
    mode: str,
    context_model: str = "o1-2024-12-17",
    continue_mode: str = "new",
    example_analysis_file: str = "",
    load_contexts: bool = True,
) -> None:
    """Async implementation of benchmark runner."""
    runner = ToMBenchmarkRunner(
        model_name,
        dataset_name=dataset_name,
        existing_socialized_contexts_path={
            "data_path": Path(
                f"data/{benchmark_type}_results/socialized_contexts_{context_model}_{dataset_name}"
            ),
            "identifier_key": (
                "set_id" if benchmark_type in ["fantom", "confaide", "hitom"] else None
            ),
        },
    )
    print(f"Running {benchmark_type.upper()} benchmark with {len(data)} examples")
    all_results = []
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i : i + batch_size].to_dict("records")
        print(
            f"\nProcessing batch {i//batch_size + 1}/{(len(data) + batch_size - 1)//batch_size}"
        )

        tasks = [
            runner.run_single_experiment(
                cast(dict[str, Any], row),
                benchmark_type=cast(
                    BenchmarkType,
                    (
                        benchmark_type
                        if benchmark_type in get_args(BenchmarkType)
                        else "tomi"
                    ),
                ),
                save_result=save,
                mode=cast(ModeType, mode if mode in get_args(ModeType) else "vanilla"),
                continue_mode=cast(
                    ContinueModeType,
                    (
                        continue_mode
                        if continue_mode in get_args(ContinueModeType)
                        else "new"
                    ),
                ),
                example_analysis_file=example_analysis_file,
            )
            for row in batch
        ]
        results = await asyncio.gather(*tasks)
        all_results.extend(results)
    
    # Final evaluation report
    if benchmark_type == "tomi":
        tomi_evaluation_report(all_results)
    elif benchmark_type == "fantom":
        fantom_evaluation_report(all_results)
    elif benchmark_type == "confaide":
        confaide_evaluation_report(all_results)
    elif benchmark_type == "cobra_frames":
        cobra_frames_evaluation_report(all_results)
    elif benchmark_type == "hitom":
        hitom_evaluation_report(all_results)
    elif benchmark_type == "mmtom":
        mmtom_evaluation_report(all_results)


if __name__ == "__main__":
    app()
