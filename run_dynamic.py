import typer
from sotopia.cli import benchmark
from social_world_model.agents import SocialWorldModelAgent
from sotopia.agents import LLMAgent
from typing import Any, Type
from typing_extensions import Annotated

app = typer.Typer(pretty_exceptions_enable=False)


class CustomSocialWorldModelAgent(SocialWorldModelAgent):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Set default social_world_model_name if not provided
        if "social_world_model_name" not in kwargs:
            kwargs["social_world_model_name"] = "gpt-4.1-2025-04-14"
        super().__init__(*args, **kwargs)


def get_agent_class(agent_type: str) -> Type[Any]:
    """Get the agent class based on the type string."""
    if agent_type == "social_world_model":
        return CustomSocialWorldModelAgent
    elif agent_type == "vanilla":
        return LLMAgent
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


@app.command(name="run-dynamic-benchmark")
def run_dynamic_benchmark(
    models: Annotated[
        str, typer.Option(help="Comma-separated list of models to benchmark")
    ] = "gpt-4.1-2025-04-14",
    partner_model: Annotated[
        str, typer.Option(help="Partner model to use")
    ] = "gpt-4o-2024-08-06",
    experiment_tag: Annotated[
        str, typer.Option(help="Tag for the benchmark run")
    ] = "social_world_model_trial_22_social_world_model_agent",
    batch_size: Annotated[int, typer.Option(help="Batch size for processing")] = 100,
    push_to_db: Annotated[
        bool, typer.Option(help="Whether to push results to database")
    ] = True,
    evaluator_model: Annotated[
        str, typer.Option(help="Model to use for evaluation")
    ] = "o3-2025-04-16",
    task: Annotated[str, typer.Option(help="Task difficulty level")] = "hard",
    agent_type: Annotated[
        str, typer.Option(help="Type of agent to use (social_world_model or vanilla)")
    ] = "social_world_model",
    social_world_model_name: Annotated[
        str, typer.Option(help="Name of the social world model to use")
    ] = "gpt-4.1-2025-04-14",
) -> None:
    """Run benchmark with Social World Model agents."""
    # Validate agent_type
    if agent_type not in ["social_world_model", "vanilla"]:
        raise typer.BadParameter(
            "Agent type must be either 'social_world_model' or 'vanilla'"
        )

    agent_class = get_agent_class(agent_type)

    # If using CustomSocialWorldModelAgent, pass the social_world_model_name
    if agent_type == "social_world_model":
        def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
            kwargs["social_world_model_name"] = social_world_model_name
            super(CustomSocialWorldModelAgent, self).__init__(*args, **kwargs)
        
        agent_class = type(
            "SocialWorldModelAgent",
            (CustomSocialWorldModelAgent,),
            {"__init__": __init__}
        )

    benchmark(
        models=models.split(","),
        partner_model=partner_model,
        agent_class=agent_class,
        tag=experiment_tag,
        batch_size=batch_size,
        push_to_db=push_to_db,
        evaluator_model=evaluator_model,
        task=task,
    )


if __name__ == "__main__":
    app()
