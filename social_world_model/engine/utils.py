from typing import Any
from social_world_model.database import (
    SocializedContextForModel,
    SocializedContext,
    SocialSimulation,
)
import json
from pathlib import Path


GENERAL_GUIDELINES = """
If all the agents observations are <same_as_state /> or <same_as_last_action_x /> (x is the index of the agent), then the socialized context could be potentially problematic.

If all the agents show no <mental_state> </mental_state> tags, then the socialized context could be potentially problematic.

If none of the agents act in a specific time step (except the last time step), then the socialized context could be potentially problematic.

If none of the agents observe anything in a specific time step (except step 0), then the socialized context could be potentially problematic. (as the agent should be able to observe what they did in the last time step)

If any agent has actions in the last timestep, then the socialized context could be potentially problematic (as the last timestep should be only about the observation of the last actions)
"""


GENERAL_GUIDELINES = """
If all the agents observations are <same_as_state /> or <same_as_last_action_x /> (x is the index of the agent), then the socialized context could be potentially problematic.

If all the agents show no <mental_state> </mental_state> tags, then the socialized context could be potentially problematic.

If none of the agents act in a specific time step (except the last time step), then the socialized context could be potentially problematic.

If none of the agents observe anything in a specific time step (except step 0), then the socialized context could be potentially problematic. (as the agent should be able to observe what they did in the last time step)

If any agent has actions in the last timestep, then the socialized context could be potentially problematic (as the last timestep should be only about the observation of the last actions)
"""


def load_existing_socialized_contexts(
    data_path: Path, identifier_key: str
) -> tuple[dict[str, SocializedContext], dict[str, SocialSimulation]]:
    existing_socialized_contexts: dict[str, SocializedContext] = {}
    existing_social_simulations: dict[str, SocialSimulation] = {}
    for file in data_path.glob("*.json"):
        with open(file, "r") as f:
            simulation_dict = json.load(f)
            if "simulations" in simulation_dict["socialized_context"]:
                # Handle SocialSimulation type
                try:
                    social_simulation = SocialSimulation(**simulation_dict)
                except Exception:
                    # If direct parsing fails, try to process each simulation individually
                    simulations: list[SocializedContext] = []
                    for sim in simulation_dict["socialized_context"]["simulations"]:
                        try:
                            socialized_context = SocializedContext(**sim)
                            simulations.append(socialized_context)
                        except Exception:
                            socialized_context_dict = dictlize(sim)
                            socialized_context = SocializedContext(
                                **socialized_context_dict
                            )
                            simulations.append(socialized_context)
                    social_simulation = SocialSimulation(simulations=simulations)

                if identifier_key:
                    existing_social_simulations[simulation_dict[identifier_key]] = (
                        social_simulation
                    )
                else:
                    existing_social_simulations[file.stem] = social_simulation
            else:
                # Handle SocializedContext type
                socialized_context_dict = simulation_dict["socialized_context"]
                try:
                    socialized_context = SocializedContext(**socialized_context_dict)
                except Exception:
                    socialized_context_dict = dictlize(socialized_context_dict)
                    socialized_context = SocializedContext(**socialized_context_dict)
                if identifier_key:
                    existing_socialized_contexts[simulation_dict[identifier_key]] = (
                        socialized_context
                    )
                else:
                    existing_socialized_contexts[file.stem] = socialized_context
    return existing_socialized_contexts, existing_social_simulations


def dictlize(d: SocializedContextForModel | dict[str, Any]) -> dict[str, Any]:
def dictlize(d: SocializedContextForModel | dict[str, Any]) -> dict[str, Any]:
    """Convert a list of observations/actions into a dictionary format.

    Args:
        d: Input data that may contain lists of "key: value" strings

    Returns:
        Transformed dictionary with nested dictionaries instead of lists
    """
    if isinstance(d, SocializedContextForModel):
        socialized_events = d.model_dump()["socialized_context"]
        agents_names = d.agents_names
    else:
        socialized_events = d["socialized_context"]
        agents_names = d["agents_names"]

    if isinstance(d, SocializedContextForModel):
        socialized_events = d.model_dump()["socialized_context"]
        agents_names = d.agents_names
    else:
        socialized_events = d["socialized_context"]
        agents_names = d["agents_names"]

    for event in socialized_events:
        for key, value in event.items():
            if key in ["observations", "actions"] and isinstance(value, list):
                event[key] = {}
                for item in value:
                    if isinstance(item, str) and ":" in item:
                        k, v = item.split(":", 1)
                        event[key][k.strip()] = v.strip()
    return {"agents_names": agents_names, "socialized_context": socialized_events}
    return {"agents_names": agents_names, "socialized_context": socialized_events}
