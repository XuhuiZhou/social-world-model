from typing import Any, List, Optional
from social_world_model.database import (
    SocializedContextForModel,
    SocializedContext,
    SocialSimulation,
    SocializedStructureForModel,
)
import json
from pathlib import Path
from difflib import SequenceMatcher


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

    for event in socialized_events:
        for key, value in event.items():
            if key in ["observations", "actions"] and isinstance(value, list):
                event[key] = {}
                for item in value:
                    if isinstance(item, str) and ":" in item:
                        k, v = item.split(":", 1)
                        event[key][k.strip()] = v.strip()
    return {"agents_names": agents_names, "socialized_context": socialized_events}


def dictlize_socialized_structure(
    d: SocializedStructureForModel | dict[str, Any],
) -> dict[str, Any]:
    """Convert a SocializedStructureForModel or a dictionary into a dictionary format.

    Args:
        d: Input data that may be a SocializedStructureForModel or a dictionary

    Returns:
        Transformed dictionary with observations and actions in dictionary format
    """
    if isinstance(d, SocializedStructureForModel):
        event = d.model_dump()
    else:
        event = d.copy()

    for key, value in event.items():
        if key in ["observations", "actions"] and isinstance(value, list):
            event[key] = {}
            for item in value:
                if isinstance(item, str) and ":" in item:
                    k, v = item.split(":", 1)
                    event[key][k.strip()] = v.strip()
                else:
                    event[key]["X"] = item
    return event


def find_best_match(
    name: str, candidates: List[str], threshold: float = 0.7
) -> Optional[str]:
    """Find the best matching name from a list of candidates using string similarity.

    Args:
        name: The name to match
        candidates: List of candidate names to match against
        threshold: Minimum similarity score to consider a match (default: 0.7)

    Returns:
        The best matching name if found, None otherwise
    """
    best_match = None
    best_score = 0.0
    for candidate in candidates:
        score = SequenceMatcher(None, name.lower(), candidate.lower()).ratio()
        if score > best_score and score >= threshold:
            best_score = score
            best_match = candidate
    return best_match


def standardize_agent_names(
    data: dict[str, Any],
    agents_names: List[str],
    fields: List[str] = ["observations", "actions"],
) -> tuple[dict[str, Any], List[str]]:
    """Standardize agent names in a dictionary using fuzzy matching.

    Args:
        data: Dictionary containing agent names to standardize
        agents_names: List of canonical agent names
        fields: List of fields in data to process (default: ["observations", "actions"])

    Returns:
        Tuple of (standardized data, updated list of agent names)
    """
    for field in fields:
        if field in data:
            for key in list(data[field].keys()):
                if key not in agents_names:
                    best_match = find_best_match(key, agents_names)
                    if best_match:
                        data[field][best_match] = data[field].pop(key)
                    else:
                        agents_names.append(key)
        for agent_name in agents_names:
            if agent_name not in data[field]:
                data[field][agent_name] = "none"
    return data, agents_names
