"""
Decode special tags in socialized contexts for evaluation.

This module provides functions to resolve special tags like <same_as_state />
and <same_as_last_action /> to their actual content, making it easier to
compare socialized contexts using BLEU scores.
"""

from typing import Any, Dict, Optional
import re


def decode_observation(
    observation: str,
    current_state: str,
    last_action: Optional[str] = None,
    last_action_agent: Optional[str] = None,
) -> str:
    """
    Decode an observation string by replacing special tags with actual content.

    Args:
        observation: The observation string that may contain special tags
        current_state: The current state of the world at this timestep
        last_action: The last action taken (for <same_as_last_action /> tags)
        last_action_agent: The agent who took the last action

    Returns:
        Decoded observation with tags replaced by actual content
    """
    if not observation or observation == "none":
        return observation

    decoded = observation

    # Replace <same_as_state /> with actual state
    if "<same_as_state />" in decoded:
        decoded = decoded.replace("<same_as_state />", current_state)

    # Replace <same_as_last_action /> with actual last action
    if "<same_as_last_action />" in decoded and last_action:
        decoded = decoded.replace("<same_as_last_action />", last_action)

    # Handle <same_as_last_action_x /> where x is agent index
    # For simplicity, we'll replace with last_action if it exists
    pattern = r"<same_as_last_action_\d+ />"
    if re.search(pattern, decoded) and last_action:
        decoded = re.sub(pattern, last_action, decoded)

    # Keep <mental_state>...</mental_state> tags as-is (they contain actual content)
    # No replacement needed

    return decoded.strip()


def decode_timestep(
    timestep: Dict[str, Any],
    last_action: Optional[str] = None,
    last_action_agent: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Decode a single timestep by resolving all special tags.

    Args:
        timestep: Dictionary containing timestep, state, observations, actions
        last_action: The last action from previous timestep
        last_action_agent: The agent who took the last action

    Returns:
        Decoded timestep dictionary with tags resolved
    """
    decoded = timestep.copy()
    current_state = timestep.get("state", "")

    # Decode state (handle <same_as_last_action /> if present)
    if current_state and "<same_as_last_action />" in current_state and last_action:
        decoded["state"] = current_state.replace("<same_as_last_action />", last_action)

    # Decode observations
    decoded_observations = {}
    for agent, obs in timestep.get("observations", {}).items():
        decoded_observations[agent] = decode_observation(
            obs, decoded["state"], last_action, last_action_agent
        )
    decoded["observations"] = decoded_observations

    # Update last_action for next timestep
    # Find the first non-none action
    new_last_action = last_action
    new_last_action_agent = last_action_agent
    for agent, action in timestep.get("actions", {}).items():
        if action and action != "none":
            # Format action with agent name if not already present
            if not action.startswith(agent):
                new_last_action = f"{agent} {action}"
            else:
                new_last_action = action
            new_last_action_agent = agent
            break

    decoded["_last_action"] = new_last_action
    decoded["_last_action_agent"] = new_last_action_agent

    return decoded


def decode_socialized_context_dict(context_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decode an entire socialized context dictionary by resolving all special tags.

    Args:
        context_dict: Dictionary containing socialized context with special tags

    Returns:
        Decoded context dictionary with all tags resolved
    """
    decoded = context_dict.copy()
    socialized_context = context_dict.get("socialized_context", [])

    decoded_steps = []
    last_action = None
    last_action_agent = None

    for step in socialized_context:
        decoded_step = decode_timestep(step, last_action, last_action_agent)
        # Remove internal tracking fields
        last_action = decoded_step.pop("_last_action", None)
        last_action_agent = decoded_step.pop("_last_action_agent", None)
        decoded_steps.append(decoded_step)

    decoded["socialized_context"] = decoded_steps
    return decoded
