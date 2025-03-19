from typing import Any
from social_world_model.database import SocializedContextForModel, SocializedContext
import json
from pathlib import Path

def load_existing_socialized_contexts(data_path: Path, identifier_key: str) -> dict[str, SocializedContext]:
    existing_socialized_contexts = {}
    for file in data_path.glob('*.json'):
        with open(file, 'r') as f:
            simulation_dict = json.load(f)
            socialized_context_dict = simulation_dict['socialized_context']
            socialized_context = SocializedContext(**socialized_context_dict)
            if identifier_key:
                existing_socialized_contexts[simulation_dict[identifier_key]] = socialized_context
            else:
                existing_socialized_contexts[file.stem] = socialized_context
    return existing_socialized_contexts

def dictlize(d: SocializedContextForModel) -> dict[str, Any]:
    """Convert a list of observations/actions into a dictionary format.
    
    Args:
        d: Input data that may contain lists of "key: value" strings
        
    Returns:
        Transformed dictionary with nested dictionaries instead of lists
    """
    socialized_events = d.model_dump()['socialized_context']
    for event in socialized_events:
        for key, value in event.items():
            if key in ['observations', 'actions'] and isinstance(value, list):
                event[key] = {}
                for item in value:
                    if isinstance(item, str) and ':' in item:
                        k, v = item.split(':', 1)
                        event[key][k.strip()] = v.strip()
    return {
        'agents_names': d.agents_names,
        'socialized_context': socialized_events
    }