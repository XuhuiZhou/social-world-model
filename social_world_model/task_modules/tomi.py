import pandas as pd
from pathlib import Path
from social_world_model.tom_engine import ToMEngine
from social_world_model.database import SocializedContext
import json
from typing import Any, Optional
from .utils import dictlize

async def tomi_simulation(row: pd.Series[Any], engine: Optional[ToMEngine] = None) -> dict[str, Any]:
    """Run experiment in simulation mode (using ToM engine for memory tracking)."""
    assert engine is not None, "Engine must be provided"
    socialized_context = engine.existing_socialized_contexts[str(row['index'])]
    socialized_context_dict = dictlize(socialized_context)
    # Extract char1 and char2 if they don't exist
    if not row['char1'] or str(row['char1']) == "nan":
        question = row['question'].lower()
        if "where will" in question:
            # Type: "Where will Ella look for the celery?"
            row['char1'] = question.split("where will")[1].split("look")[0].strip()
            row['char2'] = ""
            row['char1'] = row['char1'].capitalize()
        elif "where does" in question and "think that" in question:
            # Type: "Where does Owen think that Logan searches for the grapes?"
            row['char1'] = question.split("where does")[1].split("think")[0].strip()
            row['char2'] = question.split("that")[1].split("searches")[0].strip()
            row['char1'] = row['char1'].capitalize()
            row['char2'] = row['char2'].capitalize()
        else:
            # Type: "Where was the xx at the beginning?"
            row['char1'] = ""
            row['char2'] = ""
    engine.set_agent_prompt(
        "You will be asking some questions about your beliefs. The previous history of the interaction below is your memory (i.e., you perceive the entire history of the interaction). Assume that you can perceive every scene in your location but not scenes occurring elsewhere. If something is being moved, that means it is not in its original location anymore.\
You need to first reason about the question (majorly focusing where the object has been moved to, and answer the most detailed position possible e.g., the object is in A and A is in B, then you should answer 'A') and then respond to the question with the following format:<reasoning>(reasoning)</reasoning> <answer>(answer; the answer should be just the position and nothing else)</answer>")
    agent_names = socialized_context_dict['agents_names']
    socialized_events = socialized_context_dict['socialized_context']
    if row['char2'] and str(row['char2'])!="nan":
        imagined_socialized_events = []
        for index, event in enumerate(socialized_events):
            if event['observations'][row['char1']] == "none":
                if index > 0:
                    socialized_events[index-1]['actions'][row['char2']] = "none"
            else:
                imagined_socialized_events.append(event)
        socialized_context_dict['socialized_context'] = imagined_socialized_events
    await engine.initialize_simulation_from_socialized_context(socialized_context_dict)

    if row['char2'] and str(row['char2'])!="nan":
        object = row["question"].split("searches for")[1].split("?")[0]
        restructure_question = f"Where will you look for the {object}?"
        reasoning, answer = await engine.reason_about_belief(restructure_question, agent_names, target_agent=row['char2'])
    elif row['char1'] and str(row['char1'])!="nan":
        restructure_question = row['question'].replace(row['char1'], "you")
        reasoning, answer = await engine.reason_about_belief(restructure_question, agent_names, target_agent=row['char1'])
    else:
        return {} 
    correct_answer = row['answer']
    assert isinstance(answer, str)
    is_correct = correct_answer in answer

    simulation = engine.get_simulation()
    simulation_dict = simulation.dict()
    result = {
        "question": row['question'],
        "reasoning": reasoning,
        "answer": answer,
        "correct_answer": correct_answer,
        "is_correct": is_correct,
        "socialized_context": socialized_context,
        "transformed_question": simulation_dict["question"],
        "memory": simulation_dict["agent_memories"],
        "agents": simulation_dict["agents"]
    }
    breakpoint()
    return result