from social_world_model.social_world_model import SocialWorldModel
from typing import Any, Optional
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from collections import defaultdict

NEGOTIATIONTOM_SOCIALIZED_CONTEXT_PROMPT = """You are dissecting a negotiation conversation for a camping trip and need to answer a question about it. There are two agents who own some basic supplies and negotiate with each other to split the additional food packages, water bottles, and firewood to make their camping trip even better. Each of these items will be of either High, Medium or Low priority for these two agents. Each of the additional items only has an available quantity of 3. The available intent choices are: Build-Rapport, Show-Empathy, Promote-Coordination, Callout-Fairness, Undermine-Requirements, Discover-Preference, Describe-Need, No-Need, No-Intention. You may select one or more of these options. The available desire and belief choices are: Not Given, Water, Food, Firewood. You may select one for each preference level."""

INTENT_CHOICES = {
    "A": "Build-Rapport",
    "B": "Show-Empathy",
    "C": "Promote-Coordination",
    "D": "Callout-Fairness",
    "E": "Undermine-Requirements",
    "F": "Discover-Preference",
    "G": "Describe-Need",
    "H": "No-Need",
    "I": "No-Intention",
}

DESIRE_CHOICES = {
    "A": "Not Given",
    "B": "Water",
    "C": "Food",
    "D": "Firewood",
}

BELIEF_CHOICES = {
    "A": "Not Given",
    "B": "Water",
    "C": "Food",
    "D": "Firewood",
}


def reformat_negotiationtom_data(
    data_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    # Step 1: Build a map of the full dialogue for each set_id
    full_dialogue_map = {}
    for entry in data_list:
        dialogue_id = entry["dialogue_id"]
        set_id = int(dialogue_id.split("-")[0])
        if set_id not in full_dialogue_map or len(entry["dialogue"]) > len(
            full_dialogue_map[set_id]
        ):
            full_dialogue_map[set_id] = entry["dialogue"]

    # Step 2: Generate the reformatted list
    result = []
    index = 0

    for entry in data_list:
        dialogue_id = entry["dialogue_id"]
        set_id = int(dialogue_id.split("-")[0])
        round_id = int(dialogue_id.split("-")[1])
        full_dialogue = full_dialogue_map[set_id]

        # Append each of the six reformatted dicts
        def append_record(record_type, agent, content, question):
            nonlocal index
            result.append(
                {
                    "index": index,
                    "set_id": set_id,
                    "round_id": round_id,
                    "dialogue_id": dialogue_id,
                    "dialogue_full": full_dialogue,
                    "dialogue": entry["dialogue"],
                    "agent1_desire": entry["agent1_desire"],
                    "agent2_desire": entry["agent2_desire"],
                    "question_type": record_type,
                    "agent_index": agent,
                    "question": question,
                    "correct_answer": content,
                }
            )
            index += 1

        question_intent = """What are the likely intentions of agent {agent_index} in their most recent utterance? Based on the dialogue history, select one or more intentions (i.e., "A", "B", "C", ..., "I") from the list of choices below. Separate the letters of the selected choices with commas.

A. Intents to build a rapport with the opponent
B. Intents to show empathy with the opponent
C. Intents to promote coordination with the opponent
D. Intents to callout to fairness
E. Intents to undermine the requirements of the opponent
F. Intents to discover the preference order of the opponent
G. Intents to describe a need for an item
H. Intents to point out they do not need an item
I. No clear intention in the utterance
"""
        question_desire = """According to the dialogue history, what are agent {agent_index}'s preferences for items at the high, medium, and low levels? For each level, choose one option from "A", "B", "C", or "D". List the letters of the selected choices in the order of high, medium, and low, separated by commas.

A. Not given
B. Water
C. Food
D. Firewood
"""
        question_belief = """Based on the dialogue, what does agent {agent_index} believe are agent {agent_other_index}'s item preferences at the high, medium, and low levels? For each level, choose one option from "A", "B", "C", or "D". List the letters of the selected choices in the order of high, medium, and low, separated by commas.

A. Not given
B. Water
C. Food
D. Firewood
"""

        # Determine which utterance is
        agent1_intent, agent2_intent = None, None
        if entry["utterance1_agent"] == "agent_1":
            agent1_intent = entry["utterance1_intent"].split(",")
        elif entry["utterance1_agent"] == "agent_2":
            agent2_intent = entry["utterance1_intent"].split(",")
        else:
            pass

        if entry["utterance2_agent"] == "agent_2":
            agent2_intent = entry["utterance2_intent"].split(",")
        elif entry["utterance2_agent"] == "agent_1":
            agent1_intent = entry["utterance2_intent"].split(",")
        else:
            pass

        if agent1_intent:
            append_record(
                "intent",
                "agent_1",
                agent1_intent,
                question_intent.format(agent_index="1"),
            )

            append_record(
                "desire",
                "agent_1",
                {
                    "high": entry["agent1_desire_high"],
                    "medium": entry["agent1_desire_medium"],
                    "low": entry["agent1_desire_low"],
                },
                question_desire.format(agent_index="1"),
            )

            append_record(
                "belief",
                "agent_1",
                {
                    "high": entry["agent1_belief_high"],
                    "medium": entry["agent1_belief_medium"],
                    "low": entry["agent1_belief_low"],
                },
                question_belief.format(agent_index="1", agent_other_index="2"),
            )

        if agent2_intent:
            append_record(
                "intent",
                "agent_2",
                agent2_intent,
                question_intent.format(agent_index="2"),
            )

            append_record(
                "desire",
                "agent_2",
                {
                    "high": entry["agent2_desire_high"],
                    "medium": entry["agent2_desire_medium"],
                    "low": entry["agent2_desire_low"],
                },
                question_desire.format(agent_index="2"),
            )

            append_record(
                "belief",
                "agent_2",
                {
                    "high": entry["agent2_belief_high"],
                    "medium": entry["agent2_belief_medium"],
                    "low": entry["agent2_belief_low"],
                },
                question_belief.format(agent_index="2", agent_other_index="1"),
            )

    return result


def prepare_negotiationtom_vanilla(
    row: dict[str, Any], pure_context: bool = False
) -> tuple[str, dict[str, Any]]:
    story = "\n".join(row["dialogue"])
    extra_info = row.get("extra_info", "")
    if extra_info:
        if pure_context:
            story = extra_info
            extra_info = ""
        else:
            story = story

    question = row["question"]
    template = """You are analyzing a negotiation conversation for a camping trip and need to answer a question about it. There are two agents who own some basic supplies and negotiate with each other to split the additional food packages, water bottles, and firewood to make their camping trip even better. Each of these items will be of either High, Medium or Low priority for these two agents. Each of the additional items only has an available quantity of 3. First give step-by-step analysis about the question. Then output the answer. Provide your reasoning within the <reasoning></reasoning>tag. For the answer, use <answer>(put your answer here)</answer> and include only the letter corresponding to your choice but not other information.

Below is the story and question:
## Story
{story}
## Extra Information
(to help you better understand and answer the question)
{extra_info}
## Question
{question}"""

    input_values = {"story": story, "extra_info": extra_info, "question": question}

    return template, input_values


def evaluate_response(result: dict[str, Any]) -> dict[str, Any]:
    answer = result["answer"].strip().split(",")
    answer = [item.strip() for item in answer]
    if result["question_type"] == "intent":
        answer_list = []
        for item in answer:
            if item not in INTENT_CHOICES:
                result["is_correct"] = False
                break
            else:
                answer_list.append(INTENT_CHOICES[item])
        if len(answer_list) == result["correct_answer"] and set(answer_list) == set(
            result["correct_answer"]
        ):
            result["is_correct"] = True
            result["answer_list"] = answer_list
        else:
            result["is_correct"] = False
            result["answer_list"] = answer_list

    elif result["question_type"] == "desire":
        answer_list = []
        for item in answer:
            if item not in DESIRE_CHOICES:
                result["is_correct"] = False
                break
            else:
                answer_list.append(DESIRE_CHOICES[item])
        if (
            len(answer_list) == 3
            and answer_list[0] == result["correct_answer"]["high"]
            and answer_list[1] == result["correct_answer"]["medium"]
            and answer_list[2] == result["correct_answer"]["low"]
        ):
            result["is_correct"] = True
        else:
            result["is_correct"] = False

    elif result["question_type"] == "belief":
        answer_list = []
        for item in answer:
            if item not in BELIEF_CHOICES:
                result["is_correct"] = False
                break
            else:
                answer_list.append(BELIEF_CHOICES[item])
        if (
            len(answer_list) == 3
            and answer_list[0] == result["correct_answer"]["high"]
            and answer_list[1] == result["correct_answer"]["medium"]
            and answer_list[2] == result["correct_answer"]["low"]
        ):
            result["is_correct"] = True
        else:
            result["is_correct"] = False

    return result


def create_negotiationtom_result(
    parsed_result: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    targeted_entries = [
        "index",
        "set_id",
        "round_id",
        "dialogue_id",
        "dialogue_full",
        "dialogue",
        "question_type",
        "agent_index",
        "question",
        "answer",
        "correct_answer",
        "is_correct",
        "answer_list",
        "socialized_context",
        "extra_info",
    ]
    if not parsed_result:
        return {}

    result = {}
    for entry in targeted_entries:
        if entry in parsed_result:
            result[entry] = parsed_result[entry]
        elif entry in row:
            result[entry] = row[entry]
        else:
            continue
    result = evaluate_response(result)
    return result


def negotiationtom_evaluation_report(results: list[dict[str, Any]]) -> None:
    total_count_intent, correct_count_intent = 0, 0
    total_count_desire, correct_count_desire = 0, 0
    total_count_belief, correct_count_belief = 0, 0
    all_intent_pred, all_intent_true = [], []

    # For consistency
    intent_per_set = defaultdict(list)
    desire_per_set = defaultdict(list)
    belief_per_set = defaultdict(list)

    all_correct_by_set = defaultdict(list)

    for result in results:
        all_correct_by_set[result["set_id"]].append(result["is_correct"])

        set_id = result["set_id"]
        question_type = result["question_type"]

        if question_type == "intent":
            total_count_intent += 1
            # intent micro-F1 and macro-F1
            all_intent_pred.append(result["answer_list"])
            all_intent_true.append(result["correct_answer"])
            intent_per_set[set_id].append(result["is_correct"])
            if result["is_correct"]:
                correct_count_intent += 1
        elif question_type == "desire":
            total_count_desire += 1
            desire_per_set[set_id].append(result["is_correct"])
            if result["is_correct"]:
                correct_count_desire += 1
        elif question_type == "belief":
            total_count_belief += 1
            belief_per_set[set_id].append(result["is_correct"])
            if result["is_correct"]:
                correct_count_belief += 1

    mlb = MultiLabelBinarizer()
    all_intent_true_binary = mlb.fit_transform(all_intent_true)
    all_intent_pred_binary = mlb.transform(all_intent_pred)
    f1_score_micro = f1_score(
        all_intent_true_binary, all_intent_pred_binary, average="micro"
    )
    f1_score_macro = f1_score(
        all_intent_true_binary, all_intent_pred_binary, average="macro"
    )

    # Consistency computation
    consistent_intent = sum(all(corrects) for corrects in intent_per_set.values())
    consistent_desire = sum(all(corrects) for corrects in desire_per_set.values())
    consistent_belief = sum(all(corrects) for corrects in belief_per_set.values())

    total_sets_intent = len(intent_per_set)
    total_sets_desire = len(desire_per_set)
    total_sets_belief = len(belief_per_set)

    fully_correct_sets = sum(
        all(is_correct_list) for is_correct_list in all_correct_by_set.values()
    )
    total_sets = len(all_correct_by_set)

    print(
        f"Desire Exact Match: {correct_count_desire}/{total_count_desire} = {correct_count_desire/total_count_desire:.2%}"
    )
    print(
        f"Belief Exact Match: {correct_count_belief}/{total_count_belief} = {correct_count_belief/total_count_belief:.2%}"
    )
    print(f"Intent Micro-F1: {f1_score_micro:.2%}")
    print(f"Intent Macro-F1: {f1_score_macro:.2%}")
    print(
        f"All Exact Match: {fully_correct_sets}/{total_sets} = {fully_correct_sets/total_sets:.2%}"
    )
    print(
        f"Intent Consistency: {consistent_intent}/{total_sets_intent} = {consistent_intent/total_sets_intent:.2%}"
    )
    print(
        f"Desire Consistency: {consistent_desire}/{total_sets_desire} = {consistent_desire/total_sets_desire:.2%}"
    )
    print(
        f"Belief Consistency: {consistent_belief}/{total_sets_belief} = {consistent_belief/total_sets_belief:.2%}"
    )


async def negotiationtom_simulation(
    row: dict[str, Any], engine: Optional[SocialWorldModel] = None
) -> dict[str, Any]:
    pass
