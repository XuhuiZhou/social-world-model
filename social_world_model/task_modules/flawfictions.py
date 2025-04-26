from social_world_model.social_world_model import SocialWorldModel
from typing import Any, Optional


FLAWFICTIONS_SOCIALIZED_CONTEXT_PROMPT = """You are given a narrative folk-tale and must generate a sequence of socialized context steps that precisely capture character interactions, observations, and evolving world states.

If you detect anything weird such as a continuity error about an event or internal attitude that contradicts earlier established facts, you should flag it inside the relevant agent’s <mental_state> tag."""


def reformat_flawfictions_data(data_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result_data = []
    index = 0
    for item in data_list:
        item["index"] = index
        item["set_id"] = index
        result_data.append(item)
        index += 1

    return result_data


def prepare_flawfictions_vanilla(
    row: dict[str, Any], pure_context: bool = False
) -> tuple[str, dict[str, Any]]:
    row["extra_info"] = row.get("extra_info", "")
    if row["extra_info"]:
        if pure_context:
            row["story"] = row["extra_info"]
            row["extra_info"] = ""
        else:
            pass

    template = """You are tasked with detecting the presence of continuity errors in a short story. A continuity error occurs when an event or detail in the story contradicts or is incompatible with previously established information about the story's world or characters.

Please carefully read and analyze the story above. Your goal is to identify any continuity errors that may exist within the narrative.

Guidelines for identifying continuity errors:
1. Pay attention to character descriptions, settings, and plot events.
2. Look for inconsistencies in timelines, character abilities, or established rules of the story's world.
3. Note any contradictions between earlier and later parts of the story.

If you find any continuity errors, please provide a clear explanation of the error and why it contradicts earlier information in the story.

Identify and quote the specific lines that:
1. Introduce the continuity error
2. Contain the earlier information that is contradicted by the error

If you do not find any continuity errors, state that no errors were found and briefly explain why the story maintains consistency.

Based on your analysis, make a final decision on whether a continuity error exists in the story.

Some tips and tricks for the task:
- Pay attention to even little details in the story, the continuity errors often are not limited to the central plot point.
- You might observe some logical error in the story, but make sure that it qualifies as a continuity error i.e. you should be able to find sentences in the story which have the error and the sentences with the original fact that was contradicted (see definitions below for a concrete example).

First give step-by-step analysis about the question. Then output the answer. Provide your reasoning within the <reasoning></reasoning>tag. For the answer, use <answer>(put your answer here)</answer> and format your response as follows:

<answer>
<error_lines>
[If applicable, quote the lines that introduce the continuity error]
</error_lines>

<contradicted_lines>
[If applicable, quote the lines from earlier in the story that are contradicted by the error]
</contradicted_lines>

<decision>
[State your final decision on whether a continuity error exists in the story. State "No continuity error found" if you think there is no continuity error.]
</decision>
</answer>


Here is the story to analyze:

<story>
{story}
</story>

<extra_info>
(to help you better understand the memory and answer the question)
{extra_info}
</extra_info>

"""

    input_values = {
        "story": row["story"],
        "extra_info": row["extra_info"],
    }
    return template, input_values


def parse_output(response_content: str) -> dict[str, Any]:
    def clean_lines(lines):
        lines = lines.split("\n")
        lines = [
            line.strip().replace("-", "").replace("*", "").strip() for line in lines
        ]
        lines = [line for line in lines if line != ""]
        return lines

    if "<error_lines>" not in response_content:
        cont_error_lines = "No error lines found"
    elif "</error_lines>" not in response_content:
        cont_error_lines = response_content.split("<error_lines>")[1].strip()
    else:
        cont_error_lines = (
            response_content.split("<error_lines>")[1]
            .split("</error_lines>")[0]
            .strip()
        )

    if "<contradicted_lines>" not in response_content:
        contradicted_lines = "No contradicted lines found"
    elif "</contradicted_lines>" not in response_content:
        contradicted_lines = response_content.split("<contradicted_lines>")[1].strip()
    else:
        contradicted_lines = (
            response_content.split("<contradicted_lines>")[1]
            .split("</contradicted_lines>")[0]
            .strip()
        )

    if "<decision>" not in response_content:
        cont_error = False
        cont_error_expl_short = "No decision found"
    elif "</decision>" not in response_content:
        decision = response_content.split("<decision>")[1].strip()
        cont_error_expl_short = decision
        cont_error = "no continuity error found" not in decision.lower()
    else:
        decision = (
            response_content.split("<decision>")[1].split("</decision>")[0].strip()
        )
        cont_error_expl_short = decision
        cont_error = "no continuity error found" not in decision.lower()

    return {
        "cont_error": float(cont_error),
        "cont_error_expl_detailed": response_content,
        "cont_error_expl_short": cont_error_expl_short,
        "cont_error_lines": clean_lines(cont_error_lines),
        "contradicted_lines": clean_lines(contradicted_lines),
    }


def evaluate_response(result: dict[str, Any]) -> dict[str, Any]:
    answer_info = parse_output(result["answer"])
    if answer_info["cont_error"] == result["cont_error"]:
        result["is_correct"] = True
    else:
        result["is_correct"] = False
    return result


def create_flawfictions_result(
    parsed_result: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    """Create FlawFictions result dictionary."""
    targeted_entries = [
        "set_id",
        "index",
        "story",
        "reasoning",
        "answer",
        "cont_error",
        "is_correct",
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
    result = evaluate_response(result)
    return result


def flawfictions_evaluation_report(results: list[dict[str, Any]]) -> None:
    """Evaluate FlawFictions result."""
    correct_count = 0
    for result in results:
        if result["is_correct"]:
            correct_count += 1
    print(
        f"Current accuracy: {correct_count}/{len(results)} = {correct_count/len(results):.2%}"
    )


async def flawfictions_simulation(
    row: dict[str, Any], engine: Optional[SocialWorldModel] = None
) -> dict[str, Any]:
    pass
