import json
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="/data/jiarui_liu/social_reasoning_rl/social-world-model/data/flawedfictions_data/flawed_fictions100_seed0.json",
    )
    parser.add_argument(
        "--full_data_path",
        type=str,
        default="/data/jiarui_liu/social_reasoning_rl/social-world-model/data/flawedfictions_data/flawed_fictions.json",
    )
    parser.add_argument(
        "--full_output_path",
        type=str,
        default="/data/jiarui_liu/social_reasoning_rl/social-world-model/data/flawfictions_results/socialized_context_o3-2025-04-16_flawed_fictions.json_o3-2025-04-16/",
    )
    parser.add_argument(
        "--out_data_path",
        type=str,
        default="/data/jiarui_liu/social_reasoning_rl/social-world-model/data/flawedfictions_data/flawed_fictions_for_training.json",
    )
    parser.add_argument(
        "--out_train_data_path",
        type=str,
        default="/data/jiarui_liu/social_reasoning_rl/social-world-model/data/flawedfictions_data/flawed_fictions_train.json",
    )
    parser.add_argument(
        "--out_test_data_path",
        type=str,
        default="/data/jiarui_liu/social_reasoning_rl/social-world-model/data/flawedfictions_data/flawed_fictions_test.json",
    )
    parser.add_argument(
        "--example_analysis_path",
        type=str,
        default="/data/jiarui_liu/social_reasoning_rl/social-world-model/data/social_contexts_example/flawfictions.json",
    )
    args = parser.parse_args()

    test_data = json.load(open(args.test_data_path, "r"))
    indices = [item["index"] for item in test_data]

    full_data = json.load(open(args.full_data_path, "r"))
    full_output_path = args.full_output_path

    out_f = open(args.out_data_path, "w")

    res_data = []
    for file in os.listdir(full_output_path):
        data = json.load(open(os.path.join(full_output_path, file), "r"))
        if data["index"] in indices:
            data["split"] = "test"
        else:
            data["split"] = "train"

        supplement_data = full_data[data["index"]]
        for key, val in supplement_data.items():
            if key not in data:
                data[key] = val

        res_data.append(data)

    json.dump(res_data, out_f, ensure_ascii=False, indent=2)
    out_f.flush()

    import re
    from social_world_model.database import SocializedContextForModel

    def format_docstring(docstring: str) -> str:
        """Format a docstring for use in a prompt template."""
        return re.sub("\n +", "\n", docstring).strip()

    training_data = []
    test_data = []
    for item in res_data:
        context = item["story"]
        example_analysis = str(json.load(open(args.example_analysis_path, "r")))
        template = (
            "Please analyze the following narrative/context.\n\n"
            "#### Context: {context}\n\n"
            f"Use the following ground truth explanation about continuity errors as subtle guidance when generating mental states for the socialized context. You may incorporate relevant insights into characters' perspectives or reactions, but avoid stating the explanation directly.\n\n#### Ground truth explanation about continuity error (if any): {item['cont_error_expl']}\n\n"
        )

        input_values = {"context": context}
        task_specific_instructions = """You are given a narrative folk-tale and must generate a sequence of socialized context steps that precisely capture character interactions, observations, and evolving world states.

    If you detect anything weird such as a continuity error about an event or internal attitude that contradicts earlier established facts, you should flag it inside the relevant agent’s <mental_state> tag."""
        template += "#### Task specific instructions: {task_specific_instructions}\n\n"
        input_values["task_specific_instructions"] = task_specific_instructions
        template += "Example analysis: {example_analysis}\n\n"
        input_values["example_analysis"] = example_analysis
        template += "Follow these format instructions:\n{format_instructions}"
        input_values["format_instructions"] = (
            SocializedContextForModel.model_json_schema()
        )
        template = format_docstring(template)
        for key, value in input_values.items():
            template = template.replace(f"{{{key}}}", str(value))
        messages = [{"role": "user", "content": template}]

        socialized_context = str(item["socialized_context"])

        if item["split"] == "train":
            training_data.append({"input": messages, "output": socialized_context})
        else:
            test_data.append({"input": messages, "output": socialized_context})

    # full_data = json.load(open("flawed_fictions_for_training.json", 'r'))
    with open(args.out_train_data_path, "w") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
        f.flush()
    with open(args.out_test_data_path, "w") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
        f.flush()
