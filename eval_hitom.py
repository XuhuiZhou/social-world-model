import json
import os
import re
from tqdm import tqdm

# output_dir = "/home/jiaruil5/social_reasoning_rl/social-world-model/data/hitom_results/vanilla_o1-2024-12-17_Hi-ToM_data.json"
output_dir = "/data/jiarui_liu/social_reasoning_rl/social-world-model/data/hitom_results/socialized_context_o1-2024-12-17_processed_hitom_data.csv"

def evaluate_response(result):
    # dict(re.findall(r'([A-Z])\. ([^,]+)', choice_str))
    choices = result["choices"]
    answer = result["answer"].strip().capitalize()
    
    answer_dict = dict(re.findall(r'([A-Z])\. ([^,]+)', choices))
    answer_list = list(answer_dict.keys())
    
    if answer not in answer_list:
        result["is_correct"] = False
    else:
        if answer_dict[answer] == result["correct_answer"]:
            result["is_correct"] = True
        else:
            result["is_correct"] = False
    
    return result


correct_count = 0
total_count = 0
for file in os.listdir(output_dir):
    with open(os.path.join(output_dir, file), "r") as f:
        data = json.load(f)
    data = evaluate_response(data)
    with open(os.path.join(output_dir, file), "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()

    if data["is_correct"]:
        correct_count += 1
    total_count += 1

print(f"Accuracy: {correct_count}/{total_count} = {correct_count/total_count:.2%}")