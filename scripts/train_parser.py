# Using Python
from together import Together # type: ignore
import os
import json

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")  # Optional, for logging fine-tuning to wandb

# Check the file format
from together.utils import check_file # type: ignore

client = Together(api_key=TOGETHER_API_KEY)

sft_report = check_file("./data/flawedfictions/flawedfictions_data/flawed_fictions_train.jsonl")
print(json.dumps(sft_report, indent=2))

assert sft_report["is_check_passed"] == True

# Upload the data to Together
train_file_resp = client.files.upload("./data/flawedfictions/flawedfictions_data/flawed_fictions_train.jsonl", check=True)
print(train_file_resp.id)  # Save this ID for starting your fine-tuning job

# Start the fine-tuning job
# Using Python - This fine-tuning job should take ~10-15 minutes to complete
ft_resp = client.fine_tuning.create(
    training_file = train_file_resp.id,
    model = 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
    train_on_inputs = "auto",
    n_epochs = 3,
    n_checkpoints = 1,
    wandb_api_key = WANDB_API_KEY,  # Optional, for visualization
    lora = True,   # Default True
    warmup_ratio = 0,
    learning_rate = 1e-5,
    suffix = 'flawedfictions_train',
)

print(ft_resp.id)  # Save this job ID for monitoring