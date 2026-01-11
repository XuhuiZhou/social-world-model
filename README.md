# social-world-model
use social simulation as a world model

## Setup

### 1. Install uv
```bash
pip install uv
```

### 2. Install Dependencies
```bash
# For basic API-based inference
uv sync --all-extras

# For offline inference with vLLM and training with wandb
uv sync --extra training
```

### 3. Configure Environment Variables
Create a `.env` file in the project root with your API keys:

```bash
# OpenAI API Key (required for OpenAI models like GPT-4)
OPENAI_API_KEY=your-openai-api-key-here

# Together AI API Key (required for Together AI models like DeepSeek-R1)
TOGETHER_API_KEY=your-together-api-key-here

# Weights & Biases API Key (optional, for training tracking)
WANDB_API_KEY=your-wandb-api-key-here

# Storage backend: "redis" or "local"
SOTOPIA_STORAGE_BACKEND=local
```

**Note:** For offline vLLM inference, you don't need API keys - vLLM runs models locally on GPU.

### 4. Create Data Folder
```bash
mkdir -p data
```

## Run ToM Benchmarks

All commands should be run with `--env-file .env` to load your API keys:

```bash
# Get help on available options
uv run --env-file .env python run_benchmarks.py --help
```

### Example: Run with API-based models

```bash
# Using Together AI with DeepSeek-R1
uv run --env-file .env python run_benchmarks.py tomi \
  --dataset-path=data/rephrased_tomi_test_600.csv \
  --batch-size=8 \
  --save \
  --model-name=together_ai/deepseek-ai/DeepSeek-R1 \
  --mode=vanilla
```

```bash
# Using OpenAI's o1 model with simulation mode
uv run --env-file .env python run_benchmarks.py tomi \
  --dataset-path=data/rephrased_tomi_test_600.csv \
  --batch-size=1 \
  --save \
  --model-name=o1-2024-12-17 \
  --mode=simulation
```

## Offline Inference with vLLM

The social-world-model now supports offline inference using vLLM for running models locally on GPU without API calls.

### Installation

```bash
# Install with training dependencies (includes vLLM and wandb)
uv sync --extra training
```

### Usage

Simply prefix your model name with `vllm/`:

```python
from social_world_model.generation_utils import agenerate, StrOutputParser

# Offline inference with local model
response = await agenerate(
    model_name="vllm/microsoft/Phi-4-mini-instruct",
    template="Question: {question}\nAnswer: ",
    input_values={"question": "What is Theory of Mind?"},
    output_parser=StrOutputParser(),
    temperature=0.7,
)
```

### Supported Models

- `vllm/microsoft/Phi-4-mini-instruct` (recommended, 14B parameters)
- Any HuggingFace model compatible with vLLM

### GPU Requirements

- NVIDIA GPU with CUDA support
- Minimum 16GB VRAM for Phi-4-mini-instruct
- Recommended: 24GB+ VRAM for optimal performance

### Running Benchmarks with vLLM

```bash
uv run --env-file .env python run_benchmarks.py tomi \
  --dataset-path=data/rephrased_tomi_test_600.csv \
  --batch-size=8 \
  --save \
  --model-name=vllm/microsoft/Phi-4-mini-instruct \
  --mode=vanilla
```

**Note:**
- Model loading takes 30-60 seconds on first use, but subsequent calls are fast (<1s) as the model is cached in GPU memory
- No API keys needed for offline vLLM inference!

## Fine-Tuning with VERL (Local Training)

The project supports local fine-tuning using [VERL](https://github.com/volcengine/verl) (Volcano Engine Reinforcement Learning) as an alternative to Together AI's cloud-based fine-tuning.

### Docker Setup (Recommended)

Docker provides a pre-configured environment with all dependencies (PyTorch, vLLM, flash-attn, CUDA) properly installed, avoiding version conflicts.

**System Requirements:**
- NVIDIA GPU with CUDA support
- Docker with nvidia-container-toolkit
- 3x RTX A6000 (48GB) or similar GPUs recommended

**One-Line Setup:**

```bash
sudo docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v $(pwd):/workspace/verl --name verl verlai/verl:vllm012.latest sleep infinity && sudo docker start verl && sudo docker exec -it verl bash
```

**Inside the container:**

```bash
# Install VERL
cd /workspace/verl/verl_repo
pip3 install --no-deps -e .

# Install social-world-model and dependencies
cd /workspace/verl
pip3 install -e .

# Install correct sotopia version (from GitHub)
pip3 install git+https://github.com/sotopia-lab/sotopia.git@main

# Run training (full 6000-example dataset)
python training/finetune_verl_sft.py \
  --data-dir="data/tomi_results/socialized_context_o3-2025-04-16_rephrased_tomi_train_6000.csv" \
  --output-dir="training/verl_output" \
  --n-gpus=3 \
  --total-steps=1000

# Or quick test (100-example dataset)
python training/finetune_verl_sft.py \
  --data-dir="data/tomi_results/socialized_context_o3-2025-04-16_rephrased_tomi_train_6000.csv_o3-2025-04-16" \
  --output-dir="training/verl_test" \
  --n-gpus=1 \
  --total-steps=50
```

**Or as a single command (full training):**

```bash
cd /workspace/verl/verl_repo && pip3 install --no-deps -e . && cd /workspace/verl && pip3 install -e . && pip3 install git+https://github.com/sotopia-lab/sotopia.git@main && python training/finetune_verl_sft.py --data-dir="data/tomi_results/socialized_context_o3-2025-04-16_rephrased_tomi_train_6000.csv_o3-2025-04-16" --output-dir="training/verl_output" --n-gpus=3 --total-steps=1000
```

**Single command (quick test):**

```bash
cd /workspace/verl/verl_repo && pip3 install --no-deps -e . && cd /workspace/verl && pip3 install -e . && pip3 install git+https://github.com/sotopia-lab/sotopia.git@main && python training/finetune_verl_sft.py --data-dir="data/tomi_results/socialized_context_groundtruth_rephrased_tomi_test_100.csv" --output-dir="training/verl_test" --n-gpus=1 --total-steps=50
```

**Container Management:**

```bash
# Stop container
sudo docker stop verl

# Restart container
sudo docker start verl

# Re-enter container
sudo docker exec -it verl bash

# Remove container (keeps image)
sudo docker rm verl
```

### Alternative: Manual Installation (Advanced)

If Docker is not available, you can attempt manual installation:

```bash
# Install VERL and dependencies
uv sync --extra verl_training

# Install flash-attn separately (requires torch 2.4.x, GLIBC 2.31+)
uv pip install --python .venv/bin/python flash-attn==2.7.4.post1 --no-build-isolation

# Run training
bash training/verl_test/run_sft.sh
```

**Warning:** Manual installation has strict version constraints:
- PyTorch 2.4.x (incompatible with vLLM 0.7+)
- vLLM 0.6.0-0.6.4 only
- flash-attn 2.7.4 may fail on systems with GLIBC < 2.31
- Dependency conflicts are common and difficult to resolve

Docker is strongly recommended to avoid these issues.

### Using Fine-Tuned Models

Load with Transformers:

```python
from transformers import AutoModelForCausalLM

model_path = "training/verl_output/checkpoints/step_1000"
model = AutoModelForCausalLM.from_pretrained(model_path)
```

Or use with vLLM for faster inference:
```python
model_name = "vllm/training/verl_output/checkpoints/step_1000"
```

## Evaluating Socialized Contexts

### Evaluate Benchmark Results (Task Accuracy)

To evaluate the accuracy of model answers against ground truth:

```bash
# Continue from previous run and evaluate existing results
uv run --env-file .env python run_benchmarks.py "tomi" \
  --dataset-path="data/rephrased_tomi_test_600.csv" \
  --batch-size=1 \
  --save \
  --model-name="o1-2024-12-17" \
  --mode="simulation" \
  --continue-mode="continue"
```

### Evaluate Socialized Context Quality (with Reference)

To evaluate the quality of generated socialized contexts against ground truth references:

```bash
# Evaluate ToMi socialized contexts
uv run --env-file .env python run_socialized_context_eval.py \
  --gt-dir data/tomi_results/socialized_context_groundtruth_rephrased_tomi_test_100.csv \
  --gen-dir data/tomi_results/socialized_context_gpt-4o-2024-08-06_rephrased_tomi_test_100.csv_vllm/microsoft/Phi-4-mini-instruct \
  --judge-model gpt-5-2025-08-07 \
  --batch-size 50 \
  --output tomi_eval_results.md

# Evaluate FANToM socialized contexts
uv run --env-file .env python run_socialized_context_eval.py \
  --gt-dir data/fantom_results/fixed_socialized_contexts \
  --gen-dir data/fantom_results/socialized_context_[model]_[dataset]_[context_model] \
  --judge-model gpt-5-2025-08-07 \
  --batch-size 50 \
  --output fantom_eval_results.md

# Evaluate HiToM socialized contexts
uv run --env-file .env python run_socialized_context_eval.py \
  --gt-dir data/hitom_results/fixed_socialized_contexts \
  --gen-dir data/hitom_results/socialized_context_[model]_[dataset]_[context_model] \
  --judge-model gpt-5-2025-08-07 \
  --batch-size 50 \
  --output hitom_eval_results.md
```

**Evaluation Metrics:**
- **Structural Score** (30%): Schema compliance, agent consistency, timestep format
- **Observation Accuracy** (70%): LLM judge assesses if agents observe correct events based on their location
- **Overall Score**: Weighted composite of structural + observation accuracy

**Output Format:**
```
| File         | Structural | Observation Accuracy | Overall |
|--------------|------------|----------------------|---------|
| 1198         | 0.900      | 0.850                | 0.865   |
| 1321         | 0.850      | 0.780                | 0.799   |
| **Mean**     | 0.875      | 0.815                | 0.832   |
```


## Contributing

To contribute, please use the following pattern for a new feature:
`feature_name/xx`

for a bug fix:
`bugfix/xx`

Run:
```bash
uv run pre-commit run --all-files
```
to check the code quality before pushing.

Run:
```bash
uv run mypy --strict .
```
to check the type safety of the code before pushing.

## Programming Social Contexts

The socialized context format uses several special tags to represent repeated information and mental states. Understanding these tags is essential when working with the data.

| Tag | Description |
|-----|-------------|
| `<same_as_state />` | Indicates that the observation is identical to the current state. Used to avoid redundancy in the context representation. |
| `<same_as_last_action />` | Indicates that the current state is identical to the last action taken. Used to maintain continuity between timesteps. |
| `<mental_state>...</mental_state>` | Encapsulates an agent's internal thoughts, beliefs, or emotions that isn't directly observable by other agents. |
| `none` | Indicates that there is no observation or action for a particular agent at a given timestep. |

These tags help maintain a compact representation of the socialized context while preserving all necessary information for understanding agent interactions and mental states.

## SAP Prompt

```
Follow these format instructions:
{"$defs": {"SocializedStructureForModel": {"properties": {"timestep": {"description": "The timestep of the current socialized structure, it could be a integer number or
a description of the time of the state.", "title": "Timestep", "type": "string"}, "state": {"description": "The current state of the world (including all the agents) at
this timestep. Important note: this is the state before the action is taken (e.g., the initial state could be 'none' at the beginning if there are no prior contexts
before the interaction starts).", "title": "State", "type": "string"}, "observations": {"description": "The observations for each agent in the social world at this
timestep (similar to the definition in partial observable Markov Decision Process, observation is derived from the obervation function with the current state as the
argument). Note that the different agents may have different observations. The observation would go into corresponding agent's memory, so make sure the observation is
clear for the agent to understand (first person perspective narrative is preferred). If it is the same as the current state, use the special tag '<same_as_state />' to
indicate the observation. For the internal thoughts, beliefs, or emotions of the agent that is not directly observable by other agents, use the special tag
'<mental_state>...</mental_state>' to indicate the internal observation. Put 'none' if the agent does not observe anything at this timestep. Important note: this is the
observation before the action is taken (e.g., the observation could be 'none' at the beginning if there are no prior contexts before the interaction starts). The format
for each entry in the list is: 'agent_name: observation'", "items": {"type": "string"}, "title": "Observations", "type": "array"}, "actions": {"description": "The
actions for each agent in the social world at this timestep. The length of the list should be the same as the number of agents. Put 'none' if the agent does not take
any action at this timestep. The format for each entry in the list is: 'agent_name: action'", "items": {"type": "string"}, "title": "Actions", "type": "array"}},
"required": ["timestep", "state", "observations", "actions"], "title": "SocializedStructureForModel", "type": "object"}}, "properties": {"agents_names": {"description":
"The names of the agents", "items": {"type": "string"}, "title": "Agents Names", "type": "array"}, "socialized_context": {"description": "A list of
SocializedStructureForModel objects, each representing a timestep of the social world. At the last timestep, all agents' actions should be 'none' as they have already
completed the interaction.", "items": {"$ref": "#/$defs/SocializedStructureForModel"}, "title": "Socialized Context", "type": "array"}}, "required": ["agents_names",
"socialized_context"], "title": "SocializedContextForModel", "type": "object"}
```
