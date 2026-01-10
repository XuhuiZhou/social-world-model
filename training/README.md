# Together AI Fine-Tuning Pipeline for Social World Model

Fine-tune Meta-Llama-3.1-8B-Instruct to generate socialized contexts for Theory of Mind tasks using Together AI's API with Weights & Biases monitoring.

## Overview

This pipeline automates the complete fine-tuning workflow:
1. **Data Preparation**: Load 3,533 pre-generated socialized context JSONs and convert to ChatML format
2. **File Upload**: Upload training and validation JSONL files to Together AI
3. **Job Submission**: Create and configure fine-tuning job with LoRA
4. **Monitoring**: Real-time tracking of training metrics via Weights & Biases

## Setup

### 1. Install Dependencies

```bash
# Install required packages
uv pip install together wandb pyyaml typer rich
```

### 2. Configure API Keys

You'll need API keys from Together AI and Weights & Biases.

#### Together AI
1. Sign up at [together.ai](https://together.ai)
2. Get your API key from the dashboard
3. Set the environment variable:

```bash
export TOGETHER_API_KEY="your-together-api-key"
```

#### Weights & Biases
1. Sign up at [wandb.ai](https://wandb.ai)
2. Get your API key from settings
3. Login:

```bash
export WANDB_API_KEY="your-wandb-api-key"
wandb login
```

Or add both to your `.env` file:

```bash
echo "TOGETHER_API_KEY=your-together-key" >> .env
echo "WANDB_API_KEY=your-wandb-key" >> .env
```

## Usage

### Basic Fine-Tuning

Run with default configuration:

```bash
uv run python training/finetune_socialized_context.py finetune
```

This will:
- Load 3,533 socialized contexts from `data/tomi_results/socialized_context_o3-2025-04-16_rephrased_tomi_train_6000.csv_o3-2025-04-16/`
- Convert to ChatML format
- Split 90/10 into train (3,179) and validation (354)
- Upload to Together AI
- Start fine-tuning Meta-Llama-3.1-8B-Instruct for 3 epochs
- Monitor progress and log metrics to W&B

### Custom Configuration

Use a custom config file:

```bash
uv run python training/finetune_socialized_context.py finetune \
  --config-path=training/custom_config.yaml
```

Or override specific parameters:

```bash
uv run python training/finetune_socialized_context.py finetune \
  --data-dir=path/to/your/json/files \
  --output-dir=training/custom_output
```

### Configuration Options

Edit `training/finetune_config.yaml` to customize:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_model` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | Base model to fine-tune |
| `n_epochs` | `3` | Number of training epochs |
| `learning_rate` | `1e-5` | Learning rate |
| `batch_size` | `"max"` | Batch size (auto-optimized) |
| `lora` | `true` | Use LoRA for efficient fine-tuning |
| `lora_rank` | `64` | LoRA rank (higher = more parameters) |
| `lora_alpha` | `16` | LoRA alpha (scaling factor) |
| `val_ratio` | `0.1` | Validation split ratio |
| `wandb_project` | `social-world-model-finetuning` | W&B project name |

## Expected Output

### 1. Processed Data Files

After data preparation:
- `training/output/train.jsonl` - 3,179 training samples in ChatML format
- `training/output/val.jsonl` - 354 validation samples in ChatML format

### 2. Together AI Fine-Tuning Job

You'll see output like:

```
Together AI Fine-Tuning Pipeline
Fine-tuning Meta-Llama-3.1-8B-Instruct for socialized context generation

Step 1: Preparing data
Found 3533 JSON files in data/tomi_results/...
Successfully loaded 3533 valid records
Converting to ChatML format...
Split: 3179 train, 354 validation
✓ Data preparation complete

Step 2: Uploading files to Together AI
✓ Train file uploaded: file-abc123
✓ Validation file uploaded: file-def456

Step 3: Creating fine-tuning job
✓ W&B initialized: https://wandb.ai/...
✓ Fine-tuning job created: ft-job-xyz789

Step 4: Monitoring fine-tuning job
Job ID: ft-job-xyz789
Status: running...
```

### 3. Weights & Biases Dashboard

View real-time training progress at:
```
https://wandb.ai/your-username/social-world-model-finetuning
```

The dashboard shows:
- Training and validation loss curves
- Learning rate schedule
- Hyperparameters
- Training duration
- Model checkpoints

### 4. Fine-Tuned Model

When complete, you'll receive a model identifier:

```
Fine-tuning Complete!
Model name: your-username/Meta-Llama-3.1-8B-Instruct-socialized-context-generator-abc123

To use in your code:
  model_name="together_ai/your-username/Meta-Llama-3.1-8B-Instruct-socialized-context-generator-abc123"
```

## Using the Fine-Tuned Model

### With agenerate

```python
from social_world_model.generation_utils import agenerate

story = "Mia entered the lounge. Chloe arrived at the lounge..."

result = await agenerate(
    model_name="together_ai/your-username/Meta-Llama-3.1-8B-Instruct-socialized-context-generator-abc123",
    template="Generate a detailed socialized context for the following story:\n\n{story}",
    input_values={"story": story},
    temperature=0.7
)

print(result)  # Socialized context JSON
```

### With Together SDK directly

```python
from together import Together

client = Together()

response = client.chat.completions.create(
    model="your-username/Meta-Llama-3.1-8B-Instruct-socialized-context-generator-abc123",
    messages=[
        {"role": "system", "content": "You are dissecting the TOMI scenarios..."},
        {"role": "user", "content": f"Generate socialized context for: {story}"}
    ],
    temperature=0.7
)

print(response.choices[0].message.content)
```

## Timeline & Costs

| Phase | Duration | Cost |
|-------|----------|------|
| Data Preparation | 5-10 minutes | Free |
| File Upload | 2-5 minutes | Free |
| Fine-Tuning (3 epochs) | 2-4 hours | $3-15 |
| **Total** | **~3-5 hours** | **~$3-15** |

Inference cost after fine-tuning: ~$0.20 per 1M tokens

## Troubleshooting

### Issue: `TOGETHER_API_KEY environment variable not set`

**Solution**: Set your Together AI API key:

```bash
export TOGETHER_API_KEY="your-api-key"
```

Or add to `.env` file.

### Issue: File upload fails

**Possible causes:**
- Check that `TOGETHER_API_KEY` is correct
- Verify file size is under 1GB
- Check network connectivity

**Solution**: The script will retry failed uploads 3 times with exponential backoff.

### Issue: Job remains in "queued" status for a long time

**Explanation**: Together AI may have high demand. Typical wait time is 5-30 minutes.

**Solution**: Be patient. The script will continue monitoring. You can check job status at:
```
https://api.together.xyz/playground/fine-tuning
```

### Issue: W&B not logging metrics

**Possible causes:**
- `WANDB_API_KEY` not set
- W&B login failed
- Together AI's W&B integration issue

**Solution**:
1. Verify `WANDB_API_KEY` is set correctly
2. Run `wandb login` to authenticate
3. Check that `wandb.init()` succeeds in the logs
4. If Together AI integration fails, metrics may not be available (fine-tuning will still work)

### Issue: Out of memory during fine-tuning

**Solution**: This is handled automatically by Together AI's infrastructure. If issues persist:
- Reduce `lora_rank` from 64 to 32
- Reduce `n_epochs` to 2

### Issue: Poor model performance after fine-tuning

**Possible causes:**
- Insufficient training data
- Learning rate too high/low
- Too few epochs

**Solution**:
1. Increase training data (use more than 3,533 samples)
2. Adjust learning rate in config (try `5e-6` or `2e-5`)
3. Train for more epochs (try 5-10)
4. Increase `lora_rank` to 128 for more capacity

## File Structure

```
training/
├── data_utils.py                 # Data loading and conversion utilities
├── finetune_socialized_context.py # Main fine-tuning pipeline
├── finetune_config.yaml          # Hyperparameter configuration
├── README.md                     # This file
└── output/                       # Generated files (created automatically)
    ├── train.jsonl              # Training data in ChatML format
    └── val.jsonl                # Validation data in ChatML format
```

## Advanced Usage

### Resume Monitoring an Existing Job

If your connection drops during monitoring, you can resume by modifying the script to directly call `monitor_job()` with your job ID:

```python
from training.finetune_socialized_context import TogetherAIFineTuner, FinetuneConfig

config = FinetuneConfig()
finetuner = TogetherAIFineTuner(config)
model_name = finetuner.monitor_job("ft-job-your-id")
```

### Hyperparameter Sweep with W&B

Create a sweep configuration:

```yaml
# sweep_config.yaml
program: training/finetune_socialized_context.py
method: bayes
metric:
  name: val/loss
  goal: minimize
parameters:
  learning_rate:
    values: [1e-6, 5e-6, 1e-5, 5e-5]
  n_epochs:
    values: [2, 3, 5]
  lora_rank:
    values: [32, 64, 128]
```

Then run:

```bash
wandb sweep sweep_config.yaml
wandb agent your-sweep-id
```

## Data Format

### Input (Socialized Context JSON)

Each JSON file contains:

```json
{
  "story": "['Sentence 1.', 'Sentence 2.', ...]",
  "question": "Where does X think that Y searches for Z?",
  "answer": "location",
  "socialized_context": {
    "agents_names": ["Agent1", "Agent2"],
    "socialized_context": [
      {
        "timestep": "0",
        "state": "Description of world state",
        "observations": {
          "Agent1": "What Agent1 observes",
          "Agent2": "What Agent2 observes"
        },
        "actions": {
          "Agent1": "What Agent1 does",
          "Agent2": "What Agent2 does"
        }
      }
    ]
  }
}
```

### Output (ChatML Format)

Converted to:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are dissecting the TOMI scenarios. The assumptions are..."
    },
    {
      "role": "user",
      "content": "Generate a detailed socialized context for the following story:\n\nSentence 1. Sentence 2. ..."
    },
    {
      "role": "assistant",
      "content": "{\"agents_names\": [...], \"socialized_context\": [...]}"
    }
  ]
}
```

## References

- [Together AI Fine-Tuning Documentation](https://docs.together.ai/docs/fine-tuning)
- [Together AI Python SDK](https://github.com/togethercomputer/together-python)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

## Support

For issues or questions:
- Together AI: support@together.ai
- W&B: support@wandb.com
- Project issues: Create an issue in the repository
