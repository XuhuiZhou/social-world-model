# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements a **social world model** for Theory of Mind (ToM) benchmarking. The system uses social simulation as a world model to evaluate LLMs on various ToM tasks by creating "socialized contexts" - structured representations of agent interactions including states, observations, and actions at each timestep.

## Environment Setup

```bash
# Install uv package manager first
pip install uv

# Install dependencies
uv sync --all-extras

# OR: Install with training dependencies (includes vLLM for offline inference and wandb)
uv sync --extra training

# Set required environment variable (for API-based models)
export OPENAI_API_KEY="your-key-here"
```

**Required:** A `data` folder must exist in the project root.

### Offline Inference with vLLM

The project supports offline model inference using vLLM for GPU-accelerated local models:

```bash
# Install with vLLM support
uv sync --extra training

# Use offline models by prefixing with "vllm/"
uv run python run_benchmarks.py "tomi" \
  --dataset-path="data/rephrased_tomi_test_600.csv" \
  --batch-size=8 \
  --save \
  --model-name="vllm/microsoft/Phi-4-mini-instruct" \
  --mode="vanilla"
```

**GPU Requirements for vLLM:**
- NVIDIA GPU with CUDA support
- Minimum 16GB VRAM for Phi-4-mini-instruct
- First model load: 30-60s (cached thereafter)

## Common Commands

### Running Benchmarks

```bash
# Get help on available options
uv run python run_benchmarks.py --help

# Run a ToM benchmark (example with ToMi dataset)
uv run python run_benchmarks.py "tomi" \
  --dataset-path="data/rephrased_tomi_test_600.csv" \
  --batch-size=8 \
  --save \
  --model-name="together_ai/deepseek-ai/DeepSeek-R1" \
  --mode="vanilla"

# Run with simulation mode
uv run python run_benchmarks.py "tomi" \
  --dataset-path="data/rephrased_tomi_test_600.csv" \
  --batch-size=1 \
  --save \
  --model-name="o1-2024-12-17" \
  --mode="simulation"

# Continue from previous run
uv run python run_benchmarks.py "tomi" \
  --dataset-path="data/rephrased_tomi_test_600.csv" \
  --batch-size=1 \
  --save \
  --model-name="o1-2024-12-17" \
  --mode="simulation" \
  --continue-mode="continue"

# Run dynamic social simulation (Sotopia-based)
uv run python run_dynamic.py \
  --models "gpt-4.1-2025-04-14" \
  --partner-model "gpt-4o-2024-08-06" \
  --agent-type "social_world_model" \
  --social-world-model-name "gpt-4.1-2025-04-14" \
  --experiment-tag "my_experiment" \
  --batch-size 100 \
  --push-to-db \
  --evaluator-model "o3-2025-04-16" \
  --task "hard"

# Run agent baselines
uv run python run_benchmarks_agent_baselines.py
```

### Code Quality Checks

```bash
# Run pre-commit checks (includes ruff linting/formatting, trailing whitespace, etc.)
uv run pre-commit run --all-files

# Type checking with mypy (strict mode)
uv run mypy --strict .
```

### Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_social_world_model.py

# Run with coverage
uv run pytest --cov=social_world_model
```

### UI Development

```bash
# Navigate to UI directory
cd ui

# Install dependencies (using bun)
bun install

# Run development server
bun dev

# Run on specific port with external access
bun dev -- -p 12000 -H 0.0.0.0

# Build for production
bun run build

# Start production server
bun start
```

### Fine-Tuning

```bash
# Run fine-tuning pipeline (requires TOGETHER_API_KEY and WANDB_API_KEY in .env)
uv run --env-file .env python training/finetune_socialized_context.py

# With custom config
uv run --env-file .env python training/finetune_socialized_context.py --config-path=custom.yaml

# Override data directory
uv run --env-file .env python training/finetune_socialized_context.py --data-dir=data/custom_dir
```

## Architecture

### Core Components

1. **SocialWorldModel** (`social_world_model/social_world_model.py`)
   - Main simulation engine that generates socialized contexts
   - Handles step-by-step simulation of agent interactions
   - Supports multiple modes: vanilla, socialized_context, simulation, etc.

2. **Database Models** (`social_world_model/database/database.py`)
   - `SocializedStructure`: Represents a single timestep with state, observations (per-agent), and actions (per-agent)
   - `SocializedContext`: Complete context with agent names and list of timesteps
   - `SocialSimulation`: Collection of multiple simulation runs
   - Special tags used in observations:
     - `<same_as_state />`: Observation identical to current state
     - `<same_as_last_action />`: Observation covers last action
     - `<mental_state>...</mental_state>`: Internal agent thoughts/beliefs
     - `none`: No observation/action at this timestep

3. **Task Modules** (`social_world_model/task_modules/`)
   - Each file handles a specific ToM benchmark:
     - `tomi.py`: ToMi dataset
     - `fantom.py`: FANToM dataset
     - `confaide.py`: ConFAIDE dataset
     - `hitom.py`: HiToM dataset
     - `cobra_frames.py`: COBRA-Frames dataset
     - `mmtom.py`: MMToM dataset
   - Each module provides:
     - Simulation function (e.g., `tomi_simulation`)
     - Vanilla preparation (e.g., `prepare_tomi_vanilla`)
     - Result creation (e.g., `create_tomi_result`)
     - Evaluation reporting (e.g., `tomi_evaluation_report`)
     - Socialized context prompt constant

4. **Agents** (`social_world_model/agents/`)
   - `llm_agent.py`: Basic LLM agent
   - `sotopia_agent.py`: Social agent using Sotopia framework (for dynamic simulations)

5. **Engine** (`social_world_model/engine/`)
   - Utility functions for context manipulation
   - `load_existing_socialized_contexts`: Load cached contexts
   - `dictlize`, `dictlize_socialized_structure`: Convert to dict format
   - `standardize_agent_names`: Normalize agent naming

6. **UI** (`ui/`)
   - Next.js web application for visualizing agent interactions
   - Features: agent timeline, conversation view, timestep details
   - Built with React, TypeScript, Tailwind CSS

### Benchmark Runner Architecture

The `ToMBenchmarkRunner` class (in `run_benchmarks.py`) orchestrates all benchmark execution:
- Supports 7+ benchmark types: tomi, fantom, confaide, cobra_frames, hitom, ori_tomi, mmtom
- Multiple modes: vanilla, socialized_context, simulation, generate_socialized_context, etc.
- Handles dataset loading, batch processing, result saving, and evaluation

### Data Flow

1. **Input**: Task-specific dataset (CSV/JSONL) with scenarios and questions
2. **Generation**: Create socialized contexts using LLM (optional, mode-dependent)
3. **Simulation**: Run agents through scenarios with generated contexts
4. **Evaluation**: Assess agent performance on ToM tasks
5. **Output**: Results saved to `logs/` and `data/` directories

## Branch Naming Convention

- New features: `feature_name/xx`
- Bug fixes: `bugfix/xx`

## Key Implementation Details

### Socialized Context Format

The system uses a structured JSON format to represent social interactions:
- **agents_names**: List of participating agents
- **socialized_context**: Array of timesteps, each containing:
  - `timestep`: Identifier (integer or description)
  - `state`: Current world state before actions
  - `observations`: Per-agent observations (dict)
  - `actions`: Per-agent actions (dict)

### Model Configuration

The project uses `sotopia` as a dependency (from git branch `redis-optional`) and supports various LLM backends through the generation utilities. Default models vary by task but commonly include GPT-4 variants, O1 models, and DeepSeek.

### Important Files

- `pyproject.toml`: Python dependencies and tool configuration
- `.pre-commit-config.yaml`: Pre-commit hooks for code quality
- `Experiments_doc.md`: Historical experiment logs and results
- `run_benchmarks.py`: Main benchmark orchestration script
- `run_dynamic.py`: Dynamic Sotopia-based simulation runner
- `run_benchmarks_agent_baselines.py`: Agent baseline evaluations

### Testing Philosophy

Tests use pytest with async support (`pytest-asyncio`). The test suite verifies core simulation functionality, particularly the `simulate_one_step` method which advances the socialized context by one timestep.
