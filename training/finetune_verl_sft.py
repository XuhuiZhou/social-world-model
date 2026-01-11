"""VERL SFT Fine-tuning pipeline for socialized context generation.

This replaces the Together AI fine-tuner with local VERL SFT training.
Uses the same data preparation from data_utils.py but trains locally on your GPUs.
"""
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import typer
import yaml  # type: ignore
from rich import print

from data_utils import (
    create_training_format,
    load_socialized_contexts,
    train_val_split,
)

app = typer.Typer(pretty_exceptions_enable=False)


@dataclass
class VERLSFTConfig:
    """Configuration for VERL SFT training pipeline."""

    # Data
    data_dir: str = ""
    output_dir: str = "training/verl_output"
    val_ratio: float = 0.1
    seed: int = 42

    # Model
    base_model: str = "microsoft/Phi-3-mini-4k-instruct"
    model_max_length: int = 4096

    # Training
    n_gpus: int = 3
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    total_training_steps: int = 1000
    warmup_steps: int = 100
    save_freq: int = 100
    val_freq: int = 50

    # Parallelism
    use_fsdp: bool = True
    use_sequence_parallel: bool = False
    sequence_parallel_size: int = 1

    # W&B
    wandb_project: str = "social-world-model-verl-sft"
    wandb_run_name: Optional[str] = None


class VERLSFTTrainer:
    """VERL SFT training pipeline manager."""

    def __init__(self, config: VERLSFTConfig):
        self.config = config

    def prepare_data(self) -> Tuple[Path, Path]:
        """
        Load socialized contexts, convert to training format, and save as parquet.

        Returns:
            Tuple of (train_path, val_path) for the generated parquet files
        """
        print("\n[bold]Step 1: Preparing data for VERL SFT[/bold]")

        # Load JSON files (reuse existing data_utils)
        records = load_socialized_contexts(self.config.data_dir)

        if len(records) == 0:
            raise ValueError(f"No valid records found in {self.config.data_dir}")

        # Convert to training format (ChatML messages)
        print("\nConverting to training format...")
        training_records = [create_training_format(r) for r in records]

        # Train/val split
        print("\nSplitting into train/validation sets...")
        train_records, val_records = train_val_split(
            training_records, val_ratio=self.config.val_ratio, seed=self.config.seed
        )

        # Convert to VERL format (parquet with specific schema)
        print("\nConverting to VERL parquet format...")
        train_path = Path(self.config.output_dir) / "train.parquet"
        val_path = Path(self.config.output_dir) / "val.parquet"

        self._save_to_parquet(train_records, train_path)
        self._save_to_parquet(val_records, val_path)

        print("\n[green]✓[/green] Data preparation complete")
        print(f"  Train: {train_path} ({len(train_records)} samples)")
        print(f"  Val: {val_path} ({len(val_records)} samples)")

        return train_path, val_path

    def _save_to_parquet(self, records: list, output_path: Path) -> None:
        """
        Convert records to VERL-compatible parquet format.

        VERL expects:
        - data.prompt_key: column name for prompts
        - data.response_key: column name for responses
        - Or data.prompt_dict_keys / data.response_dict_keys for structured data
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract prompts and responses from messages
        data = []
        for record in records:
            messages = record["messages"]

            # Find user and assistant messages
            user_msg = next((m for m in messages if m["role"] == "user"), None)
            assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)

            if user_msg and assistant_msg:
                data.append({
                    "prompt": user_msg["content"],
                    "response": assistant_msg["content"],
                })

        # Create DataFrame and save
        df = pd.DataFrame(data)
        df.to_parquet(output_path, engine='pyarrow', index=False)

        print(f"Saved {len(df)} records to {output_path}")

    def create_training_script(
        self,
        train_path: Path,
        val_path: Path
    ) -> Path:
        """
        Generate the VERL SFT training script.

        Returns:
            Path to the generated training script
        """
        print("\n[bold]Step 2: Creating VERL training script[/bold]")

        script_path = Path(self.config.output_dir) / "run_sft.sh"
        script_path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate train_batch_size (must be divisible by n_gpus)
        train_batch_size = self.config.micro_batch_size * self.config.n_gpus * self.config.gradient_accumulation_steps

        # Build torchrun command
        script_content = f"""#!/bin/bash
# VERL SFT Training Script for Social World Model
# Auto-generated by finetune_verl_sft.py

set -x  # Print commands for debugging

# Environment setup
export CUDA_HOME=/usr/local/cuda-12.3
export PATH=/usr/local/cuda-12.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH
export VLLM_ATTENTION_BACKEND=XFORMERS

# Training configuration
N_GPUS={self.config.n_gpus}
TRAIN_FILE="{train_path.absolute()}"
VAL_FILE="{val_path.absolute()}"
MODEL="{self.config.base_model}"
OUTPUT_DIR="{Path(self.config.output_dir).absolute()}"

# WandB configuration
WANDB_PROJECT="{self.config.wandb_project}"
WANDB_RUN_NAME="{self.config.wandb_run_name or 'verl-sft-run'}"

# Launch training with torchrun
torchrun --standalone --nnodes=1 --nproc_per_node=$N_GPUS \\
    -m verl.trainer.fsdp_sft_trainer \\
    data.train_files=$TRAIN_FILE \\
    data.val_files=$VAL_FILE \\
    data.prompt_key=prompt \\
    data.response_key=response \\
    data.micro_batch_size={self.config.micro_batch_size} \\
    data.train_batch_size={train_batch_size} \\
    data.max_length={self.config.model_max_length} \\
    model.partial_pretrain=$MODEL \\
    optim.lr={self.config.learning_rate} \\
    optim.lr_warmup_steps={self.config.warmup_steps} \\
    trainer.default_local_dir=$OUTPUT_DIR \\
    trainer.project_name=$WANDB_PROJECT \\
    trainer.experiment_name=$WANDB_RUN_NAME \\
    trainer.logger=['wandb','console'] \\
    trainer.total_training_steps={self.config.total_training_steps} \\
    trainer.save_freq={self.config.save_freq} \\
    trainer.test_freq={self.config.val_freq} \\
    trainer.n_gpus_per_node=$N_GPUS \\
    trainer.nnodes=1
"""

        # Add optional params
        if self.config.use_sequence_parallel:
            script_content += f" \\\n    ulysses_sequence_parallel_size={self.config.sequence_parallel_size}"

        script_path.write_text(script_content)
        script_path.chmod(0o755)  # Make executable

        print(f"[green]✓[/green] Training script created: {script_path}")
        return script_path

    def run_training(self, script_path: Path) -> None:
        """
        Execute the VERL SFT training script.

        Args:
            script_path: Path to the training script
        """
        print("\n[bold]Step 3: Running VERL SFT training[/bold]")
        print(f"Script: {script_path}")
        print(f"GPUs: {self.config.n_gpus} x RTX A6000")
        print(f"Model: {self.config.base_model}")
        print(f"\nStarting training...\n")

        # Check W&B API key (VERL will handle initialization)
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key:
            print(f"[green]✓[/green] W&B API key found. VERL will log to project: {self.config.wandb_project}")
        else:
            print("[yellow]Warning:[/yellow] WANDB_API_KEY not set. W&B logging disabled.")

        # Run training subprocess
        try:
            result = subprocess.run(
                [str(script_path)],
                cwd=Path.cwd(),
                check=True,
                text=True,
                env={**os.environ, "WANDB_API_KEY": wandb_api_key or ""},
            )

            print("\n[green bold]✓ Training completed successfully![/green bold]")

        except subprocess.CalledProcessError as e:
            print(f"\n[red bold]✗ Training failed with error:[/red bold]")
            print(f"  {e}")
            raise

    def run(self) -> str:
        """
        Execute the complete VERL SFT training pipeline.

        Returns:
            Path to the fine-tuned model checkpoint
        """
        try:
            # Step 1: Prepare data
            train_path, val_path = self.prepare_data()

            # Step 2: Create training script
            script_path = self.create_training_script(train_path, val_path)

            # Step 3: Run training
            self.run_training(script_path)

            # Find final checkpoint
            checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
            if checkpoint_dir.exists():
                checkpoints = sorted(checkpoint_dir.glob("step_*"))
                if checkpoints:
                    final_checkpoint = checkpoints[-1]
                    return str(final_checkpoint)

            return str(Path(self.config.output_dir) / "final_model")

        finally:
            # Cleanup (VERL handles wandb.finish() automatically)
            pass


@app.command()
def finetune(
    config_path: str = typer.Option(
        "training/verl_sft_config.yaml", help="Path to config YAML file"
    ),
    data_dir: str = typer.Option(
        "data/tomi_results/socialized_context_o3-2025-04-16_rephrased_tomi_train_6000.csv_o3-2025-04-16",
        help="Directory containing training JSON files",
    ),
    output_dir: str = typer.Option(
        "training/verl_output", help="Output directory for checkpoints and data"
    ),
    base_model: str = typer.Option(
        "microsoft/Phi-3-mini-4k-instruct",
        help="Base model to fine-tune"
    ),
    n_gpus: int = typer.Option(
        3, help="Number of GPUs to use"
    ),
    micro_batch_size: int = typer.Option(
        2, help="Micro batch size per GPU"
    ),
    total_steps: int = typer.Option(
        1000, help="Total training steps"
    ),
) -> None:
    """Run VERL SFT fine-tuning pipeline for socialized context generation."""
    print("[bold blue]VERL SFT Fine-Tuning Pipeline[/bold blue]")
    print("Replacing Together AI with local VERL training\n")

    # Load config from YAML if exists
    config = VERLSFTConfig()
    if Path(config_path).exists():
        print(f"Loading config from {config_path}")
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)

    # Override with CLI args
    config.data_dir = data_dir
    config.output_dir = output_dir
    config.base_model = base_model
    config.n_gpus = n_gpus
    config.micro_batch_size = micro_batch_size
    config.total_training_steps = total_steps

    # Display configuration
    print("\n[bold]Configuration:[/bold]")
    print(f"  Data directory: {config.data_dir}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Base model: {config.base_model}")
    print(f"  GPUs: {config.n_gpus} x RTX A6000")
    print(f"  Micro batch size: {config.micro_batch_size} per GPU")
    print(f"  Effective batch: {config.micro_batch_size * config.n_gpus * config.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Total steps: {config.total_training_steps}")
    print(f"  W&B project: {config.wandb_project}")

    # Run pipeline
    trainer = VERLSFTTrainer(config)
    model_path = trainer.run()

    # Display final results
    print("\n" + "=" * 60)
    print("[bold green]Fine-tuning Complete![/bold green]")
    print("=" * 60)
    print(f"\nModel checkpoint: [bold]{model_path}[/bold]")
    print("\nTo use the fine-tuned model:")
    print(f'  model_name="{model_path}"')
    print("\nOr load with Transformers:")
    print("  from transformers import AutoModelForCausalLM")
    print(f'  model = AutoModelForCausalLM.from_pretrained("{model_path}")')


if __name__ == "__main__":
    app()
