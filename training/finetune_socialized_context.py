"""Fine-tuning pipeline for socialized context generation using Together AI."""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import typer
import wandb
import yaml  # type: ignore
from rich import print
from together import Together  # type: ignore

from data_utils import (
    create_training_format,
    load_socialized_contexts,
    save_to_jsonl,
    train_val_split,
)

app = typer.Typer(pretty_exceptions_enable=False)


@dataclass
class FinetuneConfig:
    """Configuration for fine-tuning pipeline."""

    data_dir: str = ""
    output_dir: str = "training/output"
    base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference"
    n_epochs: int = 3
    batch_size: str = "max"
    learning_rate: float = 1e-5
    n_checkpoints: int = 3
    val_ratio: float = 0.1
    wandb_project: str = "social-world-model-finetuning"
    wandb_run_name: Optional[str] = None
    lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 16
    seed: int = 42


class TogetherAIFineTuner:
    """Fine-tuning pipeline manager for Together AI."""

    def __init__(self, config: FinetuneConfig):
        self.config = config
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError(
                "TOGETHER_API_KEY environment variable not set. "
                "Please set it with: export TOGETHER_API_KEY='your-key'"
            )
        self.client = Together(api_key=api_key)
        self.wandb_run: Optional[wandb.sdk.wandb_run.Run] = None

    def prepare_data(self) -> Tuple[Path, Path]:
        """
        Load socialized contexts, convert to ChatML, split, and save to JSONL.

        Returns:
            Tuple of (train_path, val_path) for the generated JSONL files
        """
        print("\n[bold]Step 1: Preparing data[/bold]")

        # Load JSON files
        records = load_socialized_contexts(self.config.data_dir)

        if len(records) == 0:
            raise ValueError(f"No valid records found in {self.config.data_dir}")

        # Convert to training format (matching socialize_context() template)
        print("\nConverting to training format...")
        training_records = [create_training_format(r) for r in records]

        # Train/val split
        print("\nSplitting into train/validation sets...")
        train_records, val_records = train_val_split(
            training_records, val_ratio=self.config.val_ratio, seed=self.config.seed
        )

        # Save to JSONL
        print("\nSaving to JSONL files...")
        train_path = Path(self.config.output_dir) / "train.jsonl"
        val_path = Path(self.config.output_dir) / "val.jsonl"

        save_to_jsonl(train_records, train_path)
        save_to_jsonl(val_records, val_path)

        print("\n[green]✓[/green] Data preparation complete")
        print(f"  Train: {train_path} ({len(train_records)} samples)")
        print(f"  Val: {val_path} ({len(val_records)} samples)")

        return train_path, val_path

    def upload_files(self, train_path: Path, val_path: Path) -> Tuple[str, str]:
        """
        Upload training files to Together AI.

        Args:
            train_path: Path to training JSONL file
            val_path: Path to validation JSONL file

        Returns:
            Tuple of (train_file_id, val_file_id)
        """
        print("\n[bold]Step 2: Uploading files to Together AI[/bold]")

        # Upload training file
        print(f"Uploading {train_path}...")
        train_file = self.client.files.upload(file=str(train_path))
        print(f"[green]✓[/green] Train file uploaded: {train_file.id}")

        # Upload validation file
        print(f"Uploading {val_path}...")
        val_file = self.client.files.upload(file=str(val_path))
        print(f"[green]✓[/green] Validation file uploaded: {val_file.id}")

        return train_file.id, val_file.id

    def create_finetune_job(self, train_file_id: str, val_file_id: str) -> str:
        """
        Initialize W&B and create fine-tuning job on Together AI.

        Args:
            train_file_id: Together AI file ID for training data
            val_file_id: Together AI file ID for validation data

        Returns:
            Job ID for the fine-tuning job
        """
        print("\n[bold]Step 3: Creating fine-tuning job[/bold]")

        # Initialize W&B
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if not wandb_api_key:
            print(
                "[yellow]Warning:[/yellow] WANDB_API_KEY not set. "
                "W&B integration will be disabled."
            )
            wandb_api_key = None
        else:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config={
                    "base_model": self.config.base_model,
                    "n_epochs": self.config.n_epochs,
                    "learning_rate": self.config.learning_rate,
                    "batch_size": self.config.batch_size,
                    "lora": self.config.lora,
                    "lora_rank": self.config.lora_rank,
                    "lora_alpha": self.config.lora_alpha,
                    "train_file_id": train_file_id,
                    "val_file_id": val_file_id,
                },
            )
            print(f"[green]✓[/green] W&B initialized: {self.wandb_run.url}")

        # Create fine-tuning job
        print(f"\nCreating fine-tuning job for {self.config.base_model}...")
        job = self.client.fine_tuning.create(
            training_file=train_file_id,
            validation_file=val_file_id,
            model=self.config.base_model,
            n_epochs=self.config.n_epochs,
            n_checkpoints=self.config.n_checkpoints,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            suffix="socialized-context-generator",
            wandb_api_key=wandb_api_key,
            lora=self.config.lora,
        )

        print(f"[green]✓[/green] Fine-tuning job created: {job.id}")
        if self.wandb_run:
            wandb.log({"job_id": job.id})

        return str(job.id)

    def monitor_job(self, job_id: str) -> str:
        """
        Monitor fine-tuning job and log metrics to W&B.

        Args:
            job_id: Together AI job ID to monitor

        Returns:
            Fine-tuned model identifier
        """
        print("\n[bold]Step 4: Monitoring fine-tuning job[/bold]")
        print(f"Job ID: {job_id}")
        print("Checking status every 60 seconds...")

        last_event_count = 0

        while True:
            # Get job status
            job = self.client.fine_tuning.retrieve(job_id)
            status = job.status

            # Log status to W&B
            if self.wandb_run:
                wandb.log({"status": status, "timestamp": time.time()})

            # Fetch and log training events
            try:
                events = list(self.client.fine_tuning.list_events(job_id))

                # Log new events
                if len(events) > last_event_count:
                    for event in events[last_event_count:]:
                        # Handle different event formats (dict, object, tuple)
                        if isinstance(event, dict):
                            event_dict = event
                        elif hasattr(event, "model_dump"):
                            event_dict = event.model_dump()
                        elif hasattr(event, "__dict__"):
                            event_dict = event.__dict__
                        elif isinstance(event, tuple):
                            # Skip tuple events that can't be converted
                            continue
                        else:
                            continue

                        # Log metrics if available
                        if self.wandb_run:
                            log_data = {}
                            if "training_loss" in event_dict:
                                log_data["train/loss"] = event_dict["training_loss"]
                            if "validation_loss" in event_dict:
                                log_data["val/loss"] = event_dict["validation_loss"]
                            if "step" in event_dict:
                                log_data["train/step"] = event_dict["step"]
                            if log_data:
                                wandb.log(log_data)

                    last_event_count = len(events)
            except Exception as e:
                print(f"[yellow]Warning:[/yellow] Error fetching events: {e}")

            # Display status
            print(f"Status: {status} (checked at {time.strftime('%H:%M:%S')})")

            # Check if job is complete (handle both string and enum status)
            status_str = str(status).lower()
            if any(
                s in status_str
                for s in ["succeeded", "failed", "cancelled", "completed"]
            ):
                break

            # Wait before next check
            time.sleep(60)

        # Handle completion
        if status == "succeeded":
            model_name = str(job.fine_tuned_model)
            print("\n[green bold]✓ Fine-tuning completed successfully![/green bold]")
            print(f"Model: {model_name}")

            if self.wandb_run:
                wandb.log({"final_model": model_name})

            return model_name
        else:
            error_msg = f"Fine-tuning {status}"
            if hasattr(job, "error"):
                error_msg += f": {job.error}"
            raise Exception(error_msg)

    def run(self) -> str:
        """
        Execute the complete fine-tuning pipeline.

        Returns:
            Fine-tuned model identifier
        """
        try:
            # Step 1: Prepare data
            train_path, val_path = self.prepare_data()

            # Step 2: Upload files
            train_file_id, val_file_id = self.upload_files(train_path, val_path)

            # Step 3: Create fine-tuning job
            job_id = self.create_finetune_job(train_file_id, val_file_id)

            # Step 4: Monitor job
            model_name = self.monitor_job(job_id)

            return model_name

        finally:
            # Cleanup W&B
            if self.wandb_run:
                self.wandb_run.finish()


@app.command()
def finetune(
    config_path: str = typer.Option(
        "training/finetune_config.yaml", help="Path to config YAML file"
    ),
    data_dir: str = typer.Option(
        "data/tomi_results/socialized_context_o3-2025-04-16_rephrased_tomi_train_6000.csv_o3-2025-04-16",
        help="Directory containing training JSON files",
    ),
    output_dir: str = typer.Option(
        "training/output", help="Output directory for processed data"
    ),
) -> None:
    """Run fine-tuning pipeline for socialized context generation."""
    print("[bold blue]Together AI Fine-Tuning Pipeline[/bold blue]")
    print("Fine-tuning Meta-Llama-3.1-8B-Instruct for socialized context generation\n")

    # Load config from YAML if exists
    config = FinetuneConfig()
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

    # Display configuration
    print("\n[bold]Configuration:[/bold]")
    print(f"  Data directory: {config.data_dir}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Base model: {config.base_model}")
    print(f"  Epochs: {config.n_epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  LoRA: {config.lora}")
    print(f"  W&B project: {config.wandb_project}")

    # Run pipeline
    finetuner = TogetherAIFineTuner(config)
    model_name = finetuner.run()

    # Display final results
    print("\n" + "=" * 60)
    print("[bold green]Fine-tuning Complete![/bold green]")
    print("=" * 60)
    print(f"\nModel name: [bold]{model_name}[/bold]")
    print("\nTo use in your code:")
    print(f'  model_name="together_ai/{model_name}"')
    print("\nExample usage:")
    print("  from social_world_model.generation_utils import agenerate")
    print("  result = await agenerate(")
    print(f'      model_name="together_ai/{model_name}",')
    print('      template="Generate socialized context for: {story}",')
    print('      input_values={"story": "..."},')
    print("      temperature=0.7")
    print("  )")


if __name__ == "__main__":
    app()
