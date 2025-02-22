# social-world-model
use social simulation as a world model

## Setup

```bash
uv sync --all-extras
```

## Run ToM Benchmarks

```bash
python run_tom_benchmarks.py run-benchmark "tomi" --dataset-path="data/rephrased_tomi_test_600.csv" --batch-size=8 --save --model-name="together_ai/deepseek-ai/DeepSeek-R1" --mode="vanilla"
```

or 

```bash
python run_tom_benchmarks.py run-benchmark "tomi" --dataset-path="data/rephrased_tomi_test_600.csv" --batch-size=1 --save --model-name="o1-2024-12-17" --mode="simulation"
```

## To Simply Evaluate

```bash
python run_tom_benchmarks.py evaluate-results tomi o1-2024-12-17 --mode="vanilla"
```

