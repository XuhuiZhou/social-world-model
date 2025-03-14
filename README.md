# social-world-model
use social simulation as a world model
You need to set the environment variable `OPENAI_API_KEY` to use the API.
You also need to `pip install uv` to run the following setup.
You also need to have a `data` folder in the root of the project.
And I think you are good to go.
Let me know if you run into any issues.

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
