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
uv run python run_benchmarks.py --help
```
to get to know the options.

```bash
uv run python run_benchmarks.py "tomi" --dataset-path="data/rephrased_tomi_test_600.csv" --batch-size=8 --save --model-name="together_ai/deepseek-ai/DeepSeek-R1" --mode="vanilla"
```

or

```bash
uv run python run_benchmarks.py "tomi" --dataset-path="data/rephrased_tomi_test_600.csv" --batch-size=1 --save --model-name="o1-2024-12-17" --mode="simulation"
```

## To Simply Evaluate after running the benchmarks

```bash
uv run python run_benchmarks.py "tomi" --dataset-path="data/rephrased_tomi_test_600.csv" --batch-size=1 --save --model-name="o1-2024-12-17" --mode="simulation" --continue-mode="continue"
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


