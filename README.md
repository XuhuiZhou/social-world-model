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
