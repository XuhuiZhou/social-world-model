prompts in prepare_hitom_vanilla:
- https://github.com/ying-hui-he/Hi-ToM_dataset/blob/main/Hi-ToM_data


set_id

todo:
1. modify the results json files
2. HITOM_SOCIALIZED_CONTEXT_PROMPT
2. write a socialized context example and put somewhere
3. hitom_simulation


# hitom

## vanilla test
uv run python run_benchmarks.py "hitom" --dataset-path="data/hitom_data/processed_hitom_data.csv" --batch-size=100 --save --model-name="o1-2024-12-17" --mode="vanilla" --continue-mode="continue"

## generate socialized context
uv run python run_benchmarks.py "hitom" --dataset-path="data/hitom_data/processed_hitom_data.csv" --batch-size=100 --save --model-name="o1-2024-12-17" --mode="generate_socialized_context" --continue-mode="continue" --example-analysis-file="data/social_contexts_example/hitom.json"

uv run python run_benchmarks.py "hitom" --dataset-path="data/hitom_data/processed_hitom_data.csv" --batch-size=100 --save --model-name="o1-2024-12-17" --mode="socialized_context" --continue-mode="continue" --example-analysis-file="data/social_contexts_example/hitom.json"

## 