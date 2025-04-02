### Date: 2025-03-28
run `together_ai/deepseek-ai/DeepSeek-R1` on confaide dataset.

### Date: 2025-03-22
```bash
uv run python run_benchmarks.py "confaide"  --batch-size=40 --save --model-name="o1-2024-12-17" --mode="socialized_context" --continue-mode=continue
```

```bash
uv run python run_benchmarks.py "confaide"  --batch-size=20 --save --model-name="o1-2024-12-17" --mode="generate_socialized_context" --continue-mode=continue --example-analysis-file="data/social_contexts_example/confaide.json"
```

```bash
uv run python run_benchmarks.py "confaide"  --batch-size=40 --save --model-name="o1-2024-12-17" --mode="vanilla" --continue-mode=continue
```

### Date: 2025-03-22
```python
uv run python run_benchmarks.py "fantom"  --batch-size=6 --save --model-name="o1-2024-12-17" --mode="simulation" --continue-mode=continue
```

### Date: 2025-03-17
```python
uv run python run_benchmarks.py "fantom"  --batch-size=1 --save --model-name="o1-2024-12-17" --mode="generate_socialized_context" --continue-mode=continue --example-analysis-file="data/social_contexts_example/fantom.json"
```
**Interface changed; directly use run-benchmark for all modes**

```python
uv run python run_benchmarks.py run-benchmark "fantom"  --batch-size=1 --save --model-name="o1-2024-12-17" --mode="socialized_context" --continue-mode=continue --example-analysis-file="data/social_contexts_example/fantom.json"
```
Generates the socialized context for the new Fantom dataset.

### Date: 2025-03-16
New Fantom dataset is ready.
```python
uv run python run_benchmarks.py run-benchmark "fantom"  --batch-size=6 --save --model-name="o1-2024-12-17" --mode="vanilla" --continue-mode=continue
```
The default dataset path is `data/fantom_data/fantom_for_tt_processed.jsonl`.


### Date: 2025-02-28
```python
uv run python run_benchmarks.py run-benchmark "fantom" --dataset-path="data/Percept_FANToM/Percept-FANToM-flat.csv" --batch-size=6 --save --model-name="o1-2024-12-17" --mode="socialized_context" --example-analysis-file="data/social_contexts_example/fantom.json"
```
This run has to run all the distinct set_ids in the dataset first since many of them are the same scenario.

```python
uv run python run_benchmarks.py run-benchmark "fantom" --dataset-path="data/Percept_FANToM/Percept-FANToM-flat.csv" --batch-size=6 --save --model-name="o1-2024-12-17" --mode="socialized_context" --example-analysis-file="data/social_contexts_example/fantom.json" --continue-mode="continue"
```
This run continues from the previous run.

### Date: 2025-02-27
```python
uv run python run_benchmarks.py run-benchmark "tomi" --dataset-path="data/rephrased_tomi_test_600.csv" --batch-size=6 --save --model-name="o1-2024-12-17" --mode="socialized_context"
```
This run uses the updated SocializedContext instructions. (but forgot to use the example analysis file)
The results are saved in `data/tomi_results/wo_analysis_example_socialized_context_o1-2024-12-17_rephrased_tomi_test_600.csv` (need to wait for Hyunwoo's new dataset)


### Date: 2025-02-11
```python
python run_benchmarks.py run-benchmark "tomi" --dataset-path="data/rephrased_tomi_test_600.csv" --batch-size=6 --save --model-name="o1-2024-12-17" --mode="socialized_context" --example-analysis-file="data/social_contexts_example/tomi.json"
```
