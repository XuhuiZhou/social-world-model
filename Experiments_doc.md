### Date: 2025-02-11 
```python
python run_tom_benchmarks.py run-benchmark "tomi" --dataset-path="data/rephrased_tomi_test_600.csv" --batch-size=6 --save --model-name="o1-2024-12-17" --mode="simulation" --example-analysis-file="data/social_contexts_example/tomi.json"
```

### Date: 2025-02-27
```python
uv run python run_tom_benchmarks.py run-benchmark "tomi" --dataset-path="data/rephrased_tomi_test_600.csv" --batch-size=6 --save --model-name="o1-2024-12-17" --mode="simulation"
```
This run uses the updated SocializedContext instructions. (but forgot to use the example analysis file)
The results are saved in `data/tomi_results/wo_analysis_example_simulation_o1-2024-12-17_rephrased_tomi_test_600.csv` (need to wait for Hyunwoo's new dataset)