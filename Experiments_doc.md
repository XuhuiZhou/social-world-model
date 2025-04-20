### Date: 2025-04-17

#### (gpt-4.1-2025-04-14 as the social world model)
```bash
uv run python run_dynamic.py --models "gpt-4.1-2025-04-14" --partner-model "gpt-4o-2024-08-06" --agent-type "social_world_model" --social-world-model-name "gpt-4.1-2025-04-14" --experiment-tag "social_world_model_trial_22_social_world_model_agent" --batch-size 100 --push-to-db --evaluator-model "o3-2025-04-16" --task "hard"
```

#### (gpt-4.1-2025-04-14 as the social world model)
```python
if __name__ == "__main__":
    benchmark(
        models=[
            "gpt-4.1-2025-04-14",
        ],
        partner_model="gpt-4o-2024-08-06",
        agent_class=SocialWorldModelAgent,
        tag="social_world_model_trial_21_social_world_model_agent",
        batch_size=100,
        push_to_db=True,
        evaluator_model="o3-2025-04-16",
        task="hard",
    )
```

```bash
    'gpt-4.1-2025-04-14 (test) o3-2025-04-16 as the evaluator': {
        'believability': (8.49, 0.09316471131625303),
        'relationship': (2.27, 0.15691667476472326),
        'knowledge': (3.05, 0.17628965427401846),
        'secret': (-0.4, 0.20372442082172657),
        'social_rules': (-0.53, 0.09928547623516916),
        'financial_and_material_benefits': (0.85, 0.2438421370003651),
        'goal': (5.89, 0.4114023093834762),
        'overall_score': (2.802857142857144, 0.10318589550453262),
        'setting_num': (20.0, 0.0),
        'episode_count': (100.0, 0.0)
    },
```

#### Vanilla setting again
```python
if __name__ == "__main__":
    benchmark(
        models=[
            "gpt-4.1-2025-04-14",
        ],
        partner_model="gpt-4o-2024-08-06",
        agent_class=LLMAgent,
        tag="social_world_model_trial_20_llm_agent",
        batch_size=100,
        push_to_db=True,
        evaluator_model="o3-2025-04-16",
        task="hard",
    )
```
```bash
{
    'gpt-4.1-2025-04-14 (test) o3-2025-04-16 as the evaluator': {
        'believability': (8.53, 0.09612442592708294),
        'relationship': (2.11, 0.16851156607072063),
        'knowledge': (2.98, 0.1477624028508127),
        'secret': (-0.14, 0.11899779377043286),
        'social_rules': (-0.3, 0.06887660372486616),
        'financial_and_material_benefits': (0.97, 0.21148828218803928),
        'goal': (5.53, 0.37103459954897744),
        'overall_score': (2.8114285714285727, 0.09570257295597835),
        'setting_num': (20.0, 0.0),
        'episode_count': (100.0, 0.0)
    },
}
```

#### Same setting (o1-2024-12-17 as the social world model)
```python
if __name__ == "__main__":
    benchmark(
        models=[
            "gpt-4.1-2025-04-14",
        ],
        partner_model="gpt-4o-2024-08-06",
        agent_class=SocialWorldModelAgent,
        tag="social_world_model_trial_19_social_world_model_agent",
        batch_size=100,
        push_to_db=True,
        evaluator_model="o3-2025-04-16",
        task="hard",
    )
```
```bash
    'gpt-4.1-2025-04-14 (test) o3-2025-04-16 as the evaluator': {
        'believability': (8.6, 0.09210142664205294),
        'relationship': (2.19, 0.15733175249038656),
        'knowledge': (3.05, 0.15423966921286156),
        'secret': (-0.39, 0.2549879298715988),
        'social_rules': (-0.55, 0.14431169989670878),
        'financial_and_material_benefits': (1.21, 0.24139309428362055),
        'goal': (6.82, 0.43439906714103577),
        'overall_score': (2.990000000000001, 0.12012382179421675),
        'setting_num': (20.0, 0.0),
        'episode_count': (100.0, 0.0)
    },
```

#### Vanilla agent with gpt-4.1-2025-04-14
```python
if __name__ == "__main__":
    benchmark(
        models=[
            "gpt-4.1-2025-04-14",
        ],
        partner_model="gpt-4o-2024-08-06",
        agent_class=LLMAgent,
        tag="social_world_model_trial_5_llm_agent",
        batch_size=100,
        push_to_db=True,
        evaluator_model="o3-2025-04-16",
        task="hard",
    )
```
```bash
{
    'gpt-4.1-2025-04-14 (test) o3-2025-04-16 as the evaluator': {
        'believability': (8.5, 0.10318722958514509),
        'relationship': (2.24, 0.14372550093848158),
        'knowledge': (3.08, 0.1697593419166751),
        'secret': (-0.34, 0.1010657869303964),
        'social_rules': (-0.41, 0.08308975644208082),
        'financial_and_material_benefits': (0.85, 0.21802721200046668),
        'goal': (5.71, 0.35201974271230035),
        'overall_score': (2.804285714285715, 0.08659240138066508),
        'setting_num': (20.0, 0.0),
        'episode_count': (100.0, 0.0)
    },
}
```

#### Test social world model agent (state then action generation)
```python
if __name__ == "__main__":
    benchmark(
        models=[
            "gpt-4.1-2025-04-14",
        ],
        partner_model="gpt-4o-2024-08-06",
        agent_class=SocialWorldModelAgent,
        tag="social_world_model_trial_5_social_world_model_agent",
        batch_size=100,
        push_to_db=True,
        evaluator_model="o3-2025-04-16",
        task="hard",
    )
```
```bash
{
    'gpt-4.1-2025-04-14 (test) o3-2025-04-16 as the evaluator': {
        'believability': (8.47, 0.09817418758123336),
        'relationship': (2.31, 0.13752151883647037),
        'knowledge': (3.0, 0.1410698514812268),
        'secret': (-0.06, 0.057819120974440585),
        'social_rules': (-0.34, 0.08407598060943491),
        'financial_and_material_benefits': (0.94, 0.237026924442614),
        'goal': (5.51, 0.38293730522103636),
        'overall_score': (2.8328571428571427, 0.08868266621727797),
        'setting_num': (20.0, 0.0),
        'episode_count': (100.0, 0.0)
    },
}
```

#### Test vanilla agent
```python
if __name__ == "__main__":
    benchmark(
        models=[
            "gpt-4o-2024-11-20",
        ],
        partner_model="gpt-4o-2024-08-06",
        agent_class=LLMAgent,
        tag="social_world_model_trial_2_llm_agent",
        batch_size=100,
        push_to_db=True,
        evaluator_model="o3-2025-04-16",
        task="hard",
    )
```
```bash
{    'gpt-4o-2024-11-20 (test) o3-2025-04-16 as the evaluator': {
        'believability': (8.24, 0.08642047210450769),
        'relationship': (2.04, 0.12169300555060825),
        'knowledge': (2.86, 0.14472719981545526),
        'secret': (0.0, 0.0),
        'social_rules': (-0.14, 0.07809750101543919),
        'financial_and_material_benefits': (0.39, 0.24685250485112742),
        'goal': (4.24, 0.33609579705130166),
        'overall_score': (2.518571428571429, 0.08310163365911022),
        'setting_num': (20.0, 0.0),
        'episode_count': (100.0, 0.0)
    },
}
```

#### Test social world model agent (normal socialized context generation)
```python
if __name__ == "__main__":
    benchmark(
        models=[
            "gpt-4o-2024-11-20",
        ],
        partner_model="gpt-4o-2024-08-06",
        agent_class=SocialWorldModelAgent,
        tag="social_world_model_trial_3_social_world_model_agent",
        batch_size=100,
        push_to_db=True,
        evaluator_model="o3-2025-04-16",
        task="hard",
    )
```

```bash
{
    'gpt-4o-2024-11-20 (test) o3-2025-04-16 as the evaluator': {
        'believability': (8.2, 0.12933196590235532),
        'relationship': (2.13, 0.12307135706058836),
        'knowledge': (3.04, 0.15641707092682292),
        'secret': (-0.05, 0.09936414241635139),
        'social_rules': (-0.06, 0.04870624264231726),
        'financial_and_material_benefits': (0.17, 0.25784313308254286),
        'goal': (4.13, 0.33837687192721305),
        'overall_score': (2.5085714285714302, 0.0912900850299864),
        'setting_num': (20.0, 0.0),
        'episode_count': (100.0, 0.0)
    },
}
```

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
