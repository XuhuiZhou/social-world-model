# Social World Models reproducibility artifacts

This directory contains the lean, redistributable artifacts underlying the
reported results in *Social World Models*. It intentionally does **not** copy
third-party benchmark text or release prompts, model reasoning traces, or API
credentials.

## Contents

- `benchmark_sources.csv`: benchmark sources, expected local paths, paper
  sample sizes, and redistribution notes.
- `social_reasoning_results.csv`: the complete aggregate CoT and S3AP+CoT
  accuracy table reported in the paper.
- `sotopia_vanilla_scores.csv`: SOTOPIA-hard agent baselines and confidence
  interval half-widths.
- `sotopia_agent_swm_scores.csv`: the complete agent-by-SWM score matrix used
  for the appendix heatmap.
- `splits/`: identifier-and-hash-only manifests generated from the exact local
  input files. These files contain no stories, questions, answers, prompts, or
  model outputs.
- `generate_split_manifests.py`: regenerates and verifies the split manifests
  after the benchmark files have been downloaded.

## Obtain the benchmarks

Create `data/` in the repository root and place the processed files at the
paths listed in `benchmark_sources.csv`.

- [ToMi](https://github.com/facebookresearch/ToMi) provides the original data
  generator and is licensed CC BY-NC 4.0.
- ParaToMi is described in [Minding Language Models' (Lack of) Theory of
  Mind](https://aclanthology.org/2023.acl-long.780/). The copy used for this
  paper is access-controlled and is therefore not redistributed here.
- [FANToM](https://github.com/skywalker023/fantom) provides its official
  downloader and evaluation code under the MIT License.
- [Hi-ToM](https://github.com/ying-hui-he/Hi-ToM_dataset) provides the dataset
  under the Apache License 2.0.
- [MMToM-QA](https://github.com/chuanyangjin/MMToM-QA) provides its data and
  evaluation code under the MIT License.

The licenses above apply to the corresponding third-party resources. This
repository does not grant additional rights to those datasets.

## Verify local inputs

After placing the files under `data/`, run:

```bash
python reproducibility/generate_split_manifests.py \
  --data-root data \
  --output-dir reproducibility/splits \
  --check
```

Without `--check`, the command regenerates the manifests. Each record hash is
computed from a canonical serialization of the complete local record, so it
can verify the exact input without publishing its text.

## Split notes

- ToMi and ParaToMi use all 600 records in their processed paper files.
- FANToM uses the 64 short-conversation records in the processed file.
- HiToM uses all 100 records in the processed paper file.
- The paper reports a 302-of-600 MMToM-QA sample stratified by question type.
  The surviving project archive contains the 600 candidates but not the
  random-sampling seed or exact 302 identifiers. We therefore publish the 600
  candidate identifiers and hashes, rather than claiming an unverified split.

## Reported results

`social_reasoning_results.csv` is transcribed from the complete appendix table
and contains all seven evaluated models. `sotopia_agent_swm_scores.csv` and
`sotopia_vanilla_scores.csv` contain the values plotted in the appendix.
SOTOPIA scores use 100 simulations for each evaluated configuration with a
fixed GPT-4o partner.
