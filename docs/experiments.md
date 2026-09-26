# Experiment commands

Run commands from the repository root. Use `--help` on each entry point for path overrides. Training and model evaluation require datasets and compatible checkpoints, except the synthetic model checks and head benchmark. No training is launched by the repository verification command.

## Principal comparison

```bash
python -m experiments.run_seed_study train --shot 1
python -m experiments.run_seed_study train --shot 5
python -m experiments.run_seed_study clean --shot 1
python -m experiments.run_seed_study clean --shot 5
```

`--workers 0` is useful for debugging training DataLoaders. Standard training seeds are 0--4; existing checkpoints are skipped unless `--overwrite` is supplied. `python -m scripts.check_experiment_inputs` checks the main model-evaluation inputs.

## Multi-query comparison

```bash
python -m experiments.run_multiquery_study --shot 5 --query-groups 2 3 4 5
python -m experiments.evaluate_multiquery_study --shot 5 --query-groups 1 2 3 4 5
python -m experiments.run_multiquery_study --shot 1 --query-groups 3 5
python -m experiments.evaluate_multiquery_study --shot 1 --query-groups 1 3 5
```

The q=1 condition reuses seed-study checkpoints. Other group sizes have separately trained heads. The 1-shot sweep was an exploratory follow-up to the 5-shot study. No query labels are used for predictions or grouping; labels are used afterwards to check group diversity.

## Adaptation and scoring

```bash
python -m experiments.run_factorial_study --shot 5
python -m experiments.evaluate_factorial_study --shot 5
```

The Euclidean support-only and WIPT cells reuse seed-study checkpoints. Raw Euclidean/cosine have no trained head. Other configurations train their corresponding scorer/adaptation components. Cosine training in this grid differs from the inference-only cosine ablation in both objective context and support aggregation.

## Decision changes

```bash
python -m experiments.analyze_boundary_all_domains --shots 1 5
python -m experiments.analyze_shot_domain_dependence --shot 1
python -m experiments.analyze_shot_domain_dependence --shot 5
```

The all-domain boundary script can resume from per-evaluation-seed partitions. Completed query tables are included in this repository; transient partitions are excluded. Its older EuroSAT-only counterpart, `experiments.analyze_boundary_across_seeds`, is retained as the separate earlier analysis.

## Corruption

```bash
python -m experiments.evaluate_corruption_across_seeds --shot 5
```

The replicated study uses EuroSAT with five independently trained heads and 1,000 matched episodes per corruption condition. Brightness/contrast, Gaussian noise, and their combination affect both support and query images.

## Source shifts

```bash
python -m experiments.run_shift_study --shot 1 --modes shared
python -m experiments.run_shift_study --shot 5 --modes shared
python -m experiments.evaluate_shift_study --shot 1 --modes shared
python -m experiments.evaluate_shift_study --shot 5 --modes shared
```

The reported mixed-shift follow-up uses `--modes mixed --seeds 0`. It is a single-training-seed screen, not a five-seed claim. `cross` mode remains available but no completed cross-mode result is supplied. Training randomly selects clean or shifted episodes according to the configured shift probability; the obsolete implementation that computed both losses each step is not included.

## Selected-checkpoint diagnostics

The `eval/` scripts use root-level paths from `configs.experiment.CHECKPOINTS`. These differ from the replicated seed-study checkpoint tree.

| Command | Purpose |
|---|---|
| `python -m eval.evaluate_in_domain` | Held-out miniImageNet reference |
| `python -m eval.evaluate_cross_domain` | Selected-checkpoint target accuracy and geometry |
| `python -m eval.evaluate_corruptions` | Selected-checkpoint A/B/C corruptions |
| `python -m eval.analyze_query_outcomes` | Earlier EuroSAT transitions and margins |
| `python -m eval.analyze_support_noise` | Support-pathway isolation diagnostic |
| `python -m eval.evaluate_inference_ablations` | Pooling, scoring and multiple-prototype variants |
| `python -m eval.evaluate_margin_variant` | Margin-trained comparison |

The support-noise script holds WIPT's clean transformed query fixed and recomputes noisy-support prototypes. This is not ordinary noisy WIPT inference. Its original historical reference checkpoint is not distributed, and its training seed is not established; five evaluation seeds do not imply five independently trained heads.

Selected-checkpoint training entry points include `train.train_wipt`, `train.train_baselines`, `train.train_refined_variants` and `train.train_margin_variant`.

## Computational cost

```bash
python -m eval.benchmark_efficiency --shot 1 --output results/raw/efficiency_1shot.csv
python -m eval.benchmark_efficiency --shot 5 --output results/raw/efficiency_5shot.csv
```

The benchmark uses synthetic 384-dimensional embeddings and randomly initialized heads, excluding image loading and ViT feature extraction. The historical `backward_*` CSV columns measure forward plus backward, excluding the optimizer step. Peak allocated memory includes resident tensors/modules; it is not pure activation memory. Latencies depend on the hardware and implementation.

## Optional research code

`experimental/` contains meta-training/test-time support adaptation and alternative-source runners. They are separate from the reported main study; no results are claimed for them here. Their target-time updates must not be described as WIPT's frozen-parameter inference. The optional runner's generic target arguments are handled by `experiments.evaluate_seed_models`.
