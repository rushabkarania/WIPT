# Result records

The retained CSV files support numerical analysis and figure regeneration without image datasets or model weights. Model evaluation itself requires both. The files describe several distinct experiment protocols; they should not be pooled as if they were one study.

| Location under `results/` | Contents and use |
|---|---|
| `raw/training_seed_study/` | Principal 1-shot and 5-shot comparisons, with independently trained heads and matched target episodes |
| `raw/multiquery/` | Query-group experiments, including the 1-shot follow-up |
| `raw/factorial/5shot/` | Adaptation-by-scorer comparison |
| `raw/boundary_all_domains/` | Complete query-level records for the domain/shot boundary analyses |
| `raw/shot_domain_mechanism/` | Decision-change and geometry summaries by domain and shot |
| `raw/shift_study/` | Source-shift training results; distinguish replicated shared-shift runs from the single-seed mixed-shift screen |
| `raw/efficiency_1shot.csv`, `raw/efficiency_5shot.csv` | Synthetic-embedding classification-head benchmarks |
| `paper/` | Earlier selected-checkpoint tables for supplementary diagnostics, not the principal five-training-seed comparison |
| `figure_inputs/` | Retained inputs for supplementary plots |
| `manifest.csv` | Inventory of the earlier selected-checkpoint tables; not a complete inventory of all experiments |

## Statistical units

Training seeds and evaluation seeds represent different sources of variation. The principal study uses training seeds 0–4 and evaluation seeds 42, 123, 2024, 7 and 999. There are 300 episodes per evaluation seed, giving 1,500 matched episodes per target and trained head. Principal paired confidence intervals are computed across training seeds on that fixed episode bank.

The support-noise diagnostic instead uses one historical reference model and five evaluation seeds, with 200 episodes per evaluation seed. Its numerical summary does not establish replication over independently trained models.

Compact checkpoint metadata exports are not exhaustive checkpoint inventories. In particular, an incomplete export for a query-group condition should not be interpreted as evidence that only four heads were trained. No model weight files are distributed here.

## Regenerated statistics

Run `python -m scripts.generate_figures` from the repository root. In addition to eight PNG figures, this writes:

- `figures/derived/primary_paired_statistics.csv`: paired model differences and confidence intervals;
- `figures/derived/boundary_bins_split_at_zero.csv`: boundary bins with a separate split at zero margin;
- `figures/derived/boundary_localization.csv`: summaries of where decision changes occur;
- `figures/derived/rescue_break_contributions_by_seed.csv`: gains from rescued predictions and losses from broken predictions;
- `figures/generation_summary.json`: generated filenames and validation counts.

The latest boundary figure uses the regenerated zero-split bins. Older binned summaries remain available as historical outputs and need not match those bin definitions. The generator validates 675,000 query records and 12 paired comparisons against the retained principal results.

## Efficiency measurements

The historical `backward_*` column names mean **forward plus backward latency**, excluding the optimizer step. Memory values labelled `peak_memory_mb` use a divisor of 1024 squared (MiB). Peak CUDA allocated memory includes resident tensors and model parameters; it is not a measurement of activation memory alone. Neither the latency nor the memory benchmark includes ViT image feature extraction or image loading.

## File preservation

Retained source CSV contents are unchanged. Complete combined tables are included; redundant restart partitions, development smoke-run outputs and a byte-identical efficiency-table copy are omitted. Existing result files can be overwritten by experiment commands: use a separate output directory when running new experiments.
