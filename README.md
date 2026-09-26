# WIPT: Within-Instance Prototypical Transformer

Research code for **Query-Conditioned Prototype Adaptation for Cross-Domain Few-Shot Learning: Single-Query Inference, Controlled Comparisons, and Failure Modes**, by Rushab Rasik Karania and Tomas Maul.

WIPT jointly processes one query embedding and the labelled support embeddings, then forms query-conditioned class prototypes. The image encoder remains frozen; prediction requires no target-time parameter updates.

This repository contains training and evaluation code, experimental result records, and figure generation. Dataset images, model checkpoints and the manuscript are not included. No arXiv URL is included.

## Method

- Shared encoder: `vit_small_patch16_224.augreg_in21k_ft_in1k` from `timm`, producing 384-dimensional image embeddings.
- Episodic head: two Transformer layers, six attention heads, dropout 0.1.
- Prediction: negative Euclidean distance from the transformed query to class means of transformed supports.
- Controls: frozen ProtoNet and a capacity-matched support-only Transformer.
- Tasks: 5-way classification with 1 or 5 supports and 15 queries per class.

The support-only control transforms supports but scores the raw query. Comparing it with WIPT changes both the support and query pathways; it does not isolate prototype conditioning alone.

## Main results

Mean accuracy (%) across five independently trained heads, evaluated on 1,500 matched episodes per target. ProtoNet has no trained episodic parameters.

| Shot | Target | ProtoNet | Support-only | WIPT |
|---:|---|---:|---:|---:|
| 1 | CUB | 93.26 | 93.58 | 93.47 |
| 1 | EuroSAT | 62.09 | 63.02 | 64.16 |
| 1 | ISIC | 28.40 | 28.01 | 28.18 |
| 5 | CUB | 97.88 | 97.81 | 97.79 |
| 5 | EuroSAT | 82.76 | 81.46 | 82.04 |
| 5 | ISIC | 38.92 | 37.39 | 38.38 |

WIPT's benefit depends on the domain and shot count. ProtoNet has the highest mean in all three 5-shot settings; WIPT's clearest gain over ProtoNet is on 1-shot EuroSAT. Uncertainty and paired comparisons are stored under `results/raw/training_seed_study/`.

![Controlled representation pathways](figures/architecture.png)

## Repository layout

| Directory | Contents |
|---|---|
| `configs/` | Shared settings and local data/checkpoint paths |
| `models/` | Frozen encoder, WIPT, controls and ablation models |
| `train/` | Individual head-training entry points |
| `eval/` | Selected-checkpoint evaluations and head benchmarking |
| `experiments/` | Replicated studies and decision-change analyses |
| `analysis/` | Result validation and summary building |
| `scripts/` | Dataset preparation, input checks and figure generation |
| `experimental/` | Optional meta-adaptation and alternative-source code, outside the reported study |
| `utils/` | Episode sampling, transforms, metrics and checkpoint loading |
| `results/` | Retained numerical records, including episode/query-level evidence |
| `figures/` | Regenerated PNG figures and derived statistics |
| `docs/` | Data, experiment and result-file guides |

## Regenerate figures without training

Use Python 3.10 or newer. From the repository root:

```bash
python -m pip install -r requirements-analysis.txt
python -m scripts.verify_repository --figures
```

This checks source syntax and retained result tables, then generates eight figures from CSV records. It does not load a model, download datasets, or train anything.

For a separate output directory or higher resolution:

```bash
python -m scripts.generate_figures --results results --out figures_export --dpi 600
```

The `--pdf` option also exports vector PDFs. Filenames describe figure contents rather than manuscript numbering, which can change when material moves to an appendix.

## Model environment

```bash
python -m venv .venv
# Activate .venv using the command for your shell.
python -m pip install -r requirements.txt
python -m scripts.check_models
```

Install a compatible PyTorch/torchvision build for your CPU or CUDA environment. Dependencies specify the packages used by the code; they are not a historical environment lock. The model checks use synthetic embeddings and a stub encoder, so they do not download pretrained weights or validate ViT feature extraction.

The first pretrained model construction may download the encoder weights. Full training and evaluation require local datasets; the CPU is supported, but substantial model runs are intended for a GPU.

## Datasets

The loaders expect class-folder datasets:

```text
data/
  miniimagenet_split/
    train/<class>/<image>
    val/<class>/<image>
    test/<class>/<image>
  CUB_200_2011/images/<class>/<image>
  EuroSAT/<class>/<image>
  ISIC/<class>/<image>
```

The miniImageNet partition sorts the 100 WordNet class identifiers: first 64 for training, next 16 for validation, last 20 for testing. This is a project-specific split, not the usual Ravi--Larochelle assignment.

For the flat miniImageNet image layout:

```bash
python -m scripts.prepare_miniimagenet --source data/miniimagenet --output data/miniimagenet_split
python -m scripts.check_dataset_layout
```

See [dataset preparation](docs/datasets.md) for counts, preprocessing and optional targets. Paths can be changed in `configs/experiment.py` or supplied through supported command-line arguments.

## Principal replicated study

Train WIPT and the support-only control with seeds 0--4:

```bash
python -m experiments.run_seed_study train --shot 1 --workers 6
python -m experiments.run_seed_study train --shot 5 --workers 6
```

Then evaluate on the matched target episodes:

```bash
python -m experiments.run_seed_study clean --shot 1
python -m experiments.run_seed_study clean --shot 5
```

Training uses Adam with learning rate `1e-4`, zero weight decay, cosine annealing to `1e-6`, at most 50 epochs, 200 training episodes and 50 validation episodes per epoch. The best source-validation checkpoint is retained. Training changes the episodic head only.

Checkpoints are written under `checkpoints/seed_study/<shot>shot/seed_<seed>/`. ProtoNet uses the same verified frozen encoder state as the trained heads.

Evaluation seeds are `42, 123, 2024, 7, 999`, with 300 episodes each. The principal confidence intervals summarize paired differences across five training seeds, conditional on this episode bank. Episode-level intervals are separate descriptive quantities.

## Other experiments

[Experiment commands](docs/experiments.md) cover:

- multi-query contexts: groups 1--5 in 5-shot and groups 1, 3, 5 in 1-shot;
- the 5-shot adaptation-by-scorer factorial comparison;
- rescue/break, boundary and shot/domain analyses;
- replicated EuroSAT corruptions;
- source-only shared-shift and exploratory mixed-shift training;
- selected-checkpoint architecture, scoring, geometry and support-noise diagnostics;
- synthetic-embedding latency and memory benchmarks.

`experimental/` contains optional code for different research settings. In particular, its meta-adaptation branch updates head parameters using target supports; it is not the frozen-parameter WIPT inference procedure and is not presented as a completed manuscript experiment.

## Results and checkpoints

The repository retains numerical results, including complete query-level boundary tables required by the plotting script. Restart partitions, small development-run outputs and byte-identical duplicates are omitted.

`results/paper/` contains the earlier selected-checkpoint tables used by supplementary diagnostics. It must not be confused with the principal five-training-seed comparison in `results/raw/training_seed_study/`. See [result records](docs/results.md).

Model weights and image datasets are excluded by `.gitignore`. No checkpoint download is provided. Retrain the heads or supply compatible checkpoints for model evaluations. The historical root-level checkpoint used for the support-noise diagnostic is not included; its retained numerical summary can still be plotted.

Evaluation commands normally write to `results/raw/`. Use their output-directory options for new runs if the retained records should remain unchanged. Generated figures can likewise be directed to a separate folder.

## Citation and permissions

Author and software metadata are provided in [CITATION.cff](CITATION.cff). No arXiv URL or publication DOI is asserted.

The existing copyright notice is retained in [LICENSE](LICENSE). This repository does not currently grant an open-source license.
