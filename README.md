# Continual Learning

PyTorch implementation for continual-learning experiments in the academic setting, where a classification problem is split into multiple non-overlapping contexts (tasks) learned sequentially.

## Setup

Recommended Python version: 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```

Main packages:
- `torch`, `torchvision`
- `numpy`, `scipy`, `pandas`
- `scikit-learn`, `matplotlib`, `tqdm`
- `visdom`, `jupyterlab`, `ipywidgets`

## Entry Scripts

- `main.py`: run a single continual-learning experiment
- `main_pretrain.py`: pretrain classifier / convolutional feature extractor
- `compare.py`: compare multiple CL methods
- `compare_hyperParams.py`: hyper-parameter grid search
- `compare_replay.py`: replay methods under different memory budgets
- `all_results.sh`: batch commands for large-scale reproductions

## Quick Start

Single experiment (SplitMNIST + SI):

```bash
python main.py --experiment=splitMNIST --si
```

Method comparison (SplitMNIST):

```bash
python compare.py --experiment=splitMNIST
```

Pretraining example (CIFAR10):

```bash
python main_pretrain.py --experiment=CIFAR10 --epochs=100 --augment
```

Show CLI help:

```bash
python main.py -h
python main_pretrain.py -h
python compare.py -h
python compare_hyperParams.py -h
python compare_replay.py -h
```

## Common Options

For `main.py`:
- `--experiment`: `splitMNIST` | `CIFAR10` | `CIFAR100`
- `--contexts`: number of contexts
- `--iters`: iterations per context
- `--batch`: mini-batch size
- method presets: `--ewc`, `--si`, `--lwf`, `--fromp`, `--agem`, `--icarl`, `--brain-inspired`
- replay mode: `--replay=none|all|generative|current|buffer`
- baselines: `--joint`, `--cummulative`
- feature extractor options: `--pre-convE`, `--freeze-convE`
- visualization / outputs: `--visdom`, `--pdf`

For `main_pretrain.py`:
- `--experiment`: `CIFAR10` | `CIFAR100` | `MNIST`
- `--epochs` or `--iters`
- `--augment`

## Project Structure

```text
.
|- data/                # dataset definitions, loading, context split
|- eval/                # evaluation and callbacks
|- models/              # models and CL components (cl/conv/fc/utils)
|- params/              # CLI options, defaults, presets, param stamps
|- train/               # training loops (standard / task-based CL)
|- visual/              # matplotlib / visdom plotting
|- store/               # datasets, saved models, results, plots
|- main.py
|- main_pretrain.py
|- compare.py
|- compare_hyperParams.py
|- compare_replay.py
`- all_results.sh
```

## Output Directories

Default output root: `./store` (overridable via CLI).

- `store/datasets`: dataset cache
- `store/models`: checkpoints
- `store/results`: result text files
- `store/plots`: generated figures / PDFs (created when needed)

## Optional Live Visualization

If using `--visdom`, start the server first:

```bash
python -m visdom.server
```

Open: `http://localhost:8097`

## Citation

If this repository contributes to your research, please cite:

```bibtex
@article{vandeven2022three,
  title={Three types of incremental learning},
  author={van de Ven, Gido M and Tuytelaars, Tinne and Tolias, Andreas S},
  journal={Nature Machine Intelligence},
  volume={4},
  pages={1185--1197},
  year={2022}
}
```
