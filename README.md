# Continual Learning (Simplified: Class-Incremental, EWC/SI/LwF)

This is a **simplified** version of the original continual-learning codebase, trimmed down to:
- **Class-incremental learning** only (task boundaries are explicit).
- **Three methods only**: Elastic Weight Consolidation (EWC), Synaptic Intelligence (SI), Learning without Forgetting (LwF).

The code retains the core training loop, data handling, and evaluation pipeline while removing other scenarios, replay
variants, and generative models.

## Installation
```bash
pip install -r requirements.txt
```

## Quick start
Run a single class-incremental experiment:
```bash
./main.py --experiment=splitMNIST --scenario=class --ewc
```

Switch between methods:
- **EWC**: `--ewc`
- **SI**: `--si`
- **LwF**: `--lwf`

## Outputs
Results are stored under `store/`:
- `store/results` for accuracy and logs
- `store/models` for saved checkpoints
- `store/plots` for PDF plots (if `--pdf` is used)

## Notes
- Only class-incremental learning is supported in this simplified project.
- The remaining code is intended for clarity and educational purposes.
