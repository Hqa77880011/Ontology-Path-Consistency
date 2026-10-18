# OPC

Implementation of **Endpoint-consistent continual learning under evolving label ontologies**.

## Installation

Use Python `>=3.11`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data preparation

The controlled protocol generator creates objects, historical leaf masks, stage IDs and two opposite ontology-edit paths. These are generated finite objects; train and evaluate both paths against the same prepared test split.

```bash
python -m opc.protocol --objects 300 --seed 17 --output data/protocol.npz
python -m opc prepare --source data/protocol.npz --output data/reproduce --seed 17
```

## Training

Use the same `configs/reproduce.json` throughout. Its seed is `17`; preserve the prepared splits and use separate output directories for new seeds.

```bash
python -m opc train --config configs/reproduce.json --data data/reproduce --path-id 0 --output runs/path_a
python -m opc train --config configs/reproduce.json --data data/reproduce --path-id 1 --output runs/path_b
```

## Evaluation

```bash
python -m opc evaluate --checkpoint runs/path_a/opc/model.pt --data data/reproduce/test.npz --output results/path_a
python -m opc evaluate --checkpoint runs/path_b/opc/model.pt --data data/reproduce/test.npz --output results/path_b
python -m opc.path_metrics --predictions results/path_a/predictions.npz results/path_b/predictions.npz --labels data/reproduce/test.npz --output results/path_metrics.json
```

## Outputs

Each training method directory contains `model.pt` and `history.jsonl`. Evaluation writes `predictions.npz` and `metrics.json`; plots read the supplied run directories.

```bash
python -m opc plot --inputs runs/path_a/opc results/path_a --output figures
```

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, torchvision 0.22.1, scipy 1.15.3, scikit-learn 1.6.1, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
