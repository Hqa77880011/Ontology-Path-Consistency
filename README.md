# Ontology-Path-Consistency

Implementation of **Endpoint-consistent continual learning under evolving label ontologies**.

Represents historical labels as compatible latent-leaf sets and penalizes JS discrepancy between current/replay virtual update orders at one shared endpoint.

## Install

Use Python 3.11 or newer. Install a matching PyTorch/TorchVision build for the target CPU or CUDA device, then install the package.

```bash
git clone https://github.com/Hqa77880011/Ontology-Path-Consistency.git
cd Ontology-Path-Consistency
python -m venv .venv
```

Activate with `source .venv/bin/activate` or `.venv\Scripts\Activate.ps1` in PowerShell.

```bash
python -m pip install -e .
```

## Prepare data

Use CIFAR-100 superclass/subclass ontologies or a registered finite label-edit sequence. Preserve parent image IDs across all ontology versions and legal edit paths.

`x` contains images/features, `y` contains evaluator fine-leaf labels, `leaf_mask` is [N,K] compatible-leaf membership, `replay` flags transported replay examples, and optional path identifies evaluation paths. Learner losses read leaf_mask, not fine y.

```bash
python -m opc prepare --source data/raw.npz --output data/prepared --seed 17
```

Use `group` to keep every derived view of one source object in the same partition. An explicit integer `split` array (0=train, 1=validation, 2=test) preserves a registered split. Otherwise groups are assigned 60/20/20. Every stored field must share the sample axis. Files are loaded without pickle.

## Train

Edit `configs/default.json` for the input dimensions, device and training budget.

```bash
python -m opc train --config configs/default.json --data data/prepared --output runs/main
python -m opc suite --config configs/default.json --data data/prepared --output runs/comparison
```

`--methods` selects a subset. opc, leaf_only, fine_tune and no_replay. Suite runs share architecture and sampled constraints.

The common driver uses AdamW with cosine learning-rate decay, validation-objective early stopping and a fixed seed. Custom federated/continual drivers use their registered round/task budgets. It saves only a validation-selected model and measured epoch history. `learning_rate`, `weight_decay`, `batch_size`, `epochs`, `patience` and `device` control optimization. Use separate output roots for independent runs.

## Evaluate and visualize

```bash
python -m opc evaluate --checkpoint runs/main/opc/model.pt --data data/prepared/test.npz --output results/opc
python -m opc plot --inputs runs/main/opc results/opc --output figures
```

`model.pt` contains model parameters, configuration and method identity. `history.jsonl` contains measured training/validation loss and epoch time. Evaluation writes `predictions.npz` and `metrics.json`. Plotting creates loss curves and metric comparisons from supplied run directories.

Leaf accuracy, macro F1/ECE, per-path mean/worst accuracy and local PCD. Cross-run path metrics require identical ordered test examples across the path-specific checkpoints.

## Implementation conventions

Virtual SGD steps use differentiable functional parameters and the same endpoint aggregation map in both orders. The local PCD compares two blocks; it does not assume all full ontology trajectories commute. The MLP backbone and virtual step are explicit configuration choices.


## Pack image records

A JSONL manifest can point to images, masks and NPY feature arrays. Paths are relative to the manifest. All rows use identical keys; group stores the source object ID. For example, an image-classification row is `{"x":"images/1.png","y":0,"group":"object-1"}`.

```bash
python -m opc.pack --manifest data/manifest.jsonl --output data/raw.npz --channels 3 --target class
```

## CIFAR-100 preparation

```bash
python -m opc.cifar --output data/raw.npz
```

This downloads the author dataset through TorchVision, reserves 10% of its training images for validation, and preserves the original test set. It writes fine/coarse labels, source IDs and task metadata. Paired/audited transformation fields and ontology constraints must be added from the registered pair/ontology protocol before using those objectives.

## Endpoint path comparison

```bash
python -m opc.path_metrics --predictions results/path_a/predictions.npz results/path_b/predictions.npz --labels data/prepared/test.npz --output results/path_metrics.json
```

Every prediction file must use the same ordered endpoint test set. This reports OPV, mean/worst path accuracy and cross-path ECE spread.

Inference timing runs the deployment forward pass separately from calibration, commutation and other evaluator diagnostics, after one warmup. Batch P50/P95 and examples/s retain the configured batch size. MAC/FLOP counts cover Conv/Linear/GRU multiply-adds; process RSS includes the dataset and Python runtime.

Ontology training requires stage IDs and optionally path IDs. The custom driver follows stages in order, retains a reservoir of immutable historical leaf-set constraints and validates only the current ontology stage. path_id chooses one registered edit path; repeat with matched data/budget for path comparison. epochs_per_stage and replay_size bound optimization and memory.
