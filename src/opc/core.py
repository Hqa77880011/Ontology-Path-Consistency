import numpy as np
import torch
from torch import nn
from torch.func import functional_call

from .vision import classification_metrics


EVALUATION_NEEDS_GRAD = True

METHODS = ["opc", "leaf_only", "fine_tune", "no_replay"]


class Model(nn.Module):
    def __init__(self, cfg, method):
        super().__init__()
        self.method = method
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cfg["input_dim"], cfg["hidden"]),
            nn.GELU(),
            nn.Linear(cfg["hidden"], cfg["leaves"]),
        )

    def forward(self, x):
        return self.network(x.float())


def marginal(logits, allowed):
    allowed = allowed.bool()
    if not allowed.any(-1).all():
        raise ValueError("Every ontology label must have compatible leaves")
    log_probability = logits.log_softmax(-1)
    return -torch.logsumexp(log_probability.masked_fill(~allowed, -torch.inf), -1).mean()


def virtual(model, parameters, x, allowed, step):
    value = marginal(functional_call(model, parameters, (x,)), allowed)
    gradients = torch.autograd.grad(value, tuple(parameters.values()), create_graph=True)
    return {
        key: parameter - step * gradient
        for (key, parameter), gradient in zip(parameters.items(), gradients)
    }


def path_defect(model, batch, cfg):
    replay = batch["replay"].bool()
    current = ~replay
    if not replay.any() or not current.any():
        return model(batch["x"]).sum() * 0
    parameters = dict(model.named_parameters())
    x, allowed = batch["x"], batch["leaf_mask"]
    a = virtual(model, parameters, x[current], allowed[current], cfg["virtual_step"])
    ab = virtual(model, a, x[replay], allowed[replay], cfg["virtual_step"])
    b = virtual(model, parameters, x[replay], allowed[replay], cfg["virtual_step"])
    ba = virtual(model, b, x[current], allowed[current], cfg["virtual_step"])
    mapping = cfg["endpoint_map"]
    endpoint = torch.eye(cfg["leaves"], device=x.device) if mapping == "identity" else torch.as_tensor(mapping, device=x.device, dtype=torch.float32)
    p = functional_call(model, ab, (x,)).softmax(-1) @ endpoint.T
    q = functional_call(model, ba, (x,)).softmax(-1) @ endpoint.T
    mean = (p + q) / 2
    return (
        0.5
        * (
            (p * (p.clamp_min(1e-8).log() - mean.clamp_min(1e-8).log())).sum(-1)
            + (q * (q.clamp_min(1e-8).log() - mean.clamp_min(1e-8).log())).sum(-1)
        ).mean()
    )


def loss(model, batch, cfg):
    if model.method in ("fine_tune", "no_replay"):
        active = ~batch["replay"].bool()
        if not active.any():
            return model(batch["x"]).sum() * 0
        return marginal(model(batch["x"][active]), batch["leaf_mask"][active])
    result = marginal(model(batch["x"]), batch["leaf_mask"])
    if model.method == "opc" and model.training:
        result = result + cfg["path_weight"] * path_defect(model, batch, cfg)
    return result


def predict(model, batch, cfg, calibration=None):
    defect = path_defect(model, batch, cfg)
    available = batch["replay"].bool().any() & ~batch["replay"].bool().all()
    return {"logits": model(batch["x"]), "pcd": defect.expand(len(batch["x"]), 1),
            "pcd_available": available.expand(len(batch["x"]), 1)}


def metrics(prediction, data, cfg):
    result = classification_metrics(prediction["logits"], data["y"])
    available = prediction["pcd_available"].ravel().astype(bool)
    result["pcd"] = float(prediction["pcd"].ravel()[available].mean()) if available.any() else None
    if "path" in data:
        values = []
        for path in np.unique(data["path"]):
            mask = data["path"] == path
            values.append(
                classification_metrics(prediction["logits"][mask], data["y"][mask])["accuracy"]
            )
        result["worst_path_accuracy"] = float(min(values))
        result["mean_path_accuracy"] = float(np.mean(values))
    return result


def inference(model, batch, cfg):
    return model(batch["x"])


def fit(model, train, validation, cfg, output):
    import json
    import time
    from pathlib import Path

    device = next(model.parameters()).device
    rng = np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if "stage" not in train.data:
        raise ValueError(
            "Ontology training requires a stage array defining the registered edit sequence"
        )
    source = train.data
    selected = np.ones(len(train), dtype=bool)
    if "path" in source:
        selected &= source["path"] == cfg["path_id"]
    if not selected.any():
        raise ValueError("No examples belong to the registered training path")
    stages = sorted(np.unique(source["stage"][selected]).tolist())
    memory = []
    seen = 0
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    for stage in stages:
        current = np.flatnonzero(selected & (source["stage"] == stage))
        for epoch in range(cfg["epochs_per_stage"]):
            model.train()
            start = time.perf_counter()
            order = rng.permutation(current)
            total, count = 0.0, 0
            for offset in range(0, len(order), cfg["batch_size"]):
                active = order[offset : offset + cfg["batch_size"]]
                replay = (
                    rng.choice(memory, min(len(memory), len(active)), replace=False)
                    if memory
                    else np.array([], dtype=int)
                )
                indices = np.r_[active, replay].astype(int)
                batch = {
                    key: torch.as_tensor(value[indices], device=device)
                    for key, value in source.items()
                }
                batch["replay"] = torch.as_tensor(
                    np.r_[np.zeros(len(active)), np.ones(len(replay))], device=device
                )
                optimizer.zero_grad()
                value = loss(model, batch, cfg)
                value.backward()
                optimizer.step()
                total += float(value.detach()) * len(indices)
                count += len(indices)
            model.eval()
            valid_mask = np.ones(len(validation), dtype=bool)
            if "stage" in validation.data:
                valid_mask &= validation.data["stage"] == stage
            if "path" in validation.data:
                valid_mask &= validation.data["path"] == cfg["path_id"]
            if not valid_mask.any():
                raise ValueError("Every trained ontology stage needs a matching validation split")
            with torch.no_grad():
                vx = torch.as_tensor(validation.data["x"][valid_mask], device=device)
                vm = torch.as_tensor(validation.data["leaf_mask"][valid_mask], device=device)
                validation_loss = float(marginal(model(vx), vm))
            history.append(
                {
                    "epoch": len(history) + 1,
                    "stage": int(stage),
                    "path": cfg["path_id"],
                    "train_loss": total / count,
                    "validation_loss": validation_loss,
                    "seconds": time.perf_counter() - start,
                    "replay_examples": len(memory),
                }
            )
        for index in current:
            seen += 1
            if len(memory) < cfg["replay_size"]:
                memory.append(int(index))
            else:
                replacement = int(rng.integers(seen))
                if replacement < cfg["replay_size"]:
                    memory[replacement] = int(index)
        torch.save(
            {"state": model.state_dict(), "config": cfg, "method": model.method},
            output / f"stage_{stage:04d}.pt",
        )
    torch.save(
        {"state": model.state_dict(), "config": cfg, "method": model.method}, output / "model.pt"
    )
    (output / "history.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in history), encoding="utf-8"
    )
