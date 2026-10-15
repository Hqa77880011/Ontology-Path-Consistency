import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax


def ece(probability, labels):
    confidence = probability.max(-1)
    correct = probability.argmax(-1) == labels
    value = 0.0
    for low in np.linspace(0, 1, 16)[:-1]:
        mask = (confidence > low) & (confidence <= low + 1 / 15)
        if mask.any():
            value += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    labels = np.load(args.labels, allow_pickle=False)["y"]
    probabilities = np.stack(
        [softmax(np.load(path, allow_pickle=False)["logits"], axis=-1) for path in args.predictions]
    )
    if probabilities.shape[1] != len(labels):
        raise ValueError("Every path must predict the same ordered endpoint test set")
    accuracies = np.mean(probabilities.argmax(-1) == labels[None], axis=1)
    ec = [ece(p, labels) for p in probabilities]
    average = probabilities.mean(0)
    js = np.mean(
        np.sum(
            probabilities
            * (np.log(np.maximum(probabilities, 1e-12)) - np.log(np.maximum(average[None], 1e-12))),
            axis=-1,
        )
    )
    result = {
        "mean_path_accuracy": float(accuracies.mean()),
        "worst_path_accuracy": float(accuracies.min()),
        "predictive_opv": float(probabilities.var(0).sum(-1).mean()),
        "cross_path_js": float(js),
        "ece_spread": float(max(ec) - min(ec)),
        "path_ece": ec,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
