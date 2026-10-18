"""Register two legal split orders with a common leaf endpoint."""

import argparse
from pathlib import Path

import numpy as np


def register(values, blocks):
    """Expand training/validation objects, retaining immutable historical labels."""
    labels = np.asarray(values["y"], dtype=int)
    leaves = sum(map(len, blocks))
    if sorted(sum(blocks, [])) != list(range(leaves)) or len(blocks) < 2:
        raise ValueError("Coarse blocks must partition consecutive leaf IDs")
    if (labels < 0).any() or (labels >= leaves).any():
        raise ValueError("Fine labels must belong to the registered leaf vocabulary")
    n = len(labels)
    if any(len(value) != n for value in values.values()):
        raise ValueError("Every source array must share the sample axis")
    groups, split = np.asarray(values["group"]), np.asarray(values["split"])
    for group in np.unique(groups):
        if len(np.unique(split[groups == group])) != 1:
            raise ValueError("A source object crosses registered splits")
    if set(split.tolist()) != {0, 1, 2}:
        raise ValueError("Register nonempty train, validation and test splits")
    rows = []
    orders = [list(range(len(blocks))), list(reversed(range(len(blocks))))]
    for path, order in enumerate(orders):
        for stage in range(len(blocks) + 1):
            indices = np.flatnonzero(split != 2)
            membership = np.zeros((len(indices), leaves), dtype=bool)
            refined = set(order[:stage])
            for row, index in enumerate(indices):
                label = labels[index]
                block = next(i for i, members in enumerate(blocks) if label in members)
                membership[row, [label] if block in refined else blocks[block]] = True
            rows.append(dict(
                **{key: np.asarray(value)[indices] for key, value in values.items()},
                leaf_mask=membership, stage=np.full(len(indices), stage),
                path=np.full(len(indices), path), replay=np.zeros(len(indices)),
            ))
    indices = np.flatnonzero(split == 2)
    rows.append(dict(
        **{key: np.asarray(value)[indices] for key, value in values.items()},
        leaf_mask=np.eye(leaves, dtype=bool)[labels[indices]],
        stage=np.full(len(indices), len(blocks)), path=np.full(len(indices), -1),
        replay=np.zeros(len(indices)),
    ))
    return {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}


def controlled(objects=300, seed=17):
    """Generate a finite six-leaf feature task with independent object splits."""
    if objects < 30:
        raise ValueError("Use at least 30 objects to populate all registered splits")
    rng = np.random.default_rng(seed)
    labels = np.arange(objects) % 6
    centers = rng.normal(size=(6, 8)).astype(np.float32)
    x = centers[labels] + 0.3 * rng.normal(size=(objects, 8)).astype(np.float32)
    split = np.full(objects, 2, dtype=np.int64)
    split[: int(0.6 * objects)] = 0
    split[int(0.6 * objects):int(0.8 * objects)] = 1
    return register(dict(x=x, y=labels, group=np.arange(objects), split=split),
                    [[0, 1, 2], [3, 4, 5]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="Optional NPZ x/y/group/split source")
    parser.add_argument("--blocks", help="JSON list of coarse leaf-ID blocks for --source")
    parser.add_argument("--objects", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.source:
        import json
        if not args.blocks:
            parser.error("--source requires --blocks")
        with np.load(args.source, allow_pickle=False) as source:
            values = {key: source[key] for key in ("x", "y", "group", "split")}
        arrays = register(values, json.loads(Path(args.blocks).read_text(encoding="utf-8")))
    else:
        arrays = controlled(args.objects, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(str(output))
    np.savez_compressed(output, **arrays)


if __name__ == "__main__":
    main()
