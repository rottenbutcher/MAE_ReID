#!/usr/bin/env python3
"""Generate sample-based classification splits for the Real_Ship dataset."""

import argparse
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def _scan_samples(data_root: Path, extension: str) -> Tuple[Sequence[str], List[Tuple[str, int]]]:
    """Return the sorted class folders and the list of sample paths with labels."""
    try:
        id_folders = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}") from exc

    id_to_label = {name: idx for idx, name in enumerate(id_folders)}

    samples: List[Tuple[str, int]] = []
    for id_name, label in id_to_label.items():
        folder_path = data_root / id_name
        files = list(folder_path.rglob(f"*{extension}"))
        for path in files:
            relative_path = path.relative_to(data_root).as_posix()
            samples.append((relative_path, label))

    return id_folders, samples


def _write_split_file(data_root: Path, filename: str, entries: Iterable[Tuple[str, int]]):
    output_path = data_root / filename
    with output_path.open("w", encoding="utf-8") as file:
        for rel_path, label in entries:
            file.write(f"{rel_path} {label}\n")
    print(f"Created split file: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create sample-based classification splits for the Real_Ship dataset",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to the Real_Ship dataset directory (e.g., ./data/Ship_Real_ext)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of samples for training (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Ratio of samples for validation (default: 0.1)",
    )
    parser.add_argument(
        "--extension",
        type=str,
        default=".npy",
        help="File extension to include when scanning each class folder (default: .npy)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when shuffling samples (default: 42)",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    print(f"Scanning data root: {data_root}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")

    id_folders, all_files = _scan_samples(data_root, args.extension)
    num_classes = len(id_folders)

    print(f"Found {num_classes} classes (IDs): {id_folders}")
    print(f"Found {len(all_files)} total samples.")

    if not all_files:
        raise RuntimeError(
            f"No files with extension '{args.extension}' were found under the provided data root.",
        )

    random.seed(args.seed)
    random.shuffle(all_files)

    total_count = len(all_files)
    train_count = int(total_count * args.train_ratio)
    val_count = int(total_count * args.val_ratio)

    if train_count + val_count >= total_count:
        raise ValueError(
            "The sum of train-ratio and val-ratio must be less than 1.0 to reserve samples for testing.",
        )

    train_files = all_files[:train_count]
    val_files = all_files[train_count : train_count + val_count]
    test_files = all_files[train_count + val_count :]

    print(
        "Splitting into Train: %d, Val: %d, Test: %d samples." % (
            len(train_files), len(val_files), len(test_files)
        )
    )

    _write_split_file(data_root, "real_ship_cls_train.txt", train_files)
    _write_split_file(data_root, "real_ship_cls_val.txt", val_files)
    _write_split_file(data_root, "real_ship_cls_test.txt", test_files)

    print("\nClassification splits created successfully.")
    print(f"Total classes in splits: {num_classes}")


if __name__ == "__main__":
    main()
