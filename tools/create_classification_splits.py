#!/usr/bin/env python3
"""Generate sample-based classification splits for the Real_Ship dataset."""

import argparse
import glob
import os
import random
from typing import Iterable, List, Sequence, Tuple


def _scan_samples(data_root: str) -> Tuple[Sequence[str], List[Tuple[str, int]]]:
    """Return the sorted class folders and the list of sample paths with labels."""
    id_folders = sorted(
        [
            d
            for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))
        ]
    )

    id_to_label = {id_name: idx for idx, id_name in enumerate(id_folders)}

    samples: List[Tuple[str, int]] = []
    for id_name in id_folders:
        label = id_to_label[id_name]
        folder_path = os.path.join(data_root, id_name)

        files = glob.glob(os.path.join(folder_path, "*.npy"))
        if not files:
            files = glob.glob(os.path.join(folder_path, "*.npz"))

        for path in files:
            relative_path = os.path.relpath(path, data_root)
            samples.append((relative_path, label))

    return id_folders, samples


def _write_split_file(data_root: str, filename: str, entries: Iterable[Tuple[str, int]]):
    output_path = os.path.join(data_root, filename)
    with open(output_path, "w", encoding="utf-8") as file:
        for rel_path, label in entries:
            file.write(f"{rel_path} {label}\n")
    print(f"Created split file: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create sample-based classification splits for the Real_Ship dataset"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to the Real_Ship dataset directory (e.g., ./data/Real_Ship)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of samples for training (default: 0.8)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Ratio of samples for validation (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when shuffling samples (default: 42)",
    )

    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    print(f"Scanning data root: {data_root}")

    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    id_folders, all_files = _scan_samples(data_root)
    num_classes = len(id_folders)

    print(f"Found {num_classes} classes (IDs).")
    print(f"Found {len(all_files)} total samples.")

    if not all_files:
        raise RuntimeError("No .npy or .npz files were found under the provided data root.")

    random.seed(args.seed)
    random.shuffle(all_files)

    total_count = len(all_files)
    train_count = int(total_count * args.train_ratio)
    val_count = int(total_count * args.val_ratio)

    if train_count + val_count >= total_count:
        raise ValueError(
            "The sum of train_ratio and val_ratio must be less than 1.0 to reserve samples for testing."
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

    print("Classification splits created successfully.")


if __name__ == "__main__":
    main()
