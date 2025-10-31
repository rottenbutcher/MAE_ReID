"""Utility to generate split list text files for RealShip datasets.

This script scans a subset directory (e.g., ``reid_val``) under the real-world
point cloud root and writes a text file listing every ``.npy`` file relative to
that root. The resulting text file is compatible with :class:`RealShip` and the
``SimRealValidation`` helper.

Example usage:

.. code-block:: bash

    python tools/create_split_txt.py \
        --dataset-root data/Ship_Real_ext \
        --subset reid_val

This will produce ``data/Ship_Real_ext/reid_val.txt`` with entries such as::

    reid_val/FP_target1/sample.npy
    reid_val/FP_target2/sample.npy

Use ``--dataset-root`` and ``--subset`` to match your local directory layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


def collect_files(directory: Path, extension: str) -> Iterable[Path]:
    """Yield files that match ``extension`` from ``directory`` recursively."""
    if not extension.startswith('.'):
        extension = '.' + extension
    for path in directory.rglob('*'):
        if path.is_file() and path.suffix == extension:
            yield path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate RealShip split text files.')
    parser.add_argument(
        '--dataset-root',
        required=True,
        type=Path,
        help='Path to the real-world dataset root (e.g., data/Ship_Real_ext).',
    )
    parser.add_argument(
        '--subset',
        required=True,
        type=str,
        help='Name of the subset folder to index (e.g., reid_train, reid_val).',
    )
    parser.add_argument(
        '--extension',
        default='.npy',
        type=str,
        help='File extension to include in the split list (default: .npy).',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Optional custom output path. Defaults to <dataset-root>/<subset>.txt.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    subset_dir = dataset_root / args.subset

    if not dataset_root.is_dir():
        raise FileNotFoundError(f'Dataset root does not exist: {dataset_root}')
    if not subset_dir.is_dir():
        raise FileNotFoundError(f'Subset directory does not exist: {subset_dir}')

    files = sorted(collect_files(subset_dir, args.extension))
    if not files:
        raise RuntimeError(
            f'No files with extension {args.extension!r} found under {subset_dir}'
        )

    output_path = args.output
    if output_path is None:
        output_path = dataset_root / f'{args.subset}.txt'
    else:
        output_path = output_path.expanduser().resolve()
        output_parent = output_path.parent
        if output_parent and not output_parent.exists():
            output_parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for path in files:
            rel_path = path.relative_to(dataset_root).as_posix()
            f.write(rel_path + '\n')

    print(f'Wrote {len(files)} entries to {output_path}')


if __name__ == '__main__':
    main()
