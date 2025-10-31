#!/usr/bin/env python3
"""Evaluate scratch baselines on the RealShip dataset."""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

PYTHON = sys.executable or "python"


@dataclass
class EvalTarget:
    """Description of a scratch baseline to evaluate."""

    key: str
    config: Path
    default_exp: str
    description: str


DEFAULT_TARGETS: Dict[str, EvalTarget] = {
    "dgcnn": EvalTarget(
        key="dgcnn",
        config=Path("cfgs/pretrain_dgcnn_marine.yaml"),
        default_exp="dgcnn_from_scratch",
        description="DGCNN ReID baseline",
    ),
    "pointnext": EvalTarget(
        key="pointnext",
        config=Path("cfgs/pretrain_pointnext_marine.yaml"),
        default_exp="pointnext_from_scratch",
        description="PointNeXt ReID baseline",
    ),
    "pointtransformer": EvalTarget(
        key="pointtransformer",
        config=Path("cfgs/pretrain_supervised_marine.yaml"),
        default_exp="pointtransformer_from_scratch",
        description="PointTransformer supervised baseline",
    ),
}


def load_config(path: Path) -> Dict:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def inject_real_test(config: Dict, subset: str, batch_size: int, npoints: Optional[int]) -> Dict:
    cfg = dict(config)
    dataset = cfg.setdefault("dataset", {})

    if npoints is None:
        npoints = cfg.get("npoints")
        if npoints is None:
            train_cfg = dataset.get("train", {}).get("others", {})
            npoints = train_cfg.get("npoints", 8192)

    dataset["test"] = {
        "_base_": "cfgs/dataset_configs/Real_Ship.yaml",
        "others": {
            "subset": subset,
            "npoints": int(npoints),
            "bs": int(batch_size),
        },
    }
    return cfg


def write_temp_config(original_cfg: Path, payload: Dict) -> Path:
    temp_dir = tempfile.TemporaryDirectory(prefix=f"eval_{original_cfg.stem}_", dir=None)
    temp_path = Path(temp_dir.name) / original_cfg.name
    temp_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    temp_path._cleanup = temp_dir  # type: ignore[attr-defined]
    return temp_path


def build_command(cfg_path: Path, ckpt: Path, exp_name: str, num_workers: int) -> List[str]:
    return [
        PYTHON,
        "main.py",
        "--config",
        str(cfg_path),
        "--exp_name",
        exp_name,
        "--ckpts",
        str(ckpt),
        "--test",
        "--num_workers",
        str(num_workers),
    ]


def evaluate_target(target: EvalTarget, ckpt: Path, subset: str, batch_size: int, num_workers: int) -> int:
    if not target.config.exists():
        raise FileNotFoundError(f"Missing config for {target.description}: {target.config}")
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Missing checkpoint for {target.description}: {ckpt}.\n"
            f"Provide an explicit --{target.key}-ckpt argument or train the model first."
        )

    payload = inject_real_test(load_config(target.config), subset=subset, batch_size=batch_size, npoints=None)
    temp_cfg = write_temp_config(target.config, payload)
    exp_name = f"{target.key}_real_eval"
    cmd = build_command(temp_cfg, ckpt, exp_name, num_workers=num_workers)

    print(f"\n[evaluate] {target.description}")
    print("[evaluate] Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[evaluate] Evaluation failed for {target.description} (exit code {result.returncode}).")
    else:
        print(f"[evaluate] Finished {target.description}.")
    return result.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate scratch baselines on real data")
    parser.add_argument("--subset", default="reid_test", help="RealShip subset txt to evaluate (e.g., reid_val, reid_test)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for RealShip evaluation")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader worker count for evaluation")
    parser.add_argument("--dgcnn-ckpt", type=Path, default=None, help="Path to DGCNN checkpoint (defaults to experiments/...)")
    parser.add_argument("--pointnext-ckpt", type=Path, default=None, help="Path to PointNeXt checkpoint")
    parser.add_argument("--pointtransformer-ckpt", type=Path, default=None, help="Path to PointTransformer checkpoint")
    parser.add_argument("--dgcnn-exp", default=DEFAULT_TARGETS["dgcnn"].default_exp, help="Experiment name used for DGCNN train")
    parser.add_argument(
        "--pointnext-exp",
        default=DEFAULT_TARGETS["pointnext"].default_exp,
        help="Experiment name used for PointNeXt training",
    )
    parser.add_argument(
        "--pointtransformer-exp",
        default=DEFAULT_TARGETS["pointtransformer"].default_exp,
        help="Experiment name used for PointTransformer training",
    )
    return parser.parse_args()


def resolve_checkpoint(target: EvalTarget, explicit: Optional[Path], exp_name: str) -> Path:
    if explicit is not None:
        return explicit
    return Path("experiments") / target.config.stem / target.config.parent.stem / exp_name / "ckpt-best.pth"


def main() -> None:
    args = parse_args()

    ckpt_map = {
        "dgcnn": resolve_checkpoint(DEFAULT_TARGETS["dgcnn"], args.dgcnn_ckpt, args.dgcnn_exp),
        "pointnext": resolve_checkpoint(DEFAULT_TARGETS["pointnext"], args.pointnext_ckpt, args.pointnext_exp),
        "pointtransformer": resolve_checkpoint(
            DEFAULT_TARGETS["pointtransformer"], args.pointtransformer_ckpt, args.pointtransformer_exp
        ),
    }

    failures = 0
    for key, target in DEFAULT_TARGETS.items():
        rc = evaluate_target(
            target,
            ckpt=ckpt_map[key],
            subset=args.subset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        if rc != 0:
            failures += 1

    if failures:
        raise SystemExit(f"{failures} evaluation(s) failed. See logs above for details.")


if __name__ == "__main__":
    main()
