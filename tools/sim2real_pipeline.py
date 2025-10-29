#!/usr/bin/env python3
"""Utility to pretrain on simulation data and fine-tune ReID on real data.

The script orchestrates the following steps for each experiment definition:
1. Run self-supervised pre-training on the simulation dataset with a chosen backbone.
2. Fine-tune the corresponding ReID head on the real-world dataset while loading the
   pre-trained checkpoint.
3. Collect the best validation metrics recorded by ``tools.runner_reid`` in
   ``reid_metrics.json`` and export a consolidated summary.

All commands are executed sequentially which makes it easy to inspect failures. Use
``--dry-run`` to simply print the commands without executing them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

PYTHON = sys.executable or "python"


@dataclass
class Experiment:
    """Container describing a simulation-to-real transfer experiment."""

    name: str
    pretrain_config: Path
    reid_config: Path
    description: str
    expected_order: int

    def pretrain_experiment_path(self, exp_name: str) -> Path:
        return Path("experiments") / self.pretrain_config.stem / self.pretrain_config.parent.stem / exp_name

    def reid_experiment_path(self, exp_name: str) -> Path:
        return Path("experiments") / self.reid_config.stem / self.reid_config.parent.stem / exp_name


DEFAULT_EXPERIMENTS: List[Experiment] = [
    Experiment(
        name="pointmae_viewpoint",
        pretrain_config=Path("cfgs/pretrain_mae_marine.yaml"),
        reid_config=Path("cfgs/finetune_marine_reid_pointmae.yaml"),
        description="Point-MAE with viewpoint masking",
        expected_order=1,
    ),
    Experiment(
        name="pcp_mae",
        pretrain_config=Path("cfgs/pretrain_pcp_mae_marine.yaml"),
        reid_config=Path("cfgs/finetune_marine_reid_pcpmae.yaml"),
        description="PCP-MAE with center prediction",
        expected_order=2,
    ),
    Experiment(
        name="pointm2ae",
        pretrain_config=Path("cfgs/pretrain_m2ae_marine.yaml"),
        reid_config=Path("cfgs/finetune_marine_reid_pointm2ae.yaml"),
        description="Point-M2AE hierarchical encoder",
        expected_order=3,
    ),
    Experiment(
        name="pointmae_rand",
        pretrain_config=Path("cfgs/pretrain_mae_marine_rand.yaml"),
        reid_config=Path("cfgs/finetune_marine_reid_pointmae_rand.yaml"),
        description="Point-MAE with random masking baseline",
        expected_order=4,
    ),
]


def build_command(config: Path, exp_name: str, extra: Optional[List[str]] = None) -> List[str]:
    cmd = [PYTHON, "main.py", "--config", str(config), "--exp_name", exp_name]
    if extra:
        cmd.extend(extra)
    return cmd


def run_command(cmd: List[str], dry_run: bool) -> int:
    print("[sim2real] ``%s``" % " ".join(cmd))
    if dry_run:
        return 0
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def load_best_metrics(exp_path: Path) -> Optional[Dict[str, float]]:
    metrics_file = exp_path / "reid_metrics.json"
    if not metrics_file.exists():
        return None
    try:
        payload = json.loads(metrics_file.read_text())
    except json.JSONDecodeError:
        return None
    best = payload.get("best")
    if not best:
        return None
    return {
        "rank1": float(best.get("rank1", 0.0)),
        "rank5": float(best.get("rank5", 0.0)),
        "mAP": float(best.get("mAP", 0.0)),
        "epoch": int(best.get("epoch", -1)),
    }


def orchestrate(args: argparse.Namespace) -> None:
    results: List[Dict[str, object]] = []

    for experiment in DEFAULT_EXPERIMENTS:
        print("\n[sim2real] ==== Running %s ====\n" % experiment.description)
        pretrain_exp = f"{args.pretrain_prefix}{experiment.name}"
        reid_exp = f"{args.reid_prefix}{experiment.name}"

        if not args.skip_pretrain:
            pretrain_cmd = build_command(experiment.pretrain_config, pretrain_exp)
            rc = run_command(pretrain_cmd, args.dry_run)
            if rc != 0:
                raise SystemExit(f"Pre-training failed for {experiment.name} (exit code {rc}).")
        else:
            print("[sim2real] Skipping pre-training as requested.")

        ckpt_path = experiment.pretrain_experiment_path(pretrain_exp) / "ckpt-best.pth"
        if not ckpt_path.exists() and not args.dry_run:
            raise FileNotFoundError(f"Missing checkpoint for {experiment.name}: {ckpt_path}")

        if not args.skip_reid:
            extra = ["--ckpts", str(ckpt_path)]
            if args.use_viewpoint_mask:
                extra.append("--use_viewpoint_mask")
            reid_cmd = build_command(experiment.reid_config, reid_exp, extra=extra)
            rc = run_command(reid_cmd, args.dry_run)
            if rc != 0:
                raise SystemExit(f"ReID fine-tuning failed for {experiment.name} (exit code {rc}).")
        else:
            print("[sim2real] Skipping ReID fine-tuning as requested.")

        metrics = load_best_metrics(experiment.reid_experiment_path(reid_exp))
        results.append(
            {
                "name": experiment.name,
                "description": experiment.description,
                "expected_rank": experiment.expected_order,
                "metrics": metrics,
            }
        )

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"experiments": results}, indent=2))
    print("\n[sim2real] Summary written to %s" % summary_path)

    print("\n[sim2real] === Ordered Results ===")
    for item in results:
        metrics = item["metrics"]
        if not metrics:
            print(f" - {item['description']}: metrics unavailable (run training to populate reid_metrics.json)")
        else:
            print(
                " - {desc}: Rank-1={rank1:.2f}, Rank-5={rank5:.2f}, mAP={mAP:.2f} (epoch {epoch})".format(
                    desc=item["description"],
                    rank1=metrics["rank1"],
                    rank5=metrics["rank5"],
                    mAP=metrics["mAP"],
                    epoch=metrics["epoch"],
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sim2Real evaluation harness")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    parser.add_argument("--skip-pretrain", action="store_true", help="Skip the pre-training stage")
    parser.add_argument("--skip-reid", action="store_true", help="Skip the ReID fine-tuning stage")
    parser.add_argument("--pretrain-prefix", default="sim2real_pre_", help="Prefix for pre-training experiment names")
    parser.add_argument("--reid-prefix", default="sim2real_reid_", help="Prefix for ReID experiment names")
    parser.add_argument(
        "--summary",
        default="experiments/sim2real_summary.json",
        help="Where to store the aggregated metrics JSON",
    )
    parser.add_argument(
        "--use-viewpoint-mask",
        action="store_true",
        help="Forward --use_viewpoint_mask to the ReID fine-tuning stage",
    )
    return parser.parse_args()


if __name__ == "__main__":
    orchestrate(parse_args())
