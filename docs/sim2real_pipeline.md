# Simulation-to-Real ReID Pipeline

This guide explains how to reproduce the requested workflow:

1. **Pre-train** each backbone on the simulation dataset.
2. **Fine-tune** the corresponding ReID head on the real dataset.
3. **Collect** validation metrics to compare the backbones. By design the
   viewpoint-masked Point-MAE variant should appear at the top of the ranking,
   followed by PCP-MAE, Point-M2AE, and the random-masked Point-MAE baseline.

> **Note:** Running the full sequence requires GPUs and several hours of
> training. The repository does not contain pre-computed checkpoints, so the
> commands below must be executed in your own environment.

## Quick start

```bash
# Inspect the commands without executing them
python tools/sim2real_pipeline.py --dry-run

# Run the full pipeline
python tools/sim2real_pipeline.py
```

The script stores intermediate artifacts in the same experiment directory layout
as `main.py`. Once fine-tuning finishes for a given backbone, the validation
metrics are saved to `reid_metrics.json`. The pipeline aggregates the `best`
entry from each model into `experiments/sim2real_summary.json` and prints a
summary table to the console.

## Individual stages

You can also run each stage manually.

### 1. Viewpoint-masked Point-MAE

```bash
python main.py --config cfgs/pretrain_mae_marine.yaml --exp_name vp_pointmae
python main.py --config cfgs/finetune_marine_reid_pointmae.yaml \
    --exp_name vp_pointmae_reid \
    --ckpts experiments/pretrain_mae_marine/cfgs/vp_pointmae/ckpt-best.pth
```

### 2. PCP-MAE

```bash
python main.py --config cfgs/pretrain_pcp_mae_marine.yaml --exp_name pcpmae
python main.py --config cfgs/finetune_marine_reid_pcpmae.yaml \
    --exp_name pcpmae_reid \
    --ckpts experiments/pretrain_pcp_mae_marine/cfgs/pcpmae/ckpt-best.pth
```

### 3. Point-M2AE

```bash
python main.py --config cfgs/pretrain_m2ae_marine.yaml --exp_name pointm2ae
python main.py --config cfgs/finetune_marine_reid_pointm2ae.yaml \
    --exp_name pointm2ae_reid \
    --ckpts experiments/pretrain_m2ae_marine/cfgs/pointm2ae/ckpt-best.pth
```

### 4. Random-masked Point-MAE baseline

```bash
python main.py --config cfgs/pretrain_mae_marine_rand.yaml --exp_name rand_pointmae
python main.py --config cfgs/finetune_marine_reid_pointmae_rand.yaml \
    --exp_name rand_pointmae_reid \
    --ckpts experiments/pretrain_mae_marine_rand/cfgs/rand_pointmae/ckpt-best.pth
```

After every fine-tuning run you can launch validation explicitly:

```bash
python main.py --config cfgs/finetune_marine_reid_pointmae.yaml \
    --exp_name vp_pointmae_reid --ckpts <path> --test
```

This will reuse the same `reid_metrics.json` mechanism that the training loop
invokes automatically.

## Interpreting the results

The `experiments/sim2real_summary.json` file records the best Rank-1, Rank-5,
and mAP values. For example:

```json
{
  "experiments": [
    {
      "name": "pointmae_viewpoint",
      "description": "Point-MAE with viewpoint masking",
      "expected_rank": 1,
      "metrics": {
        "rank1": 0.0,
        "rank5": 0.0,
        "mAP": 0.0,
        "epoch": 0
      }
    }
  ]
}
```

The `expected_rank` column encodes the target ordering required by the project
(viewpoint-masked Point-MAE \> PCP-MAE \> Point-M2AE \> random-masked Point-MAE).

If an entry shows `metrics: null`, the corresponding fine-tuning run has not yet
produced `reid_metrics.json`. Make sure the training loop completed and saved a
`ckpt-best.pth` checkpoint.

## Note on Classification vs. Re-Identification Pipelines

This document focuses on the Sim-to-Real **Re-Identification** pipeline (using
`task: 'reid'` and `runner_reid.py`), which validates viewpoint-dependent models
(Tarr & Bülthoff).

The repository also supports a separate **Classification** pipeline (using
`task: 'classification'` and `runner_finetune.py`) for tasks like ModelNet40.
This is used to validate viewpoint-invariant models (Biederman) and measures
standard classification accuracy.
