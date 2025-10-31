# MARINE-MAE Re-Identification Benchmarks

This repository implements a family of masked auto-encoder (MAE) backbones and
ReID heads that operate on 3D point clouds. It is designed to showcase how
viewpoint-aware masking improves re-identification performance when transferring
from simulation to real-world ship datasets. The codebase contains:

* **Point-MAE** with configurable viewpoint or random masking.
* **PCP-MAE** (patch center prediction) and **Point-M2AE** (hierarchical MAE)
  variants for comparison.
* **Point-MAE CrossView**, which encodes a masked view and reconstructs a paired
  target view to mimic human cross-view perception.
* ReID heads that combine cross-entropy and batch-hard triplet objectives, plus
  an orchestration script that pre-trains on simulation data and fine-tunes on
  real scans.

The sections below describe how to install dependencies, prepare data, and run
pre-training, fine-tuning, and evaluation for each backbone.

---

## 1. Environment setup

1. Install system dependencies (Ubuntu example):

   ```bash
   sudo apt-get update
   sudo apt-get install -y python3-dev python3-pip build-essential
   ```

2. Install Python requirements:

   ```bash
   pip install -r requirements.txt
   ```

3. Build the custom CUDA extensions used by PointNet++ operators:

   ```bash
   python setup.py install
   ```

---

## 2. Dataset preparation

All configuration files assume the following directory layout relative to the
repository root:

```
MAE_ReID/
├── data/
│   ├── Ship_Simulation/    # synthetic training set (see cfgs/dataset_configs/Simulation_Ship.yaml)
│   └── Ship_Real_ext/      # real-world scans split into reid_train/val/test (see cfgs/dataset_configs/Real_Ship.yaml)
```

The directory names are case-sensitive on Linux. If your local folders are
named `ship_simulation` or `ship_real_ext`, update
`cfgs/dataset_configs/Simulation_Ship.yaml`,
`cfgs/dataset_configs/Real_Ship.yaml`, and `cfgs/dataset_configs/SimReal_Validation.yaml`
so that `DATA_PATH`, `SIM_DATA_PATH`, and `REAL_DATA_PATH` match your layout.

Make sure each subset contains point clouds stored in the formats expected by
`datasets/ShipDataset.py` (single point cloud per sample).

### Generating RealShip split lists

`RealShip` reads text files that list the relative path to every `.npy` file
belonging to a split. Entries should look like
`reid_val/FP_target4/scan.npy`, i.e. a path relative to `data/Ship_Real_ext`
that includes the subset directory and the class folder.

Use `tools/create_split_txt.py` to generate these lists automatically:

```bash
# Example for the validation split
python tools/create_split_txt.py \
    --dataset-root data/Ship_Real_ext \
    --subset reid_val

# Example for a training split stored under data/ship_real_ext/
# The --dataset-root flag can point to lowercase paths as long as the configs match.
python tools/create_split_txt.py \
    --dataset-root data/ship_real_ext \
    --subset reid_train \
    --output data/ship_real_ext/reid_train.txt
```

The script traverses `<dataset-root>/<subset>` recursively, writes one relative
path per line, and saves the result as `<dataset-root>/<subset>.txt` unless an
explicit `--output` is provided. `RealShip` automatically picks up files named
`<subset>.txt`, `real_ship_<subset>.txt`, or `reid_<subset>.txt` from either the
subset directory or its parent.

---

## 3. Pre-training on simulation data

Launch pre-training with `python main.py --config <CONFIG> --exp_name <NAME>`.
The table summarizes the available backbones:

| Model | Description | Config |
| --- | --- | --- |
| Point-MAE (viewpoint masking) | Uses the proposed view-based masking strategy. | `cfgs/pretrain_mae_marine.yaml` |
| Point-MAE (random masking baseline) | Drops patches uniformly at random. | `cfgs/pretrain_mae_marine_rand.yaml` |
| PCP-MAE | Predicts masked patch centers instead of full points. | `cfgs/pretrain_pcp_mae_marine.yaml` |
| Point-M2AE | Hierarchical masked auto-encoder. | `cfgs/pretrain_m2ae_marine.yaml` |
| Point-MAE CrossView | Encodes one masked view and reconstructs a paired view. | `cfgs/pretrain_mae_marine_crossview.yaml` |

### Training geometric baselines from scratch

For direct sim-to-real studies you can train three geometric backbones entirely
from the synthetic dataset while validating on a mixed set that includes a
handful of real-world ship IDs:

| Backbone | Config | Notes |
| --- | --- | --- |
| DGCNN ReID | `cfgs/pretrain_dgcnn_marine.yaml` | Uses `SimRealValidation` to mix simulation and real classes during validation. |
| PointNeXt ReID | `cfgs/pretrain_pointnext_marine.yaml` | Same mixed-validation protocol with a PointNeXt encoder head. |
| PointTransformer (supervised) | `cfgs/pretrain_supervised_marine.yaml` | Baseline PointTransformer trained from scratch. |

All three configurations keep the training split purely synthetic but evaluate
on the new `cfgs/dataset_configs/SimReal_Validation.yaml` recipe. By default the
validation loader samples every synthetic test object plus the first five real
classes (controlled via `real_max_classes`). You can explicitly list the desired
real IDs with `real_class_whitelist` or increase the number of classes by
editing the `others` section in each config. The helper dataset automatically
offsets real-world class labels beyond the simulation label range so that
validation metrics treat every synthetic and real identity as distinct.

Example commands for the scratch baselines:

```bash
# DGCNN baseline
python main.py --config cfgs/pretrain_dgcnn_marine.yaml --exp_name dgcnn_from_scratch

# PointNeXt baseline
python main.py --config cfgs/pretrain_pointnext_marine.yaml --exp_name pointnext_from_scratch

# PointTransformer baseline
python main.py --config cfgs/pretrain_supervised_marine.yaml --exp_name pointtransformer_from_scratch
```

Example commands:

```bash
# Viewpoint-masked Point-MAE (recommended)
python main.py --config cfgs/pretrain_mae_marine.yaml --exp_name vp_pointmae

# PCP-MAE
python main.py --config cfgs/pretrain_pcp_mae_marine.yaml --exp_name pcpmae

# Point-M2AE
python main.py --config cfgs/pretrain_m2ae_marine.yaml --exp_name pointm2ae

# Random masking baseline
python main.py --config cfgs/pretrain_mae_marine_rand.yaml --exp_name rand_pointmae

# Cross-view Point-MAE (produces paired-view reconstruction loss)
python main.py --config cfgs/pretrain_mae_marine_crossview.yaml --exp_name crossview_pointmae
```

Each run saves checkpoints under
`experiments/<config_name>/cfgs/<exp_name>/ckpt-*.pth`. The best reconstruction
checkpoint is `ckpt-best.pth`.

---

## 4. Fine-tuning for real-world ReID

Fine-tuning reuses `main.py` with the relevant ReID configuration and the
pre-trained checkpoint:

```bash
python main.py --config <REID_CONFIG> --exp_name <RUN_NAME> --ckpts <PATH_TO_CKPT>
```

Available ReID configs:

| Backbone | Config | Notes |
| --- | --- | --- |
| PointTransformer baseline | `cfgs/finetune_marine_reid.yaml` | Trains from scratch on real scans. |
| Point-MAE (viewpoint) | `cfgs/finetune_marine_reid_pointmae.yaml` | Load the viewpoint pre-training weights. |
| Point-MAE (random) | `cfgs/finetune_marine_reid_pointmae_rand.yaml` | Compare against random masking. |
| PCP-MAE | `cfgs/finetune_marine_reid_pcpmae.yaml` | Requires `pretrain_pcp_mae_marine` checkpoint. |
| Point-M2AE | `cfgs/finetune_marine_reid_pointm2ae.yaml` | Requires `pretrain_m2ae_marine` checkpoint. |

Typical workflow:

```bash
# Viewpoint-masked Point-MAE ReID
python main.py --config cfgs/finetune_marine_reid_pointmae.yaml \
    --exp_name vp_pointmae_reid \
    --ckpts experiments/pretrain_mae_marine/cfgs/vp_pointmae/ckpt-best.pth

# PCP-MAE ReID
python main.py --config cfgs/finetune_marine_reid_pcpmae.yaml \
    --exp_name pcpmae_reid \
    --ckpts experiments/pretrain_pcp_mae_marine/cfgs/pcpmae/ckpt-best.pth

# Point-M2AE ReID
python main.py --config cfgs/finetune_marine_reid_pointm2ae.yaml \
    --exp_name pointm2ae_reid \
    --ckpts experiments/pretrain_m2ae_marine/cfgs/pointm2ae/ckpt-best.pth

# Random masking baseline
python main.py --config cfgs/finetune_marine_reid_pointmae_rand.yaml \
    --exp_name rand_pointmae_reid \
    --ckpts experiments/pretrain_mae_marine_rand/cfgs/rand_pointmae/ckpt-best.pth
```

During training the script logs Rank-1/Rank-5/mAP metrics to
`experiments/<...>/reid_metrics.json` and saves `ckpt-best.pth` with the highest
Rank-1 score.

---

## 5. Evaluation / testing

To evaluate a trained model without further updates, rerun `main.py` with
`--test` and the checkpoint path:

```bash
python main.py --config cfgs/finetune_marine_reid_pointmae.yaml \
    --exp_name vp_pointmae_reid \
    --ckpts experiments/pretrain_mae_marine/cfgs/vp_pointmae/ckpt-best.pth \
    --test
```

This command reuses the validation loader, reports Rank-1/Rank-5/mAP in the
console, and appends the results to `reid_metrics.json` for later aggregation.

### Evaluating scratch baselines on RealShip

`tools/evaluate_real_baselines.py` automates evaluation of the DGCNN,
PointNeXt, and PointTransformer scratch models directly on a RealShip split
(`reid_val` or `reid_test`). The helper clones each training config, swaps the
test loader to `cfgs/dataset_configs/Real_Ship.yaml`, and runs
`main.py --test` with the supplied checkpoints. By default it looks for the
`ckpt-best.pth` files generated by the example training commands in
`experiments/<config_name>/cfgs/<exp_name>/`.

```bash
# Evaluate the three scratch models on the RealShip test split
python tools/evaluate_real_baselines.py \
    --subset reid_test \
    --dgcnn-exp dgcnn_from_scratch \
    --pointnext-exp pointnext_from_scratch \
    --pointtransformer-exp pointtransformer_from_scratch

# Override an individual checkpoint if it lives elsewhere
python tools/evaluate_real_baselines.py --subset reid_val \
    --pointnext-ckpt /path/to/custom_pointnext_checkpoint.pth
```

Use `--batch-size` and `--num-workers` to adjust evaluation throughput if the
default settings are too small for your hardware.

---

## 6. Automated sim-to-real pipeline

`tools/sim2real_pipeline.py` automates the complete pre-train → fine-tune →
evaluate cycle for the four main backbones (viewpoint Point-MAE, PCP-MAE,
Point-M2AE, random Point-MAE). It can optionally enable viewpoint masking during
supervised fine-tuning.

```bash
# Preview the commands
python tools/sim2real_pipeline.py --dry-run

# Execute the full pipeline
python tools/sim2real_pipeline.py

# Optionally pass through --use_viewpoint_mask to the ReID stage
python tools/sim2real_pipeline.py --use-viewpoint-mask
```

After finishing, the script writes a consolidated summary to
`experiments/sim2real_summary.json` including the expected ranking
(viewpoint > PCP-MAE > Point-M2AE > random).

---

## 7. Tips and troubleshooting

* All training scripts support distributed launchers via `--launcher pytorch`.
* Use `--resume` to continue interrupted training from the last checkpoint.
* The ReID heads combine cross-entropy and batch-hard triplet losses. Adjust
  `triplet_weight` or `ce_weight` inside each config to rebalance the objectives.
* For cross-view Point-MAE experiments you can fine-tune using the same
  `cfgs/finetune_marine_reid_pointmae.yaml` configuration by pointing `--ckpts`
  at the cross-view checkpoint. The decoder learns to reconstruct alternate
  views, highlighting the benefit of paired-view masking.

Happy experimenting!
