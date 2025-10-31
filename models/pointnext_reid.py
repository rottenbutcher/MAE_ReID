import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_

from .build import MODELS
from utils.logger import print_log
from utils.loss_utils import classification_triplet_loss

from pointnet2_ops import pointnet2_utils


class PointNetSetAbstraction(nn.Module):
    def __init__(
        self,
        npoint,
        radius,
        nsample,
        in_channel,
        mlp,
        group_all,
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, features):
        xyz = xyz.float().contiguous()
        if features is not None:
            features = features.float().contiguous()
        B, N, _ = xyz.shape

        if self.group_all:
            new_xyz = torch.zeros(B, 1, 3, device=xyz.device)
            grouped_idx = (
                torch.arange(N, device=xyz.device)
                .repeat(B, 1, 1)
                .view(B, 1, N)
                .int()
            )
        else:
            fps_idx = pointnet2_utils.furthest_point_sample(xyz.float(), self.npoint)

            new_xyz_transposed = pointnet2_utils.gather_operation(
                xyz.transpose(1, 2).contiguous().float(),
                fps_idx.int(),
            )

            new_xyz = new_xyz_transposed.transpose(1, 2).contiguous()

            grouped_idx = pointnet2_utils.ball_query(
                self.radius,
                self.nsample,
                xyz.float(),
                new_xyz.float(),
            ).int()

        if features is not None:
            grouped_features = pointnet2_utils.grouping_operation(
                features,
                grouped_idx,
            )

        grouped_xyz = pointnet2_utils.grouping_operation(
            xyz.transpose(1, 2).contiguous(),
            grouped_idx,
        )
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
        else:
            new_features = grouped_xyz

        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_features = F.relu(bn(conv(new_features)))

        new_features = torch.max(new_features, 3)[0]

        return new_xyz, new_features


@MODELS.register_module(name=['PointNet2_ReID', 'PointNeXt_ReID'])
class PointNeXt_ReID(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.cls_dim = config.cls_dim

        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=3,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False,
        )

        self.pointnet_output_dim = config.get('pointnext_output_dim', 1024)

        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 + 3,
            mlp=[256, 512, self.pointnet_output_dim],
            group_all=True,
        )

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.pointnet_output_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim),
        )

        self.build_loss_func()
        self.apply(self._init_weights)

    def build_loss_func(self):
        self.loss_ce = nn.NLLLoss()
        self.triplet_margin = self.config.get('triplet_margin', 0.3)
        self.triplet_weight = self.config.get('triplet_weight', 1.0)
        self.ce_weight = self.config.get('ce_weight', 1.0)
        self.normalize_triplet = self.config.get('normalize_triplet_feature', True)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def load_model_from_ckpt(self, ckpt_path):
        if ckpt_path is not None:
            print_log(
                f'PointNet2_ReID: Checkpoint path {ckpt_path} provided, but training from scratch.',
                logger='PointNet2_ReID',
            )
        else:
            print_log('PointNet2_ReID: Training from scratch!!!', logger='PointNet2_ReID')
        self.apply(self._init_weights)

    def get_loss_acc(self, ret, gt, features=None):
        total_loss, _, _ = classification_triplet_loss(
            ret,
            gt,
            self.loss_ce,
            features=features,
            margin=self.triplet_margin,
            ce_weight=self.ce_weight,
            triplet_weight=self.triplet_weight,
            normalize_triplet=self.normalize_triplet,
        )
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return total_loss, acc * 100

    def forward(self, pts, return_features=False):
        B, N, _ = pts.shape
        xyz = pts.float()
        features = None

        xyz_sa1, features_sa1 = self.sa1(xyz, features)
        xyz_sa2, features_sa2 = self.sa2(xyz_sa1, features_sa1)
        _, features_sa3 = self.sa3(xyz_sa2, features_sa2)

        global_feat = features_sa3.view(B, self.pointnet_output_dim)

        try:
            features = self.cls_head_finetune[:-1](global_feat)
            logits = self.cls_head_finetune[-1](features)
        except Exception as e:
            print(f"Warning: PointNet2_ReID Feature extraction failed. {e}")
            logits = self.cls_head_finetune(global_feat)
            features = global_feat

        log_softmax_out = F.log_softmax(logits, dim=-1)

        if return_features:
            return log_softmax_out, features
        return log_softmax_out
