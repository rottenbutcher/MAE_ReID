import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_

from .Point_MAE import get_graph_feature
from .build import MODELS
from utils.logger import print_log
from utils.loss_utils import classification_triplet_loss


@MODELS.register_module()
class DGCNN_ReID(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.cls_dim = config.cls_dim
        self.k = config.get('k', 20)
        self.emb_dims = config.get('embed_dim', 1024)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(self.emb_dims)

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.emb_dims * 2, 256),
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
            if m.bias is not None:
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
                f'DGCNN_ReID: Checkpoint path {ckpt_path} provided, but training from scratch.',
                logger='DGCNN_ReID',
            )
        else:
            print_log('DGCNN_ReID: Training from scratch!!!', logger='DGCNN_ReID')
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
        x = pts.transpose(1, 2).float()

        x1 = get_graph_feature(x, k=self.k)
        x1 = self.conv1(x1)
        x1 = x1.max(dim=-1, keepdim=False)[0]

        x2 = get_graph_feature(x1, k=self.k)
        x2 = self.conv2(x2)
        x2 = x2.max(dim=-1, keepdim=False)[0]

        x3 = get_graph_feature(x2, k=self.k)
        x3 = self.conv3(x3)
        x3 = x3.max(dim=-1, keepdim=False)[0]

        x4 = get_graph_feature(x3, k=self.k)
        x4 = self.conv4(x4)
        x4 = x4.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)

        x_max = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        x_avg = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        global_feat = torch.cat((x_max, x_avg), dim=1)

        try:
            features = self.cls_head_finetune[:-1](global_feat)
            logits = self.cls_head_finetune[-1](features)
        except Exception as e:
            print(f"Warning: DGCNN_ReID Feature extraction failed. {e}")
            logits = self.cls_head_finetune(global_feat)
            features = global_feat

        log_softmax_out = F.log_softmax(logits, dim=-1)

        if return_features:
            return log_softmax_out, features
        return log_softmax_out
