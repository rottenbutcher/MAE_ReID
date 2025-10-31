import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_

from .Point_MAE import Encoder, Group, TransformerEncoder
from .build import MODELS
from utils.checkpoint import (
    get_missing_parameters_message,
    get_unexpected_parameters_message,
)
from utils.logger import print_log
from utils.loss_utils import classification_triplet_loss


@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(
            num_group=self.num_group,
            group_size=self.group_size,
        )

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * 2, 256),
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

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def build_loss_func(self):
        self.loss_ce = nn.NLLLoss()
        self.triplet_margin = self.config.get('triplet_margin', 0.3)
        self.triplet_weight = self.config.get('triplet_weight', 1.0)
        self.ce_weight = self.config.get('ce_weight', 1.0)
        self.normalize_triplet = self.config.get('normalize_triplet_feature', True)

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

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {
                k.replace("module.", ""): v for k, v in ckpt['base_model'].items()
            }

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]
            for k in list(base_ckpt.keys()):
                if k.startswith('cls_head_finetune'):
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer',
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer',
                )

            print_log(
                f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}',
                logger='Transformer',
            )
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, pts, return_features=False):
        neighborhood, center = self.group_divider(pts)

        group_input_tokens = self.encoder(neighborhood)
        cls_tokens = self.cls_token.expand(pts.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(pts.size(0), -1, -1)
        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)

        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)

        try:
            features = self.cls_head_finetune[:-1](concat_f)
            logits = self.cls_head_finetune[-1](features)
        except Exception as e:
            print(f"Warning: Feature extraction failed, falling back. {e}")
            logits = self.cls_head_finetune(concat_f)
            features = concat_f

        log_softmax_out = F.log_softmax(logits, dim=-1)

        if return_features:
            return log_softmax_out, features
        return log_softmax_out
