import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
from knn_cuda import KNN
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from datasets.data_transforms import PointcloudViewpointMasking
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import print_log
import torch.nn.functional as F # forward에서 사용
from pointnet2_ops import pointnet2_utils

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (B, N, k)
    return idx


def get_graph_feature(x, k=20, idx=None, x_coord=None):
    B, C, N = x.size()
    if idx is None:
        if x_coord is None: # T-Net K=3
            idx = knn(x, k=k)
        else: # T-Net K=6
            idx = knn(x_coord, k=k)
    
    device = x.device
    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1)*N
    idx = idx + idx_base
    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()   # (B, N, C)
    feature = x.view(B*N, -1)[idx, :]    # (B*N*k, C)
    feature = feature.view(B, N, k, num_dims) 
    x = x.view(B, N, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature  # (B, 2*C, N, k)

class Encoder(nn.Module):   ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x


# Pretrain model
class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio 
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth 
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads 
        print_log(f'[args] {config.transformer_config}', logger = 'Transformer')
        # embedding
        self.encoder_dims =  config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)

        self.mask_type = config.transformer_config.mask_type

        if self.mask_type == 'viewpoint':
            vp_ratio = config.transformer_config.get('viewpoint_mask_ratio', 0.5)
            # MAE에서는 전체 마스킹 비율을 mask_ratio로 제어하므로, random_mask_ratio 계산
            # 전체 중 50%가 가려지고, 남은 50% 중 일부를 랜덤 마스킹하여 총 60%를 맞추려면...
            # 0.5 + 0.5 * X = 0.6  => 0.5 * X = 0.1 => X = 0.2
            # 따라서 random_mask_ratio = 0.2
            remaining_ratio = 1 - vp_ratio
            if remaining_ratio > 0:
                rand_ratio = (self.mask_ratio - vp_ratio) / remaining_ratio
            else:
                rand_ratio = 0
            self.viewpoint_masker = PointcloudViewpointMasking(viewpoint_mask_ratio=vp_ratio, random_mask_ratio=rand_ratio)
            print_log(f'[Point_MAE] Using Ship-specific Viewpoint Masking with vp_ratio={vp_ratio:.2f}, rand_ratio={rand_ratio:.2f}', logger='Point_MAE')

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
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

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug = False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device) # B G

    def forward(self, neighborhood, center, noaug = False):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug = noaug) # B G
        elif self.mask_type == 'viewpoint':
            # PointcloudViewpointMasking은 점을 0으로 만들지만, 여기서는 마스크 인덱스만 필요함
            # __call__ 메서드를 직접 호출하지 않고, 내부 로직을 일부 활용하여 bool_masked_pos를 생성
            
            # 임시로 마스킹된 포인트를 생성하여 마스크를 얻음
            masked_centers = self.viewpoint_masker(center)
            # 점이 0이 된 위치를 찾아 마스크로 사용 (원래 점이 0이었을 경우 제외)
            bool_masked_pos = (masked_centers.abs().sum(dim=-1) == 0) & (center.abs().sum(dim=-1) != 0)
        # --- 수정 끝 ---
        
        else:
            bool_masked_pos = self._mask_center_block(center, noaug = noaug)

        group_input_tokens = self.encoder(neighborhood)  #  B G C

        batch_size, seq_len, C = group_input_tokens.size()

        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        # add pos embedding
        # mask pos center
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)

        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis, bool_masked_pos


@MODELS.register_module()
class Point_MAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_MAE] ', logger ='Point_MAE')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='Point_MAE')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()


    def forward(self, pts, vis = False, **kwargs):
        neighborhood, center = self.group_divider(pts)

        x_vis, mask = self.MAE_encoder(neighborhood, center)
        B,_,C = x_vis.shape # B VIS C

        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)

        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

        _,N,_ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        x_rec = self.MAE_decoder(x_full, pos_full, N)

        B, M, C = x_rec.shape
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024

        gt_points = neighborhood[mask].reshape(B*M,-1,3)
        loss1 = self.loss_func(rebuild_points, gt_points)

        if vis: #visualization
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
            full_vis = vis_points + center[~mask].unsqueeze(1)
            full_rebuild = rebuild_points + center[mask].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0)
            # full_points = torch.cat([rebuild_points,vis_points], dim=0)
            full_center = torch.cat([center[mask], center[~mask]], dim=0)
            # full = full_points + full_center.unsqueeze(1)
            ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
            ret1 = full.reshape(-1, 3).unsqueeze(0)
            # return ret1, ret2
            return ret1, ret2, full_center
        else:
            return loss1

# finetune model
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

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
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
                nn.Linear(256, self.cls_dim)
            )

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def build_loss_func(self):
        self.loss_ce = nn.NLLLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder') :
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
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
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

    def forward(self, pts, return_features=False):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)

        try:
            # 마지막 분류 레이어(self.cls_head_finetune[-1]) 직전까지 실행하여 피처 추출
            features = self.cls_head_finetune[:-1](concat_f) # (B, 256) ReID 피처
            
            # 피처를 마지막 분류 레이어에 통과시켜 로짓(logits) 계산
            logits = self.cls_head_finetune[-1](features)   # (B, cls_dim)
            
        except Exception as e:
            # (안전 장치) 혹시 모를 예외 발생 시, 기존 방식대로 실행
            print(f"Warning: Feature extraction failed, falling back. {e}")
            logits = self.cls_head_finetune(concat_f)
            features = concat_f # fallback feature

        log_softmax_out = F.log_softmax(logits, dim=-1)

        # if return_features:
        #     # ReID 평가 시 (Step 4에서 사용): (소프트맥스, 피처) 튜플 반환
        #     return log_softmax_out, features
        # else:
        #     # 기본(기존 Classification) 동작: 소프트맥스 결과만 반환
        #     return log_softmax_out
        
        return log_softmax_out, features


@MODELS.register_module()
class DGCNN_ReID(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.cls_dim = config.cls_dim
        self.k = config.get('k', 20)
        self.emb_dims = config.get('embed_dim', 1024) # 백본 출력 차원

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(self.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        # --- ReID 분류 헤드 ---
        # (PointTransformer와 유사한 구조, 입력 차원만 DGCNN에 맞게 수정)
        # DGCNN 백본 출력은 max+avg 풀링을 거쳐 2 * emb_dims가 됩니다.
        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.emb_dims * 2, 256), 
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.build_loss_func()
        self.apply(self._init_weights) # 가중치 초기화

    def build_loss_func(self):
        self.loss_ce = nn.NLLLoss()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def load_model_from_ckpt(self, ckpt_path):
        # DGCNN은 MAE 사전학습과 호환되지 않음
        if ckpt_path is not None:
             print_log(f'DGCNN_ReID: Checkpoint path {ckpt_path} provided, but training from scratch.', logger='DGCNN_ReID')
        else:
             print_log('DGCNN_ReID: Training from scratch!!!', logger='DGCNN_ReID')
        self.apply(self._init_weights)

    def get_loss_acc(self, ret, gt):
        # PointTransformer와 동일
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def forward(self, pts, return_features=False):
        # --- DGCNN 백본 ---
        x = pts.transpose(1, 2).float() # (B, 3, N)
        B, C, N = x.size()

        x1 = get_graph_feature(x, k=self.k)      # (B, 6, N, k)
        x1 = self.conv1(x1)                      # (B, 64, N, k)
        x1 = x1.max(dim=-1, keepdim=False)[0]    # (B, 64, N)

        x2 = get_graph_feature(x1, k=self.k)     # (B, 128, N, k)
        x2 = self.conv2(x2)                      # (B, 64, N, k)
        x2 = x2.max(dim=-1, keepdim=False)[0]    # (B, 64, N)

        x3 = get_graph_feature(x2, k=self.k)     # (B, 128, N, k)
        x3 = self.conv3(x3)                      # (B, 128, N, k)
        x3 = x3.max(dim=-1, keepdim=False)[0]    # (B, 128, N)

        x4 = get_graph_feature(x3, k=self.k)     # (B, 256, N, k)
        x4 = self.conv4(x4)                      # (B, 256, N, k)
        x4 = x4.max(dim=-1, keepdim=False)[0]    # (B, 256, N)

        x = torch.cat((x1, x2, x3, x4), dim=1)   # (B, 512, N)
        x = self.conv5(x)                        # (B, emb_dims, N)

        # 전역 피처 생성
        x_max = F.adaptive_max_pool1d(x, 1).squeeze(-1) # (B, emb_dims)
        x_avg = F.adaptive_avg_pool1d(x, 1).squeeze(-1) # (B, emb_dims)
        global_feat = torch.cat((x_max, x_avg), dim=1)  # (B, emb_dims * 2)
        # --- 백본 끝 ---

        # --- ReID 헤드 ---
        try:
            features = self.cls_head_finetune[:-1](global_feat) # (B, 256) ReID 피처
            logits = self.cls_head_finetune[-1](features)   # (B, cls_dim)
        except Exception as e:
            print(f"Warning: DGCNN_ReID Feature extraction failed. {e}")
            logits = self.cls_head_finetune(global_feat)
            features = global_feat

        log_softmax_out = F.log_softmax(logits, dim=-1)

        return log_softmax_out, features
    
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        """
        npoint: 샘플링할 포인트 수
        radius: Ball Query 반경
        nsample: 그룹당 샘플 수
        in_channel: 입력 피처 차원 (xyz 포함)
        mlp: MLP 레이어 차원 리스트
        group_all: 모든 포인트를 하나의 그룹으로 묶을지 여부 (마지막 레이어용)
        """
        super(PointNetSetAbstraction, self).__init__()
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
        """
        xyz: (B, N, 3)
        features: (B, C, N)
        """

        xyz = xyz.float().contiguous()
        if features is not None:
            features = features.float().contiguous()
        B, N, C = xyz.shape
        
        if features is not None:
            features = features.contiguous()

        xyz = xyz.contiguous()

        if self.group_all:
            # 모든 포인트를 하나의 그룹으로
            new_xyz = torch.zeros(B, 1, 3, device=xyz.device)
            grouped_idx = torch.arange(N, device=xyz.device).repeat(B, 1, 1).view(B, 1, N).int()
        else:
            # FPS (Furthest Point Sampling)
            fps_idx = pointnet2_utils.furthest_point_sample(xyz.float(), self.npoint) # (B, npoint)

            # 2. gather_operation을 사용해 인덱스로부터 실제 *좌표*를 가져옵니다.
            #    gather_operation은 (B, C, N) 입력을 기대하므로 xyz를 transpose합니다.
            new_xyz_transposed = pointnet2_utils.gather_operation(
                xyz.transpose(1, 2).contiguous().float(), 
                fps_idx.int()
            ) # (B, 3, npoint)
            
            # 3. new_xyz를 (B, npoint, 3) 형태로 다시 transpose합니다.
            new_xyz = new_xyz_transposed.transpose(1, 2).contiguous() # (B, npoint, 3)

            # 4. 이제 new_xyz는 실제 좌표이므로 ball_query가 정상 작동합니다.
            grouped_idx = pointnet2_utils.ball_query(self.radius, self.nsample, xyz.float(), new_xyz.float()).int()
        
        # Point Grouping
        # (B, C, N) -> (B, C, npoint, nsample)
        if features is not None:
            grouped_features = pointnet2_utils.grouping_operation(features, grouped_idx)
        
        # Coordinate Grouping & Normalization
        # (B, 3, N) -> (B, 3, npoint, nsample)
        grouped_xyz = pointnet2_utils.grouping_operation(xyz.transpose(1, 2).contiguous(), grouped_idx)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1) # 로컬 좌표계로 정규화
        
        if features is not None:
            # [xyz(local), features]
            new_features = torch.cat([grouped_xyz, grouped_features], dim=1) # (B, C+3, npoint, nsample)
        
        # [!!! 수정된 부분: 'else' 블록 추가 !!!]
        else:
            # features가 없으면, 로컬 좌표(grouped_xyz)만 new_features가 됩니다.
            new_features = grouped_xyz # (B, 3, npoint, nsample)
        # MLP
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_features = F.relu(bn(conv(new_features)))

        # Max Pooling
        new_features = torch.max(new_features, 3)[0] # (B, D, npoint)

        return new_xyz, new_features
    
@MODELS.register_module()
class PointNet2_ReID(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.cls_dim = config.cls_dim
        
        # PointNet++ (SSG) Encoder
        # 입력 포인트 수(npoints)에 따라 npoint를 조절할 수 있습니다.
        # (입력 1024 기준)
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        
        # PointNeXt YAML의 'pointnext_output_dim'과 맞춥니다.
        self.pointnet_output_dim = config.get('pointnext_output_dim', 1024)
        
        # Global Set Abstraction (모든 포인트를 하나로)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, self.pointnet_output_dim], group_all=True)

        # --- ReID 분류 헤드 ---
        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.pointnet_output_dim, 256), 
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )
        
        self.build_loss_func()
        self.apply(self._init_weights) # 가중치 초기화

    def build_loss_func(self):
        self.loss_ce = nn.NLLLoss()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def load_model_from_ckpt(self, ckpt_path):
        # PointNet++은 MAE 사전학습과 호환되지 않음
        if ckpt_path is not None:
             print_log(f'PointNet2_ReID: Checkpoint path {ckpt_path} provided, but training from scratch.', logger='PointNet2_ReID')
        else:
             print_log('PointNet2_ReID: Training from scratch!!!', logger='PointNet2_ReID')
        self.apply(self._init_weights)

    def get_loss_acc(self, ret, gt):
        # PointTransformer와 동일
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def forward(self, pts, return_features=False):
        # --- PointNet++ 백본 ---
        B, N, C = pts.shape
        xyz = pts.float() # (B, N, 3)
        features = None # (B, C, N) - 처음엔 xyz만 사용

        # Set Abstraction 1
        # xyz: (B, 512, 3), features: (B, 128, 512)
        xyz_sa1, features_sa1 = self.sa1(xyz, features) 
        
        # Set Abstraction 2
        # xyz: (B, 128, 3), features: (B, 256, 128)
        xyz_sa2, features_sa2 = self.sa2(xyz_sa1, features_sa1) 
        
        # Global Set Abstraction
        # xyz: (B, 1, 3), features: (B, output_dim, 1)
        xyz_sa3, features_sa3 = self.sa3(xyz_sa2, features_sa2)
        
        # (B, output_dim)
        global_feat = features_sa3.view(B, self.pointnet_output_dim)
        # --- 백본 끝 ---

        # --- ReID 헤드 ---
        try:
            features = self.cls_head_finetune[:-1](global_feat) # (B, 256) ReID 피처
            logits = self.cls_head_finetune[-1](features)   # (B, cls_dim)
        except Exception as e:
            print(f"Warning: PointNet2_ReID Feature extraction failed. {e}")
            logits = self.cls_head_finetune(global_feat)
            features = global_feat

        log_softmax_out = F.log_softmax(logits, dim=-1)

        return log_softmax_out, features