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
class Point_PCP_MAE_Pretrain(nn.Module): # 클래스 이름은 그대로 사용
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_PCP_MAE_Pretrain] ', logger ='Point_PCP_MAE_Pretrain') # 로그 이름 변경
        self.config = config
        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        # self.cls_dim = config.cls_dim # 이 클래스에서는 사용 안 함
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        # NOTE: encoder_dims는 Encoder 클래스에서 사용됨
        self.encoder_dims = config.encoder_dims if hasattr(config, 'encoder_dims') else self.trans_dim # 하위 호환

        # Patch Feature Encoder (PointNet 같은 구조)
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        print_log(f'[Point_PCP_MAE_Pretrain] Patch Encoder using PointNet architecture with output dim {self.encoder_dims}', logger ='Point_PCP_MAE_Pretrain')

        # --- MAE Encoder (Transformer) ---
        # PCP-MAE 논문 구조를 따르려면, 패치 특징을 Transformer 인코더에 넣기 전에 Linear Embedding 필요
        self.patch_embed = nn.Linear(self.encoder_dims, self.trans_dim)

        self.mask_ratio = config.mask_ratio
        # self.mask_type = config.mask_type # PCP-MAE는 랜덤 패치 마스킹 사용
        # print_log(f'[Point_PCP_MAE_Pretrain] Using {self.mask_type} mask for Point Patch Masking', logger ='Point_PCP_MAE_Pretrain')

        # 위치 임베딩 생성기 (Sinusoidal or Learned)
        # PCP-MAE는 좌표 자체를 사용하지 않고 위치 임베딩만 사용
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        print_log(f'[Point_PCP_MAE_Pretrain] Using Learned Position Embedding', logger ='Point_PCP_MAE_Pretrain')

        # Transformer Encoder Blocks
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.MAE_encoder = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
            qkv_bias=config.qkv_bias if hasattr(config, 'qkv_bias') else False,
            qk_scale=config.qk_scale if hasattr(config, 'qk_scale') else None,
            attn_drop_rate=config.attn_drop_rate if hasattr(config, 'attn_drop_rate') else 0.,
            drop_rate=config.drop_rate if hasattr(config, 'drop_rate') else 0.
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        # --- MAE Encoder 끝 ---


        # --- MAE Decoder ---
        self.decoder_depth = config.decoder_depth
        self.decoder_num_heads = config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim, # 디코더 임베딩 차원은 인코더와 동일
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
            qkv_bias=config.qkv_bias if hasattr(config, 'qkv_bias') else False, # 인코더와 동일하게 설정
            qk_scale=config.qk_scale if hasattr(config, 'qk_scale') else None,
            attn_drop_rate=config.attn_drop_rate if hasattr(config, 'attn_drop_rate') else 0.,
            drop_rate=config.drop_rate if hasattr(config, 'drop_rate') else 0.
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        trunc_normal_(self.mask_token, std=.02)
        # --- MAE Decoder 끝 ---


        # --- 그룹핑 모듈 ---
        print_log(f'[Point_PCP_MAE_Pretrain] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='Point_PCP_MAE_Pretrain')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # --- 그룹핑 모듈 끝 ---


        # --- Prediction Head (포인트 좌표 재구성) ---
        # PCP-MAE 논문은 MLP 사용
        self.rebuild_head = nn.Sequential(
           nn.Linear(self.trans_dim, self.trans_dim // 2),
           nn.GELU(),
           nn.Linear(self.trans_dim // 2, 3 * self.group_size) # 각 패치 내 group_size개의 포인트 좌표(x,y,z) 예측
        )
        # --- Prediction Head 끝 ---

        # =================================================================
        # ★★★ PCP-MAE 수정 사항 (1/3): Center Prediction Head 추가 ★★★
        # =================================================================
        self.center_pred_head = nn.Sequential(
            nn.Linear(self.trans_dim, self.trans_dim // 2),
            nn.GELU(),
            nn.Linear(self.trans_dim // 2, 3) # (x, y, z) 3차원 좌표 예측
        )
        print_log(f'[Point_PCP_MAE_Pretrain] Added Center Prediction Head for PCP-MAE.', logger ='Point_PCP_MAE_Pretrain')
        # =================================================================


        # --- Loss ---
        self.loss = config.loss
        self.build_loss_func(self.loss) # 재구성(Reconstruction) 손실

        # =================================================================
        # ★★★ PCP-MAE 수정 사항 (2/3): Center Prediction Loss 추가 ★★★
        # =================================================================
        self.center_loss_func = nn.MSELoss() # 중심점 예측은 L2 (MSE) 손실 사용
        self.center_loss_weight = config.get('center_loss_weight', 1.0) # config에서 가중치 가져오기 (기본값 1.0)
        print_log(f'[Point_PCP_MAE_Pretrain] Using Center Prediction Loss (MSE) with weight {self.center_loss_weight}', logger ='Point_PCP_MAE_Pretrain')
        # =================================================================

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d): # Encoder 내부 Conv1d 초기화 추가
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError(f"Loss type {loss_type} not supported for PCP-MAE reconstruction.")

    def forward(self, pts, vis = False, **kwargs):
        """
        pts: (B, N, 3)
        vis: bool, for visualization
        """
        B = pts.shape[0]

        # 1. 그룹핑 (패치 생성)
        # neighborhood: (B, G, S, 3), center: (B, G, 3)
        # neighborhood는 각 패치 센터 기준 로컬 좌표
        neighborhood, center = self.group_divider(pts)

        # 2. 패치 특징 추출 (Patch Feature Extraction)
        # group_input_tokens: (B, G, encoder_dims)
        group_input_tokens = self.encoder(neighborhood)

        # 3. 패치 임베딩 (Linear Projection)
        # patch_features: (B, G, trans_dim)
        patch_features = self.patch_embed(group_input_tokens)

        # 4. 위치 임베딩 (Position Embedding)
        # pos_embed: (B, G, trans_dim)
        pos_embed = self.pos_embed(center)

        # 5. 패치 마스킹 (Patch Masking)
        num_patches = self.num_group
        num_masked_patches = int(self.mask_ratio * num_patches)
        num_visible_patches = num_patches - num_masked_patches

        # (B, G) 크기의 무작위 인덱스 생성
        # 패치 인덱스: [0, 1, 2, ... , G-1]
        patch_indices = torch.argsort(torch.rand(B, num_patches, device=pts.device), dim=1)

        # 마스킹될 패치 인덱스와 보일 패치 인덱스 분리
        masked_patch_indices = patch_indices[:, :num_masked_patches]  # (B, M) - M: 마스킹된 패치 수
        visible_patch_indices = patch_indices[:, num_masked_patches:] # (B, V) - V: 보일 패치 수 (V = G - M)

        # 6. 인코더 입력 준비
        # --- 보일 패치 선택 (gather) ---
        # visible_features: (B, V, trans_dim)
        visible_features = torch.gather(patch_features, 1, visible_patch_indices.unsqueeze(-1).expand(-1, -1, self.trans_dim))
        # visible_pos_embed: (B, V, trans_dim)
        visible_pos_embed = torch.gather(pos_embed, 1, visible_patch_indices.unsqueeze(-1).expand(-1, -1, self.trans_dim))

        # 7. MAE 인코더 실행
        # encoded_visible_features: (B, V, trans_dim)
        encoded_visible_features = self.MAE_encoder(visible_features, visible_pos_embed)
        encoded_visible_features = self.norm(encoded_visible_features) # Norm 추가

        # 8. MAE 디코더 입력 준비
        # (B, G, trans_dim) 크기의 전체 피처 텐서 (마스크 토큰으로 채워짐)
        decoder_input_features = self.mask_token.repeat(B, num_patches, 1)
        # (B, G, trans_dim) 크기의 전체 위치 임베딩 텐서 (마스크 토큰용 임베딩은 그냥 원본 pos_embed 사용 가능)
        decoder_input_pos_embed = pos_embed # 마스크된 위치에도 해당 위치 임베딩 사용 (Point-MAE/PCP-MAE 동일)

        # '보일 패치' 인덱스 위치에만 인코더 출력 피처 값을 복사 (scatter)
        # (B, V, D) -> (B, G, D)
        decoder_input_features.scatter_(1, visible_patch_indices.unsqueeze(-1).expand(-1, -1, self.trans_dim), encoded_visible_features)

        # 9. MAE 디코더 실행
        # decoded_mask_features: (B, M, trans_dim)
        # return_token_num=num_masked_patches : 마스크된 토큰에 대한 출력만 받음
        decoded_mask_features = self.MAE_decoder(decoder_input_features, decoder_input_pos_embed, return_token_num=num_masked_patches)

        # 10. 재구성 헤드 (Reconstruction Head)
        # rebuild_points: (B, M, S*3) -> (B*M, S, 3)
        rebuild_points_flat = self.rebuild_head(decoded_mask_features) # B, M, S*3
        rebuild_points = rebuild_points_flat.view(B, num_masked_patches, self.group_size, 3) # B, M, S, 3
        rebuild_points = rebuild_points.view(B * num_masked_patches, self.group_size, 3) # (B*M, S, 3) - 로컬 좌표

        # 11. 손실(Loss) 계산을 위한 타겟 준비
        # --- 마스킹된 패치의 원본 neighborhood 선택 (gather) ---
        # target_masked_neighborhood: (B, G, S, 3) -> (B, M, S, 3) -> (B*M, S, 3)
        # 주의: neighborhood는 이미 로컬 좌표계임
        target_masked_neighborhood = torch.gather(neighborhood, 1, masked_patch_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.group_size, 3))
        target_masked_neighborhood = target_masked_neighborhood.view(B * num_masked_patches, self.group_size, 3) # (B*M, S, 3)

        # 12. 재구성 손실 계산 (Chamfer Distance)
        loss_recon = self.loss_func(rebuild_points, target_masked_neighborhood)

        # =================================================================
        # ★★★ PCP-MAE 수정 사항 (3/3): Center Prediction Loss 계산 ★★★
        # =================================================================
        # 12-1. 마스킹된 중심점 예측
        # decoded_mask_features: (B, M, trans_dim)
        predicted_masked_centers = self.center_pred_head(decoded_mask_features) # (B, M, 3)
        
        # 12-2. 마스킹된 중심점의 실제 타겟값
        # center: (B, G, 3)
        target_masked_centers = torch.gather(center, 1, masked_patch_indices.unsqueeze(-1).expand(-1, -1, 3)) # (B, M, 3)

        # 12-3. 중심점 예측 손실 계산
        loss_center = self.center_loss_func(predicted_masked_centers, target_masked_centers)
        # =================================================================


        # 13. (옵션) 시각화 데이터 준비
        if vis:
            # 보일 패치의 포인트 (로컬 -> 글로벌 좌표 변환)
            visible_neighborhood = torch.gather(neighborhood, 1, visible_patch_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.group_size, 3)) # B, V, S, 3
            visible_center = torch.gather(center, 1, visible_patch_indices.unsqueeze(-1).expand(-1, -1, 3)) # B, V, 3
            vis_points_global = visible_neighborhood + visible_center.unsqueeze(2) # B, V, S, 3
            vis_points_flat = vis_points_global.view(B, num_visible_patches * self.group_size, 3) # B, V*S, 3

            # 재구성된 마스크 패치의 포인트 (로컬 -> 글로벌 좌표 변환)
            rebuild_points_local_reshaped = rebuild_points.view(B, num_masked_patches, self.group_size, 3) # B, M, S, 3
            masked_center = torch.gather(center, 1, masked_patch_indices.unsqueeze(-1).expand(-1, -1, 3)) # B, M, 3
            rebuild_points_global = rebuild_points_local_reshaped + masked_center.unsqueeze(2) # B, M, S, 3
            rebuild_points_flat = rebuild_points_global.view(B, num_masked_patches * self.group_size, 3) # B, M*S, 3

            # 전체 재구성된 포인트 클라우드 (보이는 부분 + 재구성된 부분)
            # 순서가 중요할 수 있으므로, 원본 패치 순서대로 정렬 필요
            full_reconstructed_points = torch.zeros(B, num_patches, self.group_size, 3, device=pts.device)

            # 보이는 패치 채우기 (scatter_)
            # visible_neighborhood 를 사용해야 원래 형상이 보임
            full_reconstructed_points.scatter_(1, visible_patch_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.group_size, 3), visible_neighborhood)

            # 재구성된 마스크 패치 채우기 (scatter_)
            full_reconstructed_points.scatter_(1, masked_patch_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.group_size, 3), rebuild_points_local_reshaped)

            # 글로벌 좌표로 변환
            full_reconstructed_global = full_reconstructed_points + center.unsqueeze(2) # B, G, S, 3
            full_reconstructed_flat = full_reconstructed_global.view(B, num_patches * self.group_size, 3) # B, N_recon, 3

            # 시각화를 위해 첫 번째 배치 아이템만 반환
            return full_reconstructed_flat[0].unsqueeze(0), vis_points_flat[0].unsqueeze(0) # (1, N_recon, 3), (1, V*S, 3)

        # 최종 손실 반환
        total_loss = loss_recon + self.center_loss_weight * loss_center
        
        return total_loss # , {"total_loss": total_loss, "recon_loss": loss_recon, "center_loss": self.center_loss_weight * loss_center}
        # 참고: 훈련 루프가 dict를 처리할 수 있다면, 주석 처리된 dict를 반환하여 두 손실을 개별적으로 로깅하는 것이 좋습니다.
        # 훈련 루프가 단일 텐서만 받는다면, 지금처럼 total_loss만 반환합니다.