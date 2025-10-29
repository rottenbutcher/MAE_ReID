import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_
from .build import MODELS
import random
from extensions.chamfer_dist import ChamferDistanceL2

from utils.logger import *
from utils.loss_utils import classification_triplet_loss
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from .modules import * # ★★★ 중요: 이 파일은 'modules.py'가 필요합니다. (아래 설명 참조)


# Hierarchical Encoder
class H_Encoder(nn.Module):

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.mask_ratio = config.mask_ratio 
        self.encoder_depths = config.encoder_depths
        self.encoder_dims =  config.encoder_dims
        self.local_radius = config.local_radius

        # token merging and positional embeddings
        self.token_embed = nn.ModuleList()
        self.encoder_pos_embeds = nn.ModuleList()
        for i in range(len(self.encoder_dims)):
            if i == 0:
                self.token_embed.append(Token_Embed(in_c=3, out_c=self.encoder_dims[i]))
            else:
                self.token_embed.append(Token_Embed(in_c=self.encoder_dims[i - 1], out_c=self.encoder_dims[i]))
            
            self.encoder_pos_embeds.append(nn.Sequential(
                                nn.Linear(3, self.encoder_dims[i]),
                                nn.GELU(),
                                nn.Linear(self.encoder_dims[i], self.encoder_dims[i]),
                            ))

        # encoder blocks
        self.encoder_blocks = nn.ModuleList()

        depth_count = 0
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(self.encoder_depths))]
        for i in range(len(self.encoder_depths)):
            self.encoder_blocks.append(Encoder_Block(
                                embed_dim=self.encoder_dims[i],
                                depth=self.encoder_depths[i],
                                drop_path_rate=dpr[depth_count: depth_count + self.encoder_depths[i]],
                                num_heads=config.num_heads,
                            ))
            depth_count += self.encoder_depths[i]

        self.encoder_norms = nn.ModuleList()
        for i in range(len(self.encoder_depths)):
            self.encoder_norms.append(nn.LayerNorm(self.encoder_dims[i]))

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

    def rand_mask(self, center):
        B, G, _ = center.shape
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

    def local_att_mask(self, xyz, radius, dist=None):
        with torch.no_grad():
            if dist is None or dist.shape[1] != xyz.shape[1]:
                dist = torch.cdist(xyz, xyz, p=2)
            # entries that are True in the mask do not contribute to self-attention
            # so points outside the radius are not considered
            mask = dist >= radius
        return mask, dist

    def forward(self, neighborhoods, centers, idxs, eval=False):
        # generate mask at the highest level
        bool_masked_pos = []
        if eval:
            # no mask
            B, G, _ = centers[-1].shape
            bool_masked_pos.append(torch.zeros(B, G).bool().cuda())
        else:
            # mask_index: 1, mask; 0, vis
            bool_masked_pos.append(self.rand_mask(centers[-1]))

        # Multi-scale Masking by back-propagation
        for i in range(len(neighborhoods) - 1, 0, -1):
            b, g, k, _ = neighborhoods[i].shape
            idx = idxs[i].reshape(b * g, -1)
            idx_masked = ~(bool_masked_pos[-1].reshape(-1).unsqueeze(-1)) * idx
            idx_masked = idx_masked.reshape(-1).long()
            masked_pos = torch.ones(b * centers[i - 1].shape[1]).cuda().scatter(0, idx_masked, 0).bool()
            bool_masked_pos.append(masked_pos.reshape(b, centers[i - 1].shape[1]))

        # hierarchical encoding
        bool_masked_pos.reverse()
        x_vis_list = []
        mask_vis_list = []
        xyz_dist = None
        for i in range(len(centers)):
            # 1st-layer encoder, conduct token embedding
            if i == 0:
                group_input_tokens = self.token_embed[i](neighborhoods[0])
            # intermediate layers, conduct token merging
            else:
                b, g1, _ = x_vis.shape
                b, g2, k2, _ = neighborhoods[i].shape
                x_vis_neighborhoods = x_vis.reshape(b * g1, -1)[idxs[i], :].reshape(b, g2, k2, -1)
                group_input_tokens = self.token_embed[i](x_vis_neighborhoods)

            # visible_index
            bool_vis_pos = ~(bool_masked_pos[i])
            batch_size, seq_len, C = group_input_tokens.size()

            # Due to Multi-scale Masking different, samples of a batch have varying numbers of visible tokens
            # find the longest visible sequence in the batch
            vis_tokens_len = bool_vis_pos.long().sum(dim=1)
            max_tokens_len = torch.max(vis_tokens_len)
            # use the longest length (max_tokens_len) to construct tensors
            x_vis = torch.zeros(batch_size, max_tokens_len, C).cuda()
            masked_center = torch.zeros(batch_size, max_tokens_len, 3).cuda()
            mask_vis = torch.ones(batch_size, max_tokens_len, max_tokens_len).cuda()
            
            for bz in range(batch_size):
                # inject valid visible tokens
                vis_tokens = group_input_tokens[bz][bool_vis_pos[bz]]
                x_vis[bz][0: vis_tokens_len[bz]] = vis_tokens
                # inject valid visible centers
                vis_centers = centers[i][bz][bool_vis_pos[bz]]
                masked_center[bz][0: vis_tokens_len[bz]] = vis_centers
                # the mask for valid visible tokens/centers
                mask_vis[bz][0: vis_tokens_len[bz], 0: vis_tokens_len[bz]] = 0
            
            if self.local_radius[i] > 0:
                mask_radius, xyz_dist = self.local_att_mask(masked_center, self.local_radius[i], xyz_dist)
                # disabled for pre-training, this step would not change mask_vis by *
                mask_vis_att = mask_radius * mask_vis
            else:
                mask_vis_att = mask_vis # eval=True (finetune) 시에는 mask_vis_att가 mask_vis가 됨 (0, 1로 채워짐)

            pos = self.encoder_pos_embeds[i](masked_center)

            x_vis = self.encoder_blocks[i](x_vis, pos, mask_vis_att)
            x_vis_list.append(x_vis)
            mask_vis_list.append(~(mask_vis[:, :, 0].bool()))

            if i == len(centers) - 1:
                 # H_Encoder의 eval=True (finetune) forward를 위한 로직
                 if eval:
                     return self.encoder_norms[i](x_vis_list[i])
                 pass # pretrain 시에는 마지막 레이어는 아래 로직을 스킵
            else:
                group_input_tokens[bool_vis_pos] = x_vis[~(mask_vis[:, :, 0].bool())]
                x_vis = group_input_tokens

        # pretrain 시에만 norm 적용 (finetune 시에는 위에서 eval=True로 리턴됨)
        for i in range(len(x_vis_list)):
            x_vis_list[i] = self.encoder_norms[i](x_vis_list[i])

        return x_vis_list, mask_vis_list, bool_masked_pos


@MODELS.register_module()
class Point_M2AE(nn.Module):

    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_M2AE]', logger ='Point_M2AE')
        self.config = config
        
        # tokenizers
        self.group_sizes = config.group_sizes
        self.num_groups = config.num_groups
        self.group_dividers = nn.ModuleList()
        for i in range(len(self.group_sizes)):
            self.group_dividers.append(Group(num_group=self.num_groups[i], group_size=self.group_sizes[i]))

        # hierarchical encoder
        self.h_encoder = H_Encoder(config)

        # hierarchical decoder
        self.decoder_depths = config.decoder_depths
        self.decoder_dims = config.decoder_dims
        self.decoder_up_blocks = config.decoder_up_blocks

        self.mask_token = nn.Parameter(torch.zeros(1, self.decoder_dims[0]))
        trunc_normal_(self.mask_token, std=.02)

        self.h_decoder = nn.ModuleList()
        self.decoder_pos_embeds = nn.ModuleList()
        self.token_prop = nn.ModuleList()

        depth_count = 0
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(self.decoder_depths))]
        for i in range(0, len(self.decoder_dims)):
            # decoder block
            self.h_decoder.append(Decoder_Block(
                            embed_dim=self.decoder_dims[i],
                            depth=self.decoder_depths[i],
                            drop_path_rate=dpr[depth_count: depth_count + self.decoder_depths[i]],
                            num_heads=config.num_heads,
                        ))
            depth_count += self.decoder_depths[i]
            # decoder's positional embeddings
            self.decoder_pos_embeds.append(nn.Sequential(
                            nn.Linear(3, self.decoder_dims[i]),
                            nn.GELU(),
                            nn.Linear(self.decoder_dims[i], self.decoder_dims[i])
                        ))
            # token propagation
            if i > 0:
                self.token_prop.append(PointNetFeaturePropagation(
                                    self.decoder_dims[i] + self.decoder_dims[i - 1], self.decoder_dims[i],
                                    blocks=self.decoder_up_blocks[i - 1], groups=1, res_expansion=1.0, bias=True
                                ))  
        self.decoder_norm = nn.LayerNorm(self.decoder_dims[-1])
            
        # prediction head
        self.rec_head = nn.Conv1d(self.decoder_dims[-1], 3 * self.group_sizes[0], 1)
        # loss
        self.rec_loss = ChamferDistanceL2().cuda()

    def forward(self, pts, eval=False, **kwargs):
        # multi-scale representations of point clouds
        neighborhoods, centers, idxs = [], [], []
        for i in range(len(self.group_dividers)):
            if i == 0:
                neighborhood, center, idx = self.group_dividers[i](pts)
            else:
                neighborhood, center, idx = self.group_dividers[i](center)
            neighborhoods.append(neighborhood)
            centers.append(center)
            idxs.append(idx)  # neighbor indices

        # hierarchical encoder
        if eval:
            # for linear svm
            x_vis = self.h_encoder(neighborhoods, centers, idxs, eval=True) # eval=True로 H_Encoder 호출
            return x_vis.mean(1) + x_vis.max(1)[0]
        else:
            x_vis_list, mask_vis_list, masks = self.h_encoder(neighborhoods, centers, idxs, eval=False)

        # hierarchical decoder
        centers.reverse()
        neighborhoods.reverse()
        x_vis_list.reverse()
        masks.reverse()
        
        # H_Encoder에서 mask_vis_list를 반환했으므로, 여기서도 사용해야 함
        mask_vis_list.reverse() 

        for i in range(len(self.decoder_dims)):
            center = centers[i]
            # 1st-layer decoder, concatenate visible and masked tokens
            if i == 0:
                x_full, mask = x_vis_list[i], masks[i]
                B, _, C = x_full.shape
                center_0 = torch.cat((center[~mask].reshape(B, -1, 3), center[mask].reshape(B, -1, 3)), dim=1)

                pos_emd_vis = self.decoder_pos_embeds[i](center[~mask]).reshape(B, -1, C)
                pos_emd_mask = self.decoder_pos_embeds[i](center[mask]).reshape(B, -1, C)
                pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

                _, N, _ = pos_emd_mask.shape
                mask_token = self.mask_token.expand(B, N, -1)
                x_full = torch.cat([x_full, mask_token], dim=1)
            
            else:
                x_vis = x_vis_list[i]
                bool_vis_pos = ~masks[i]
                mask_vis = mask_vis_list[i] # H_Encoder에서 전달받은 mask_vis 사용
                B, N, _ = center.shape
                _, _, C_vis = x_vis.shape # x_vis 차원
                C_full = x_full.shape[-1] # 이전 레이어 x_full 차원
                
                # x_vis (Encoder 출력)와 x_full (Decoder 이전 레이어)의 차원이 다를 수 있음
                # x_full_en은 x_vis와 동일한 차원(C_vis)을 가져야 함
                x_full_en = torch.zeros(B, N, C_vis).cuda() 
                x_full_en[bool_vis_pos] = x_vis[mask_vis]

                # token propagation
                if i == 1:
                    x_full = self.token_prop[i - 1](center, center_0, x_full_en, x_full)
                else:
                    x_full = self.token_prop[i - 1](center, centers[i - 1], x_full_en, x_full)
                pos_full = self.decoder_pos_embeds[i](center)

            x_full = self.h_decoder[i](x_full, pos_full)

        # reconstruction     
        x_full  = self.decoder_norm(x_full)
        B, N, C = x_full.shape
        x_rec = x_full[masks[-1]].reshape(-1, C) # H_Encoder에서 masks[-2]가 2번째 레벨 마스크임
        L, _ = x_rec.shape

        rec_points = self.rec_head(x_rec.unsqueeze(-1)).reshape(L, -1, 3)
        gt_points = neighborhoods[-1][masks[-1]].reshape(L, -1, 3) # neighborhoods[-2]가 2번째 레벨(원본)

        # CD loss
        loss = self.rec_loss(rec_points, gt_points)
        return loss


@MODELS.register_module()
class Point_M2AE_ReID(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_M2AE_ReID]', logger ='Point_M2AE_ReID')
        self.config = config

        # 1. 토크나이저 (Tokenizer)
        self.group_sizes = config.group_sizes
        self.num_groups = config.num_groups
        self.feat_dim = config.encoder_dims[-1]
        self.group_dividers = nn.ModuleList()
        for i in range(len(self.group_sizes)):
            self.group_dividers.append(Group(num_group=self.num_groups[i], group_size=self.group_sizes[i]))

        # 2. 계층적 인코더 (H_Encoder) - 백본
        self.h_encoder = H_Encoder(config)

        # 3. Norm 레이어
        self.norm = nn.LayerNorm(self.feat_dim)
        self.cls_dim = config.cls_dim # config에서 cls_dim 가져오기

        # PointTransformer와 동일한 Re-ID 헤드 구조 사용
        # 입력 차원은 self.feat_dim * 2 (Mean + Max 풀링)
        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.feat_dim * 2, 256), 
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )
        # ★★★ 수정 끝 ★★★

        # ★★★ [수정 2/4] 손실 함수 빌드 추가 ★★★
        self.build_loss_func()
        self.apply(self._init_weights) # 가중치 초기화 (헤드 부분)
        # ★★★ 수정 끝 ★★★

    # ★★★ [수정 3/4] 가중치 초기화, 손실 함수, 정확도 계산 함수 추가 ★★★
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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
    # ★★★ 수정 끝 ★★★
        # ★★★ ReID 모델은 CLS Head가 필요 없음 ★★★
        # self.cls_head_finetune = ... (이 부분을 삭제)

    def load_model_from_ckpt(self, ckpt_path):
        # Point_M2AE_ModelNet40의 load_model_from_ckpt 로직을 그대로 사용
        print_log(f'Loading pretrain model from {ckpt_path}', logger='Point_M2AE_ReID')
        state_dict = torch.load(ckpt_path)
        ckpt_key = 'model' if 'model' in state_dict else 'base_model'
        if ckpt_key not in state_dict:
             ckpt_state_dict = state_dict
        else:
             ckpt_state_dict = state_dict[ckpt_key]

        h_encoder_state_dict = {k: v for k, v in ckpt_state_dict.items() if k.startswith('h_encoder.')}
        if not h_encoder_state_dict:
            print_log(f'No weights starting with "h_encoder." found.', logger='Point_M2AE_ReID')
            return

        incompatible = self.load_state_dict(h_encoder_state_dict, strict=False)
        # ... (missing/unexpected key 로깅 로직) ...

    def forward(self, pts, **kwargs):
        # 1. 멀티스케일 그룹핑
        neighborhoods, centers, idxs = [], [], []
        for i in range(len(self.group_dividers)):
            if i == 0:
                neighborhood, center, idx = self.group_dividers[i](pts)
            else:
                neighborhood, center, idx = self.group_dividers[i](center)
            neighborhoods.append(neighborhood)
            centers.append(center)
            idxs.append(idx)

        # 2. 인코더 실행 (eval=True 모드)
        x_vis = self.h_encoder(neighborhoods, centers, idxs, eval=True)

        # 3. 풀링 및 특징 벡터 반환
        x_vis = self.norm(x_vis)
        concat_f = torch.cat([x_vis.mean(1), x_vis.max(1)[0]], dim=1) # Mean + Max

        # ★★★ ReID를 위해 분류 헤드 대신 특징 벡터(feature)를 반환 ★★★
        try:
            # ReID 피처 (256-dim)
            features = self.cls_head_finetune[:-1](concat_f) 
            # 분류 로짓 (cls_dim)
            logits = self.cls_head_finetune[-1](features)   
        except Exception as e:
            print(f"Warning: Point-M2AE_ReID Feature extraction failed. {e}")
            logits = self.cls_head_finetune(concat_f)
            features = concat_f # fallback

        log_softmax_out = F.log_softmax(logits, dim=-1)

        if kwargs.get('return_features', False):
            return log_softmax_out, features  # (logits, features) 튜플 반환
        return log_softmax_out
        # ★★★ 수정 끝 ★★★
