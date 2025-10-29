import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
import json
from pathlib import Path

import numpy as np
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from torchvision import transforms

from datasets.data_transforms import PointcloudViewpointMasking




train_transforms = transforms.Compose(
    [
         # data_transforms.PointcloudScale(),
         # data_transforms.PointcloudRotate(),
         # data_transforms.PointcloudTranslate(),
         # data_transforms.PointcloudJitter(),
         # data_transforms.PointcloudRandomInputDropout(),
         # data_transforms.RandomHorizontalFlip(),
         data_transforms.PointcloudScaleAndTranslate(),
    ]
)

test_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        # data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudTranslate(),
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)


def _update_reid_metrics_file(experiment_path, epoch, rank1, rank5, mAP, is_best=False):
    """Persist validation metrics so external tools can read them."""

    exp_dir = Path(experiment_path)
    exp_dir.mkdir(parents=True, exist_ok=True)
    summary_path = exp_dir / 'reid_metrics.json'

    payload = {"history": []}
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            payload = {"history": []}

    payload.setdefault("history", [])
    payload["history"].append({
        "epoch": int(epoch),
        "rank1": float(rank1),
        "rank5": float(rank5),
        "mAP": float(mAP),
        "is_best": bool(is_best),
        "timestamp": time.time(),
    })

    if is_best or "best" not in payload:
        payload["best"] = {
            "epoch": int(epoch),
            "rank1": float(rank1),
            "rank5": float(rank5),
            "mAP": float(mAP),
        }

    summary_path.write_text(json.dumps(payload, indent=2))


# --- [REID] Acc_Metric 클래스를 mAP와 Rank-1을 저장하도록 수정 ---
class Acc_Metric:
    def __init__(self, mAP=0., rank1=0., rank5=0.):
        if type(mAP).__name__ == 'dict':
            self.mAP = mAP.get('mAP', 0.)
            self.rank1 = mAP.get('rank1', 0.)
            self.rank5 = mAP.get('rank5', 0.)
        elif type(mAP).__name__ == 'Acc_Metric':
            self.mAP = mAP.mAP
            self.rank1 = mAP.rank1
            self.rank5 = mAP.rank5
        else:
            self.mAP = mAP
            self.rank1 = rank1
            self.rank5 = rank5

    def better_than(self, other):
        # Rank-1 정확도를 우선 기준으로 비교
        if self.rank1 > other.rank1:
            return True
        # Rank-1이 같다면 mAP로 비교
        elif abs(self.rank1 - other.rank1) < 1e-6 and self.mAP > other.mAP:
            return True
        else:
            return False

    def state_dict(self):
        return {'mAP': self.mAP, 'rank1': self.rank1, 'rank5': self.rank5}
# --- [REID] 수정 끝 ---


# --- [REID] mAP, Rank-k 계산을 위한 헬퍼 함수 추가 ---
@torch.no_grad()
def calculate_reid_metrics(all_features, all_labels):
    """ 
    추출된 모든 피처와 레이블을 사용하여 ReID 메트릭(mAP, Rank-k)을 계산합니다.
    (mAP 수동 계산 로직으로 수정됨)
    """
    # GPU를 사용하여 거리 계산 가속
    all_features = all_features.cuda()
    all_features = F.normalize(all_features, p=2, dim=1)
    all_labels = all_labels.cuda()
    
    N = all_features.size(0)
    
    # (N, N) L2 거리 행렬 계산
    dist_matrix = torch.cdist(all_features, all_features, p=2)
    
    # (N, N) 레이블 일치 여부 행렬 (True/False)
    labels_equal = all_labels.unsqueeze(0) == all_labels.unsqueeze(1)
    
    # --- 각 쿼리별 총 정답 개수 계산 (자기 자신 제외) ---
    # (N, N) -> (N,)
    num_positives = labels_equal.sum(dim=1) - 1 # 1 (자기 자신) 빼기
    num_positives = torch.clamp(num_positives, min=0) # 0개 미만 방지

    # 거리 기준 오름차순 정렬 (가까운 순)
    indices = dist_matrix.argsort(dim=1) # (N, N)
    
    # 정렬된 인덱스에 해당하는 레이블 일치 여부
    sorted_labels = torch.gather(labels_equal, 1, indices) # (N, N)
    
    # --- Rank-k 계산 (기존과 동일) ---
    # 자기 자신(index 0)을 제외
    sorted_labels_no_self = sorted_labels[:, 1:] # (N, N-1)
    
    rank1_acc = (sorted_labels_no_self[:, 0]).float().mean()
    rank5_acc = (sorted_labels_no_self[:, :5].any(dim=1)).float().mean()
    
    # --- mAP 수동 계산 ---
    
    # (N, N-1) tensor: [True, False, True, False, ...]
    relevance_mask = sorted_labels_no_self
    relevance_mask_float = relevance_mask.float()
    
    # (N, N-1) tensor: [1, 1, 2, 2, ...] (정답 누적 합)
    cumulative_positives = torch.cumsum(relevance_mask_float, dim=1)
    
    # (N, N-1) tensor: [1, 2, 3, 4, ...] (K 값)
    k_values = torch.arange(1, N, device=all_features.device).unsqueeze(0).expand(N, -1)
    
    # (N, N-1) tensor: [1/1, 1/2, 2/3, 2/4, ...] (Precision@K)
    precision_at_k = cumulative_positives / k_values.float()
    
    # (N, N-1) tensor: [P@1*Rel@1, P@2*Rel@2, ...]
    # (정답인 위치(Rel@k=1)의 Precision@K 값만 남김)
    precision_times_relevance = precision_at_k * relevance_mask_float
    
    # (N,) tensor: [Sum(P@k*Rel@k) for each query]
    sum_precision_times_relevance = precision_times_relevance.sum(dim=1)
    
    # (N,) tensor: AP for each query
    # AP = Sum(P@k*Rel@k) / Total_Positives
    ap_per_query = torch.zeros(N, device=all_features.device)
    
    # 정답이 0개인 쿼리(num_positives=0)는 AP가 0이므로, 0으로 나누기 방지
    has_positives = num_positives > 0
    ap_per_query[has_positives] = sum_precision_times_relevance[has_positives] / num_positives[has_positives]
    
    # mAP = Mean(AP)
    mAP = ap_per_query.mean().item() * 100 # (0-100 스케일)

    return rank1_acc.item() * 100, rank5_acc.item() * 100, mAP


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)

    if args.use_viewpoint_mask:
        # ( ... 기존 코드 ... )
        print_log('Using Viewpoint Masking for data augmentation.', logger=logger)
    else:
        # ( ... 기존 코드 ... )
        print_log('Using default ScaleAndTranslate transforms for training.', logger=logger)

    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    base_model = builder.model_builder(config.model)
    
    start_epoch = 0
    
    # --- [REID] Metric 초기화 수정 ---
    best_metrics = Acc_Metric(mAP=0., rank1=0.)
    best_metrics_vote = Acc_Metric(mAP=0., rank1=0.)
    metrics = Acc_Metric(mAP=0., rank1=0.)
    # --- [REID] 수정 끝 ---

    if args.resume:
        # --- [REID] resume 로직 수정 ---
        start_epoch, best_metrics_state = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metrics_state) # dict로부터 Acc_Metric 객체 생성
        # --- [REID] 수정 끝 ---
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger = logger)

    if args.use_gpu:    
        base_model.to(args.local_rank)

    # ( ... DDP, Sync BN 관련 코드 ... )
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    
    # ( ... Encoder Freeze 관련 코드 ... )
    encoder_is_frozen = True
    if args.unfreeze_threshold > 0:
        print_log('Encoder is initially frozen.', logger=logger)
        for param in base_model.module.encoder.parameters():
            param.requires_grad = False
    else:
        encoder_is_frozen = False # 기능 비활성화

    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        
        # --- [REID] losses 미터기 수정 (mAP, Rank1은 배치 단위로 계산 불가) ---
        losses = AverageMeter(['loss', 'train_acc']) # Train 정확도만 추적
        # --- [REID] 수정 끝 ---
        
        num_iter = 0
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)

        npoints = config.npoints
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            
            points = data[0].cuda()
            label = data[1].cuda()

            # ( ... Npoints 샘플링 코드 ... )
            num_points_in_file = points.size(1)
            if num_points_in_file < npoints:
                indices = np.random.choice(num_points_in_file, npoints, replace=True)
                points = points[:, indices, :]
            elif num_points_in_file > npoints:
                indices = np.random.choice(num_points_in_file, npoints, replace=False)
                points = points[:, indices, :]

            points = train_transforms(points)

            forward_out = base_model(points, return_features=True)
            if isinstance(forward_out, tuple) and len(forward_out) == 2:
                log_softmax_out, features = forward_out
            else:
                raise ValueError('Model forward must return (logits, features) when return_features=True')

            ret = log_softmax_out # 첫 번째 값 사용
            loss, acc = base_model.module.get_loss_acc(ret, label, features=features)
            # --- [REID] 수정 끝 ---

            _loss = loss
            _loss.backward()

            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                acc = dist_utils.reduce_tensor(acc, args)
                losses.update([loss.item(), acc.item()])
            else:
                losses.update([loss.item(), acc.item()])

            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/TrainAcc', acc.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()
        
        # ( ... 스케줄러 스텝 ... )
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/TrainAcc', losses.avg(1), epoch) # TrainAcc

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],optimizer.param_groups[0]['lr']), logger = logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            # --- [REID] validate 함수는 이제 ReID 메트릭을 반환 (Acc_Metric) ---
            metrics = validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)
            _update_reid_metrics_file(
                args.experiment_path,
                epoch,
                metrics.rank1,
                metrics.rank5,
                metrics.mAP,
                is_best=False,
            )
            # --- [REID] 수정 끝 ---

            # --- [REID] Encoder 동결 해제 로직 수정 (acc -> rank1) ---
            # unfreeze_threshold를 Rank-1 기준으로 사용 (e.g., 50.0)
            if encoder_is_frozen and metrics.rank1 > args.unfreeze_threshold:
                print_log(f'Rank-1 {metrics.rank1:.4f} > threshold {args.unfreeze_threshold:.4f}. Unfreezing encoder.', logger=logger)
            # ( ... 동결 해제 로직 ... )
                for param in base_model.module.encoder.parameters():
                    param.requires_grad = True
                print_log(f'Rebuilding optimizer with new learning rate: {args.lr_finetune}', logger=logger)
                config.optimizer.kwargs.lr = args.lr_finetune
                optimizer, scheduler = builder.build_opti_sche(base_model, config)
                encoder_is_frozen = False
            # --- [REID] 수정 끝 ---

            better = metrics.better_than(best_metrics)
            if better:
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
                print_log("--------------------------------------------------------------------------------------------", logger=logger)
                _update_reid_metrics_file(
                    args.experiment_path,
                    epoch,
                    metrics.rank1,
                    metrics.rank5,
                    metrics.mAP,
                    is_best=True,
                )
            
            # --- [REID] VOTE 기능 비활성화 (ReID와 호환되지 않음) ---
            # if args.vote:
            #     if metrics.acc > 92.1 or (better and metrics.acc > 91):
            #         metrics_vote = validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)
            #         if metrics_vote.better_than(best_metrics_vote):
            #             best_metrics_vote = metrics_vote
            #             print_log(
            #                 "****************************************************************************************",
            #                 logger=logger)
            #             builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics_vote, 'ckpt-best_vote', args, logger = logger)
            # --- [REID] 비활성화 끝 ---

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

# --- [REID] validate 함수를 ReID 평가 로직으로 (mAP, Rank-k) 전면 교체 ---
def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger = None):
    # print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    all_features = []
    all_labels = []
    npoints = config.npoints
    
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            forward_out = base_model(points, return_features=True)
            if isinstance(forward_out, tuple) and len(forward_out) == 2:
                log_softmax_out, features = forward_out
            else:
                raise ValueError('Model forward must return (logits, features) when return_features=True')
            target = label.view(-1)

            all_features.append(features.detach()) # GPU에 유지
            all_labels.append(target.detach())     # GPU에 유지

        # 모든 피처와 레이블을 하나로 합침
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        if args.distributed:
            all_features = dist_utils.gather_tensor(all_features, args)
            all_labels = dist_utils.gather_tensor(all_labels, args)

        # ReID 메트릭 계산 (GPU에서)
        rank1, rank5, mAP = calculate_reid_metrics(all_features, all_labels)
        
        print_log(f'[Validation] EPOCH: {epoch}  Rank-1 = {rank1:.4f}  Rank-5 = {rank5:.4f}  mAP = {mAP:.4f}', logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/mAP', mAP, epoch)
        val_writer.add_scalar('Metric/Rank-1', rank1, epoch)
        val_writer.add_scalar('Metric/Rank-5', rank5, epoch)

    metrics = Acc_Metric(mAP=mAP, rank1=rank1, rank5=rank5)
    return metrics
# --- [REID] validate 함수 교체 완료 ---


# validate_vote 함수는 ReID용으로 수정되지 않았으므로 그대로 둡니다.
# (run_net에서 호출 코드를 비활성화했기 때문에 실행되지 않습니다.)


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts, logger = logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    if args.distributed:
        raise NotImplementedError()
     
    # --- [REID] test 함수를 ReID 평가용으로 호출 ---
    test(base_model, test_dataloader, args, config, logger=logger)
    # --- [REID] 수정 끝 ---


def test_net_reid(args, config, logger = None):
    """ReID(mAP, Rank-1) 테스트 전용 함수"""
    
    if logger is None:
        logger = get_logger(args.log_name)
    print_log('ReID Tester start ... ', logger = logger)
    
    # 데이터셋 빌드 (test_dataloader)
    # [수정] config.dataset.test를 사용하도록 명시
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    
    # 모델 빌드 및 체크포인트 로드
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts, logger = logger)
    
    if args.use_gpu:
        base_model.to(args.local_rank)
    if args.distributed:
        raise NotImplementedError()
    base_model = nn.DataParallel(base_model).cuda()
    
    base_model.eval()  # set model to eval mode

    all_features = []
    all_labels = []
    npoints = config.npoints

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()
            points = misc.fps(points, npoints)

            # 피처 추출
            forward_out = base_model(points, return_features=True)
            if isinstance(forward_out, tuple) and len(forward_out) == 2:
                log_softmax_out, features = forward_out
            else:
                raise ValueError('Model forward must return (logits, features) when return_features=True')
            target = label.view(-1)
            all_features.append(features.detach())
            all_labels.append(target.detach())

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # ReID 메트릭 계산 (mAP 수동 계산 로직이 포함된 함수)
    rank1, rank5, mAP = calculate_reid_metrics(all_features, all_labels)
    
    print_log(f'[TEST] Rank-1 = {rank1:.4f}  Rank-5 = {rank5:.4f}  mAP = {mAP:.4f}', logger=logger)
# --- [REID] test 함수를 ReID 평가 로직으로 (mAP, Rank-k) 전면 교체 ---


        # --- [REID] VOTE 테스트 비활성화 ---
        # print_log(f"[TEST_VOTE]", logger = logger)
        # acc = 0.
        # for time in range(1, 300):
        #     this_acc = test_vote(base_model, test_dataloader, 1, None, args, config, logger=logger, times=10)
        #     if acc < this_acc:
        #         acc = this_acc
        #     print_log('[TEST_VOTE_time %d]  acc = %.4f, best acc = %.4f' % (time, this_acc, acc), logger=logger)
        # print_log('[TEST_VOTE] acc = %.4f' % acc, logger=logger)
        # --- [REID] 비활성화 끝 ---
# --- [REID] test 함수 교체 완료 ---

def test_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger = None, times = 10):

    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
                
            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)

            fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)
            local_pred = []

            for kk in range(times):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(), 
                                                        fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

                points = test_transforms(points)

                logits = base_model(points)
                target = label.view(-1)

                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)


            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)
    # print_log('[TEST] acc = %.4f' % acc, logger=logger)
    
    return acc
