# /datasets/ReIDDataset.py

import os
import torch
import numpy as np
import random
import torch.utils.data as data
from collections import defaultdict

# 현재 프로젝트의 다른 모듈을 임포트
from .build import DATASETS
# ShipDataset에서 포인트 클라우드 로딩 및 정규화 함수 가져오기
from .ShipDataset import pc_normalize

@DATASETS.register_module()
class ReIDDataset(data.Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.subset = config.subset
        self.npoints = config.npoints

        # 1. subset에 따라 사용할 ID 목록 파일 결정
        if self.subset == 'train':
             id_list_file = os.path.join(os.path.dirname(self.root), 'reid_train_ids.txt')
        elif self.subset == 'val':
             id_list_file = os.path.join(os.path.dirname(self.root), 'reid_val_ids.txt')
        elif self.subset == 'test':
             id_list_file = os.path.join(os.path.dirname(self.root), 'reid_test_ids.txt')
        else:
            raise ValueError(f"Unknown subset: {self.subset}")

        print(f"[{self.subset.upper()} SET] Loading IDs from: {id_list_file}")
        with open(id_list_file, 'r') as f:
            self.ids_in_subset = [line.strip() for line in f.readlines()]

        # 2. 파일 목록 생성 및 ID(label) 매핑
        self.file_list = [] # 각 파일의 (filepath, label) 튜플 저장
        self.id_map = {id_name: i for i, id_name in enumerate(sorted(self.ids_in_subset))} # ID 문자열 -> 숫자 레이블
        self.label_to_indices = defaultdict(list) # 숫자 레이블 -> 해당 레이블을 가진 file_list 인덱스 목록

        print(f"[{self.subset.upper()} SET] Loading files for {len(self.ids_in_subset)} IDs...")
        for id_name in self.ids_in_subset:
            label = self.id_map[id_name]
            class_path = os.path.join(self.root, id_name)
            if os.path.isdir(class_path):
                for filename in os.listdir(class_path):
                    if filename.endswith('.npy'):
                        filepath = os.path.join(class_path, filename)
                        current_index = len(self.file_list)
                        self.file_list.append((filepath, label))
                        self.label_to_indices[label].append(current_index)

        print(f'[DATASET] ReIDDataset({self.subset}): {len(self.file_list)} files loaded for {len(self.ids_in_subset)} IDs.')
        print(f'ID mapping: {self.id_map}')


    def _load_npy_point_cloud(self, file_path):
        """ Helper function to load point cloud from .npy file, handling structured arrays """
        try:
            data = np.load(file_path)
            if data.dtype.names and all(name in data.dtype.names for name in ['x', 'y', 'z']):
                points = np.vstack([data['x'], data['y'], data['z']]).T.astype(np.float32)
            else:
                points = data.astype(np.float32)
            return points
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None # 오류 발생 시 None 반환

    def _get_point_cloud_tensor(self, index):
        """ Loads, samples, normalizes, and converts a point cloud to tensor """
        filepath, label = self.file_list[index]
        points = self._load_npy_point_cloud(filepath)

        if points is None: # 로딩 실패 시 처리
            return None, None

        # 샘플링 로직
        num_points_in_file = points.shape[0]
        if num_points_in_file > self.npoints:
            indices = np.random.choice(num_points_in_file, self.npoints, replace=False)
            current_points = points[indices]
        elif num_points_in_file < self.npoints:
            indices = np.random.choice(num_points_in_file, self.npoints, replace=True)
            current_points = points[indices]
        else:
            current_points = points

        current_points = pc_normalize(current_points)
        return torch.from_numpy(current_points).float(), label


    def __getitem__(self, index):
        # 1. Anchor 샘플 가져오기
        anchor_pc, anchor_label = self._get_point_cloud_tensor(index)
        if anchor_pc is None: # 로딩 실패 시 더미 데이터 반환 또는 에러 처리
             # 간단하게 첫번째 샘플을 다시 로드 (실제 구현에서는 더 나은 방식 고려)
             anchor_pc, anchor_label = self._get_point_cloud_tensor(0)
             print(f"Warning: Failed to load anchor sample at index {index}, using index 0 instead.")


        # 2. Positive 샘플 선택 (Anchor와 같은 ID, 다른 샘플)
        positive_indices = self.label_to_indices[anchor_label]
        # Anchor 자신을 제외한 인덱스 목록 생성
        possible_positive_indices = [i for i in positive_indices if i != index]

        if len(possible_positive_indices) > 0:
            positive_index = random.choice(possible_positive_indices)
        else:
            # 같은 ID의 다른 샘플이 없으면 Anchor 자신을 Positive로 사용 (경고 출력)
            # print(f"Warning: No different positive sample found for index {index} (label {anchor_label}). Using anchor itself.")
            positive_index = index

        positive_pc, _ = self._get_point_cloud_tensor(positive_index)
        if positive_pc is None:
             positive_pc, _ = self._get_point_cloud_tensor(0) # 로딩 실패 처리
             print(f"Warning: Failed to load positive sample at index {positive_index}, using index 0 instead.")


        # 3. Negative 샘플 선택 (Anchor와 다른 ID)
        negative_label = anchor_label
        # 다른 레이블이 나올 때까지 반복해서 랜덤 선택
        while negative_label == anchor_label:
            negative_label = random.choice(list(self.label_to_indices.keys()))

        negative_index = random.choice(self.label_to_indices[negative_label])
        negative_pc, _ = self._get_point_cloud_tensor(negative_index)
        if negative_pc is None:
             negative_pc, _ = self._get_point_cloud_tensor(0) # 로딩 실패 처리
             print(f"Warning: Failed to load negative sample at index {negative_index}, using index 0 instead.")


        # Triplet (Anchor, Positive, Negative) 텐서와 Anchor 레이블 반환
        return anchor_pc, positive_pc, negative_pc, anchor_label


    def __len__(self):
        # 데이터셋의 전체 길이는 Anchor로 사용될 수 있는 샘플의 총 개수
        return len(self.file_list)