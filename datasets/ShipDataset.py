# /datasets/ShipDataset.py

import os
from types import SimpleNamespace

import torch
import numpy as np
import torch.utils.data as data

from .build import DATASETS
def pc_normalize(pc):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

@DATASETS.register_module()
class SimulationShip(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.npoints = config.npoints
        
        # Simulation_Ship 폴더 내의 모든 .npy 파일을 읽어옵니다.
        self.file_list = sorted([f for f in os.listdir(self.data_root) if f.endswith('.npy')])
        
        # 파일 이름 순서를 기준으로 0부터 699까지의 고유한 레이블을 생성합니다.
        self.labels = {file_name: i for i, file_name in enumerate(self.file_list)}
        
        print(f'[DATASET] SimulationShip: {len(self.file_list)} files loaded.')
        self.permutation = np.arange(self.npoints)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_root, file_name)
        
        points = np.load(file_path).astype(np.float32)

        # npoints보다 많으면 샘플링, 적으면 그대로 사용
        num_points_in_file = points.shape[0]
        if num_points_in_file > self.npoints:
            indices = np.random.choice(num_points_in_file, self.npoints, replace=False)
            points = points[indices]
        elif num_points_in_file < self.npoints:
            indices = np.random.choice(num_points_in_file, self.npoints, replace=True)
            points = points[indices]

        points = pc_normalize(points)
        points = torch.from_numpy(points).float()
        
        label = self.labels[file_name]
        # Cross-view 학습을 지원하기 위해 데이터셋 내부에서 마스크를 생성하지 않습니다.
        # 모든 인코더는 단일 포인트 클라우드만 입력으로 사용하며, 마스킹은 transform 또는 모델에서 처리됩니다.
        return 'simulation_ship', os.path.splitext(file_name)[0], (points, label)

    def __len__(self):
        return len(self.file_list)

@DATASETS.register_module()
class RealShip(data.Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.npoints
        self.subset = config.subset
        self.class_whitelist = getattr(config, 'class_whitelist', None)
        self.max_classes = getattr(config, 'max_classes', None)

        list_file = os.path.join(os.path.dirname(self.root), f'real_ship_{self.subset}.txt')
        if not os.path.exists(list_file):
            raise FileNotFoundError(f"List file not found: {list_file}")

        with open(list_file, 'r') as f:
            # (e.g., 'target1/file.npy', 'FP_target20/file.npy')
            self.file_list = [line.strip().replace('/', os.path.sep) for line in f.readlines()]

        # 1-1. 클래스 화이트리스트가 지정된 경우 필터링합니다.
        if self.class_whitelist is not None:
            if isinstance(self.class_whitelist, (list, tuple, set)):
                allowed = set(self.class_whitelist)
            else:
                raise TypeError("class_whitelist must be a sequence of class names")
        else:
            allowed = None

        # 1-2. 사용할 클래스 수를 제한할 수 있습니다.
        if self.max_classes is not None:
            try:
                max_cls = int(self.max_classes)
            except (TypeError, ValueError):
                raise TypeError("max_classes must be convertible to int")
            if max_cls <= 0:
                max_cls = None
        else:
            max_cls = None

        if allowed is not None or max_cls is not None:
            filtered_files = []
            seen_classes = []
            for rel_path in self.file_list:
                class_name = os.path.basename(os.path.dirname(rel_path))
                if allowed is not None and class_name not in allowed:
                    continue
                if max_cls is not None and class_name not in seen_classes:
                    if len(seen_classes) >= max_cls:
                        continue
                    seen_classes.append(class_name)
                filtered_files.append(rel_path)
            if filtered_files:
                self.file_list = filtered_files

        if len(self.file_list) == 0:
            raise RuntimeError('RealShip dataset received an empty file list after filtering.')

        # 2. 로드한 file_list에서 실제 사용되는 클래스(폴더명)만 추출합니다.
        unique_classes = set()
        for path in self.file_list:
            class_name = os.path.basename(os.path.dirname(path))
            unique_classes.add(class_name)

        # 3. 이 subset에만 해당하는 클래스로 class_map을 생성합니다 (e.g., 0 ~ 18)
        self.class_map = {d: i for i, d in enumerate(sorted(list(unique_classes)))}

        print(f'[DATASET] RealShip({self.subset}): {len(self.file_list)} files loaded.')
        print(f'Class mapping: {self.class_map}')


    def __getitem__(self, index):
        relative_path = self.file_list[index]
        
        # 파일 경로에서 클래스 이름(폴더명)을 추출하여 레이블로 사용
        class_name = os.path.basename(os.path.dirname(relative_path))
        label = self.class_map[class_name]

        full_path = os.path.join(self.root, relative_path)
        
        # 1. 일단 .npy 파일을 로드합니다.
        raw_array = np.load(full_path)
        
        # 2. dtype.names가 있는지 확인하여 구조화된 배열인지 단순 배열인지 판별합니다.
        if raw_array.dtype.names:
            # Case 1: 구조화된 배열 (e.g., [('x', ...), ('y', ...), ('attribute', ...)])
            try:
                # 'x', 'y', 'z' 필드를 추출합니다.
                points = np.vstack([raw_array['x'], raw_array['y'], raw_array['z']]).T.astype(np.float32)
            except ValueError:
                # (예외 처리) 'x', 'y', 'z' 필드가 없는 경우
                print(f"[Warning] Structured array at {full_path} has missing 'x', 'y', or 'z' fields.")
                # 임시로 0점 처리 (학습은 계속되도록)
                points = np.zeros((self.npoints, 3), dtype=np.float32)
        else:
            # Case 2: 단순 배열 (e.g., (N, 3) 또는 (N, 6))
            points = raw_array.astype(np.float32)
            
            # (안전 장치) (N, 3) 형태가 아닐 경우
            if points.ndim != 2:
                print(f"[Warning] Simple array at {full_path} has wrong dimensions: {points.shape}")
                points = np.zeros((self.npoints, 3), dtype=np.float32)
            elif points.shape[1] > 3:
                # (N, 6) 같은 경우, 앞의 3개(x,y,z)만 사용
                points = points[:, :3]
            elif points.shape[1] < 3:
                print(f"[Warning] Simple array at {full_path} has < 3 columns: {points.shape}")
                points = np.zeros((self.npoints, 3), dtype=np.float32)
        # 샘플링 및 정규화
        num_points_in_file = points.shape[0]
        if num_points_in_file > self.npoints:
            # 점이 많으면 npoints만큼 무작위 샘플링
            indices = np.random.choice(num_points_in_file, self.npoints, replace=False)
            current_points = points[indices]
        elif num_points_in_file < self.npoints:
            # 점이 부족하면 중복을 허용하여 npoints만큼 샘플링
            indices = np.random.choice(num_points_in_file, self.npoints, replace=True)
            current_points = points[indices]
        else:
            # 점 개수가 정확히 맞으면 그대로 사용
            current_points = points
        
        current_points = pc_normalize(current_points)
        current_points = torch.from_numpy(current_points).float()
        sample_id = os.path.splitext(os.path.basename(relative_path))[0]

        return 'RealShip', sample_id, (current_points, label)

    def __len__(self):
        return len(self.file_list)


@DATASETS.register_module()
class SimRealValidation(data.Dataset):
    """Validation dataset that mixes simulation data with a subset of real-world classes."""

    def __init__(self, config):
        self.npoints = config.npoints

        include_sim = getattr(config, 'include_simulation', True)
        include_real = getattr(config, 'include_real', True)

        if not include_sim and not include_real:
            raise ValueError('SimRealValidation requires at least one of simulation or real data to be enabled.')

        self.datasets = {}
        self.indices = []

        self.sim_label_count = 0
        if include_sim:
            if not hasattr(config, 'SIM_DATA_PATH'):
                raise ValueError('SIM_DATA_PATH must be provided when include_simulation is True.')
            sim_cfg = SimpleNamespace(DATA_PATH=config.SIM_DATA_PATH, npoints=self.npoints)
            self.datasets['sim'] = SimulationShip(sim_cfg)
            self.indices.extend([('sim', idx) for idx in range(len(self.datasets['sim']))])
            self.sim_label_count = len(self.datasets['sim'].labels)

        if include_real:
            if not hasattr(config, 'REAL_DATA_PATH'):
                raise ValueError('REAL_DATA_PATH must be provided when include_real is True.')

            real_subset = getattr(config, 'real_subset', getattr(config, 'subset', 'val'))
            real_kwargs = {
                'DATA_PATH': config.REAL_DATA_PATH,
                'npoints': self.npoints,
                'subset': real_subset,
            }

            if hasattr(config, 'real_class_whitelist'):
                real_kwargs['class_whitelist'] = config.real_class_whitelist
            if hasattr(config, 'real_max_classes'):
                real_kwargs['max_classes'] = config.real_max_classes

            real_cfg = SimpleNamespace(**real_kwargs)
            self.datasets['real'] = RealShip(real_cfg)
            self.indices.extend([('real', idx) for idx in range(len(self.datasets['real']))])
            self.real_label_offset = self.sim_label_count
            self.real_label_count = len(self.datasets['real'].class_map)
        else:
            self.real_label_offset = 0
            self.real_label_count = 0

        if len(self.indices) == 0:
            raise RuntimeError('SimRealValidation constructed an empty index list.')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source, local_idx = self.indices[index]

        if source == 'sim':
            return self.datasets['sim'][local_idx]

        taxonomy_id, model_id, (points, label) = self.datasets['real'][local_idx]
        if not torch.is_tensor(points):
            points = torch.from_numpy(points).float()
        else:
            points = points.float()
        label = int(label) + self.real_label_offset
        return taxonomy_id, model_id, (points, label)
