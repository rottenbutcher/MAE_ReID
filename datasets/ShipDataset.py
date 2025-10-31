"""Datasets for simulation and RealShip point cloud experiments."""

import os
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch
import torch.utils.data as data

from .build import DATASETS


def pc_normalize(pc: np.ndarray) -> np.ndarray:
    """Normalize a point cloud to zero-mean and unit sphere."""
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

        self.file_list = sorted([f for f in os.listdir(self.data_root) if f.endswith(".npy")])
        self.labels = {file_name: i for i, file_name in enumerate(self.file_list)}

        print(f"[DATASET] SimulationShip: {len(self.file_list)} files loaded.")

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_root, file_name)

        points = np.load(file_path).astype(np.float32)

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
        sample_id = os.path.splitext(file_name)[0]
        return "simulation_ship", sample_id, (points, label)

    def __len__(self):
        return len(self.file_list)


@DATASETS.register_module()
class RealShip(data.Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.npoints
        self.subset = config.subset
        self.class_whitelist = getattr(config, "class_whitelist", None)
        self.max_classes = getattr(config, "max_classes", None)

        candidate_names: List[str] = []
        subset_key = str(self.subset)
        if subset_key.endswith(".txt"):
            candidate_names.append(subset_key)
            stem = subset_key[:-4]
        else:
            stem = subset_key
            candidate_names.append(f"{stem}.txt")

        candidate_names.extend(
            [
                f"real_ship_{stem}.txt",
                f"reid_{stem}.txt",
                f"real_ship_cls_{stem}.txt",
            ]
        )
        if stem.startswith("reid_"):
            trimmed = stem.replace("reid_", "", 1)
            candidate_names.extend(
                [
                    f"{trimmed}.txt",
                    f"real_ship_{trimmed}.txt",
                ]
            )

        search_roots = [self.root, os.path.dirname(self.root)]
        list_file = None
        for name in candidate_names:
            for root_dir in search_roots:
                candidate_path = os.path.join(root_dir, name)
                if os.path.exists(candidate_path):
                    list_file = candidate_path
                    break
            if list_file is not None:
                break

        if list_file is None:
            searched = [os.path.join(root_dir, name) for root_dir in search_roots for name in candidate_names]
            raise FileNotFoundError(
                "Could not locate a split list for RealShip. Checked: " + ", ".join(sorted(set(searched)))
            )

        self.file_list: List[str] = []
        self.labels_from_file: Dict[str, int] = {}
        self.has_labels_in_file = False

        with open(list_file, "r", encoding="utf-8") as f:
            lines = [line.strip().replace("/", os.path.sep) for line in f.readlines() if line.strip()]

        if lines and " " in lines[0]:
            self.has_labels_in_file = True
            print(
                f"[DATASET] RealShip({self.subset}): Loading paths and labels from split file (Classification Mode)."
            )
            for line in lines:
                if " " not in line:
                    continue
                path, label_str = line.rsplit(" ", 1)
                self.file_list.append(path)
                self.labels_from_file[path] = int(label_str)
        else:
            self.has_labels_in_file = False
            print(f"[DATASET] RealShip({self.subset}): Loading paths from split file, deriving labels from folder (ReID Mode).")
            self.file_list = lines

        if self.class_whitelist is not None:
            if isinstance(self.class_whitelist, (list, tuple, set)):
                allowed = set(self.class_whitelist)
            else:
                raise TypeError("class_whitelist must be a sequence of class names")
        else:
            allowed = None

        if self.max_classes is not None:
            try:
                max_cls = int(self.max_classes)
            except (TypeError, ValueError) as exc:
                raise TypeError("max_classes must be convertible to int") from exc
            if max_cls <= 0:
                max_cls = None
        else:
            max_cls = None

        if allowed is not None or max_cls is not None:
            filtered_files = []
            seen_classes: List[str] = []
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
            raise RuntimeError("RealShip dataset received an empty file list after filtering.")

        unique_classes = {os.path.basename(os.path.dirname(path)) for path in self.file_list}
        self.class_map = {d: i for i, d in enumerate(sorted(unique_classes))}

        print(f"[DATASET] RealShip({self.subset}): {len(self.file_list)} files loaded.")
        if self.has_labels_in_file:
            print(f"Total {len(set(self.labels_from_file.values()))} classes loaded from file.")
        else:
            print(f"Class mapping (from folders): {self.class_map}")

    def __getitem__(self, index):
        relative_path = self.file_list[index]

        if self.has_labels_in_file:
            label = self.labels_from_file[relative_path]
        else:
            class_name = os.path.basename(os.path.dirname(relative_path))
            label = self.class_map[class_name]

        full_path = os.path.join(self.root, relative_path)

        try:
            raw_array = np.load(full_path)
        except FileNotFoundError:
            print(f"[ERROR] File not found: {full_path}")
            points = np.zeros((self.npoints, 3), dtype=np.float32)
            current_points = torch.from_numpy(points).float()
            return "RealShip", "error_sample", (current_points, 0)

        if raw_array.dtype.names:
            try:
                points = np.vstack([raw_array[axis] for axis in ("x", "y", "z")]).T.astype(np.float32)
            except ValueError:
                print(f"[Warning] Structured array at {full_path} has missing 'x', 'y', or 'z' fields.")
                points = np.zeros((self.npoints, 3), dtype=np.float32)
        else:
            points = raw_array.astype(np.float32)
            if points.ndim != 2:
                print(f"[Warning] Simple array at {full_path} has wrong dimensions: {points.shape}")
                points = np.zeros((self.npoints, 3), dtype=np.float32)
            elif points.shape[1] > 3:
                points = points[:, :3]
            elif points.shape[1] < 3:
                print(f"[Warning] Simple array at {full_path} has < 3 columns: {points.shape}")
                points = np.zeros((self.npoints, 3), dtype=np.float32)

        num_points_in_file = points.shape[0]
        if num_points_in_file == 0:
            print(f"[Warning] Empty point cloud at {full_path}. Using zeros.")
            points = np.zeros((self.npoints, 3), dtype=np.float32)
            num_points_in_file = self.npoints

        if num_points_in_file > self.npoints:
            indices = np.random.choice(num_points_in_file, self.npoints, replace=False)
            current_points = points[indices]
        elif num_points_in_file < self.npoints:
            indices = np.random.choice(num_points_in_file, self.npoints, replace=True)
            current_points = points[indices]
        else:
            current_points = points

        current_points = pc_normalize(current_points)
        current_points = torch.from_numpy(current_points).float()
        sample_id = os.path.splitext(os.path.basename(relative_path))[0]

        return "RealShip", sample_id, (current_points, label)

    def __len__(self):
        return len(self.file_list)


@DATASETS.register_module()
class SimRealValidation(data.Dataset):
    """Validation dataset that mixes simulation data with a subset of real-world classes."""

    def __init__(self, config):
        self.npoints = config.npoints

        include_sim = getattr(config, "include_simulation", True)
        include_real = getattr(config, "include_real", True)

        if not include_sim and not include_real:
            raise ValueError("SimRealValidation requires at least one of simulation or real data to be enabled.")

        self.datasets = {}
        self.indices = []

        self.sim_label_count = 0
        if include_sim:
            if not hasattr(config, "SIM_DATA_PATH"):
                raise ValueError("SIM_DATA_PATH must be provided when include_simulation is True.")
            sim_cfg = SimpleNamespace(DATA_PATH=config.SIM_DATA_PATH, npoints=self.npoints)
            self.datasets["sim"] = SimulationShip(sim_cfg)
            self.indices.extend([("sim", idx) for idx in range(len(self.datasets["sim"]))])
            self.sim_label_count = len(self.datasets["sim"].labels)

        if include_real:
            if not hasattr(config, "REAL_DATA_PATH"):
                raise ValueError("REAL_DATA_PATH must be provided when include_real is True.")

            real_subset = getattr(config, "real_subset", getattr(config, "subset", "val"))
            real_kwargs = {
                "DATA_PATH": config.REAL_DATA_PATH,
                "npoints": self.npoints,
                "subset": real_subset,
            }

            if hasattr(config, "real_class_whitelist"):
                real_kwargs["class_whitelist"] = config.real_class_whitelist
            if hasattr(config, "real_max_classes"):
                real_kwargs["max_classes"] = config.real_max_classes

            real_cfg = SimpleNamespace(**real_kwargs)
            self.datasets["real"] = RealShip(real_cfg)
            self.indices.extend([("real", idx) for idx in range(len(self.datasets["real"]))])
            self.real_label_offset = self.sim_label_count
            self.real_label_count = len(self.datasets["real"].class_map)
        else:
            self.real_label_offset = 0
            self.real_label_count = 0

        if len(self.indices) == 0:
            raise RuntimeError("SimRealValidation constructed an empty index list.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source, local_idx = self.indices[index]

        if source == "sim":
            return self.datasets["sim"][local_idx]

        taxonomy_id, model_id, (points, label) = self.datasets["real"][local_idx]
        if not torch.is_tensor(points):
            points = torch.from_numpy(points).float()
        else:
            points = points.float()
        label = int(label) + self.real_label_offset
        return taxonomy_id, model_id, (points, label)
