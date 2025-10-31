#!/usr/bin/env python3
"""
tools/create_reid_splits.py

data/Ship_Real_ext 폴더 내의 클래스(ID) 폴더를 기반으로
ReID(재식별)용 train/val/test 파일 목록(.txt)을 생성합니다.

ReID에서는 동일한 ID가 여러 세트에 걸쳐 존재하지 않도록,
파일 단위가 아닌 ID(폴더) 단위로 세트를 분할합니다.
"""

import argparse
import random
from pathlib import Path
import os

def main():
    parser = argparse.ArgumentParser(description='Generate ReID split .txt files from class folders.')
    parser.add_argument(
        '--data-root',
        required=True,
        type=Path,
        help='Path to the real-world dataset root (e.g., data/Ship_Real_ext).',
    )
    parser.add_argument(
        '--train-ids',
        type=int,
        default=19,
        help='Number of identities (folders) to use for the training set.',
    )
    parser.add_argument(
        '--val-ids',
        type=int,
        default=3,
        help='Number of identities (folders) to use for the validation set.',
    )
    # 나머지 ID는 자동으로 'test' 세트가 됩니다.
    parser.add_argument(
        '--output-prefix',
        default='reid',
        help='Prefix for the output files (e.g., "reid" -> reid_train.txt).',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for shuffling identities.',
    )
    parser.add_argument(
        '--extension',
        default='.npy',
        type=str,
        help='File extension to include (default: .npy).',
    )
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f'Dataset root does not exist: {data_root}')

    # 1. data_root에서 모든 클래스(ID) 폴더를 찾습니다.
    try:
        all_identities = sorted([d for d in data_root.iterdir() if d.is_dir()])
    except Exception as e:
        print(f"Error reading directories from {data_root}: {e}")
        return

    if not all_identities:
        raise RuntimeError(f'No class/identity folders found in {data_root}')

    num_total_ids = len(all_identities)
    print(f'Found {num_total_ids} total identities (classes) in {data_root}')

    if args.train_ids + args.val_ids > num_total_ids:
        raise ValueError(
            f'Train IDs ({args.train_ids}) + Val IDs ({args.val_ids}) '
            f'exceeds total IDs ({num_total_ids}).'
        )

    # 2. ID 목록을 섞습니다.
    random.seed(args.seed)
    random.shuffle(all_identities)

    # 3. ID를 train, val, test 세트로 분할합니다.
    split_point_1 = args.train_ids
    split_point_2 = args.train_ids + args.val_ids

    id_splits = {
        'train': all_identities[:split_point_1],
        'val': all_identities[split_point_1:split_point_2],
        'test': all_identities[split_point_2:],
    }

    # 4. 각 세트에 대해 파일 목록을 생성하고 .txt 파일에 씁니다.
    for split_name, id_list in id_splits.items():
        output_filename = f'{args.output_prefix}_{split_name}.txt'
        output_path = data_root / output_filename
        
        file_count = 0
        print(f'\nGenerating {output_path} ({len(id_list)} IDs)...')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for class_path in sorted(id_list):
                class_name = class_path.name
                files_in_class = 0
                # 각 ID 폴더 내의 모든 .npy 파일을 재귀적으로 찾습니다.
                for npy_file in class_path.rglob(f'*{args.extension}'):
                    # README.md 형식에 맞게 data_root 기준 상대 경로를 씁니다.
                    # 예: reid_val/FP_target4/scan.npy (X) -> FP_target4/scan.npy (O)
                    # RealShip.py는 data_root + subset_file_path로 조합합니다.
                    # [수정] README와 RealShip.py 동작을 다시 확인:
                    # RealShip.py는 list_file을 읽고, 각 줄을 os.path.join(self.root, relative_path)로 엽니다.
                    # README의 예시는 'reid_val/FP_target1/sample.npy'입니다.
                    # 하지만 'tools/create_split_txt.py'는 subset 폴더 *안에서* 실행되어
                    # 'FP_target1/sample.npy' 같은 경로를 생성할 수도 있습니다.
                    
                    # 'datasets/RealShip.py'는
                    # config.subset (예: 'reid_train.txt')을 읽고,
                    # 파일 안의 경로(예: 'FP_target1/scan.npy')를 self.root (예: 'data/Ship_Real_ext')와 조합합니다.
                    # 따라서 .txt 파일에는 'FP_target1/scan.npy' 형태의 *클래스 폴더부터 시작하는* 상대 경로가 필요합니다.
                    
                    try:
                        relative_path = npy_file.relative_to(data_root).as_posix()
                        f.write(relative_path + '\n')
                        file_count += 1
                        files_in_class += 1
                    except ValueError:
                        print(f"Warning: File {npy_file} is not under data root {data_root}?")
                
                if files_in_class == 0:
                    print(f"Warning: No '{args.extension}' files found for ID: {class_name}")

        print(f'Wrote {file_count} file paths to {output_path}')

    print(f"\nSuccessfully generated split files in {data_root}")
    print(f" - Train ({len(id_splits['train'])} IDs): {args.output_prefix}_train.txt")
    print(f" - Val   ({len(id_splits['val'])} IDs): {args.output_prefix}_val.txt")
    print(f" - Test  ({len(id_splits['test'])} IDs): {args.output_prefix}_test.txt")


if __name__ == '__main__':
    main()
