# split_data.py
import os
import random

# Real_Ship 데이터가 있는 경로
data_root = 'data/Ship_Real_ext'
train_ratio = 0.8  # 80%를 훈련용으로 사용

all_files = []
# 'target1', 'FP_target14'와 같은 모든 클래스 폴더를 순회합니다.
for class_folder in os.listdir(data_root):
    class_path = os.path.join(data_root, class_folder)
    if os.path.isdir(class_path):
        # 폴더 내의 랜덤한 이름의 .npy 파일 경로를 수집합니다.
        for filename in os.listdir(class_path):
            if filename.endswith('.npy'):
                # 'target1/random_name_1.npy' 형태의 상대 경로로 저장
                all_files.append(os.path.join(class_folder, filename))

random.shuffle(all_files)

train_size = int(len(all_files) * train_ratio)
train_files = all_files[:train_size]
test_files = all_files[train_size:]

# data/ 폴더에 train/test 목록 파일을 생성합니다.
with open('data/real_ship_train.txt', 'w') as f:
    for file in train_files:
        f.write(file + '\n')

with open('data/real_ship_test.txt', 'w') as f:
    for file in test_files:
        f.write(file + '\n')

print(f"Train/Test 분할 완료! Train: {len(train_files)}개, Test: {len(test_files)}개")