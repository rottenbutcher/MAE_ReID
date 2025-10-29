# tools/split_reid_data.py
import os
import random

# --- 사용자 설정 ---

# 1. Real_Ship 데이터가 있는 원본 경로
#    (예: 'data/Ship_Real_ext')
DATA_ROOT = 'data/Ship_Real_ext'

# 2. 분할된 .txt 목록 파일이 저장될 경로
#    (ShipDataset.py의 list_file 경로와 일치해야 함)
OUTPUT_DIR = 'data/'

# 3. 분할할 Ship 클래스(ID) 개수
TRAIN_SHIP_COUNT = 12
VAL_SHIP_COUNT = 4
# TEST_SHIP_COUNT는 나머지 Ship ID 전부가 됩니다. (19 - 12 - 4 = 3개)

# 4. (중요) Buoy 클래스를 식별하는 방법
#    클래스 폴더 이름이 'buoy'로 시작하면 True를 반환하도록 함수 수정
def is_buoy(class_folder_name):
    """폴더 이름을 기반으로 Buoy 클래스인지 판별하는 함수"""
    # 예: 폴더 이름이 'buoy_1', 'buoy_A' 등 'buoy'로 시작하는 경우
    return class_folder_name.startswith('FP_target')

# --- --- --- ---

def write_file_list(class_list, output_path, data_root):
    """
    지정된 클래스 목록에 포함된 모든 .npy 파일 경로를
    output_path 파일에 씁니다.
    """
    file_paths = []
    for class_folder in class_list:
        class_path = os.path.join(data_root, class_folder)
        
        if not os.path.isdir(class_path):
            print(f"  [Warning] 클래스 폴더를 찾을 수 없습니다: {class_path}. 건너뜁니다.")
            continue
            
        for filename in os.listdir(class_path):
            if filename.endswith('.npy'):
                # 경로 형식: 'target1/random_name.npy'
                relative_path = os.path.join(class_folder, filename)
                # ShipDataset.py가 '/'를 기준으로 읽으므로 os.path.sep을 '/'로 변경
                file_paths.append(relative_path.replace(os.path.sep, '/'))

    # 안정적인 학습을 위해 각 세트 내의 파일 순서를 섞음
    random.shuffle(file_paths)
    
    with open(output_path, 'w') as f:
        for path in file_paths:
            f.write(path + '\n')
            
    print(f"  > {output_path} 생성 완료 (파일: {len(file_paths)}개, ID: {len(class_list)}개)")
    return len(file_paths)

def main():
    if not os.path.isdir(DATA_ROOT):
        print(f"[Error] 데이터 원본 경로를 찾을 수 없습니다: {DATA_ROOT}")
        print("스크립트 상단의 DATA_ROOT 변수를 수정하세요.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 모든 클래스 폴더 목록 읽기
    try:
        all_class_folders = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])
    except FileNotFoundError:
        print(f"[Error] 데이터 원본 경로를 읽을 수 없습니다: {DATA_ROOT}")
        return

    if not all_class_folders:
        print(f"[Error] {DATA_ROOT} 경로에 하위 폴더(클래스)가 없습니다.")
        return

    # is_buoy 함수를 기준으로 Ship과 Buoy 클래스 분리
    ship_classes = []
    buoy_classes = []
    for folder in all_class_folders:
        if is_buoy(folder):
            buoy_classes.append(folder)
        else:
            ship_classes.append(folder)
            
    print("--- 클래스 식별 결과 ---")
    print(f"총 {len(all_class_folders)}개 클래스(ID) 폴더 발견")
    print(f"  - Ship ID: {len(ship_classes)}개 (e.g., {ship_classes[:3]}...)")
    print(f"  - Buoy ID: {len(buoy_classes)}개 (e.g., {buoy_classes[:3]}...)")
    
    # 요청하신 19개/7개와 맞는지 확인
    if len(ship_classes) != 19 or len(buoy_classes) != 7:
        print("[Warning] 요청하신 Ship 19개, Buoy 7개와 실제 폴더 수가 일치하지 않습니다.")
        print(f"          (is_buoy 함수가 {len(ship_classes)} ships, {len(buoy_classes)} buoys를 반환했습니다.)")
        print("          스크립트 상단의 is_buoy 함수를 확인하거나 폴더 구조를 확인하세요.")

    if len(ship_classes) < (TRAIN_SHIP_COUNT + VAL_SHIP_COUNT):
        print(f"[Error] Ship 클래스 수가 ({len(ship_classes)}개) Train/Val 분할에 필요한 수({TRAIN_SHIP_COUNT + VAL_SHIP_COUNT}개)보다 적습니다.")
        return
        
    # --- ID (클래스) 분할 ---
    print("\n--- ID 분할 시작 ---")
    
    # Ship 클래스 목록을 섞어서 Train/Val/Test 용으로 분할
    random.shuffle(ship_classes)
    
    train_ships = ship_classes[:TRAIN_SHIP_COUNT]
    val_ships = ship_classes[TRAIN_SHIP_COUNT : TRAIN_SHIP_COUNT + VAL_SHIP_COUNT]
    test_ships = ship_classes[TRAIN_SHIP_COUNT + VAL_SHIP_COUNT:]
    
    # 각 세트에 Buoy 클래스를 추가
    train_classes = sorted(train_ships + buoy_classes)
    val_classes = sorted(val_ships + buoy_classes)
    test_classes = sorted(test_ships + buoy_classes)

    print(f"Train Set ID: {len(train_ships)} ships + {len(buoy_classes)} buoys = {len(train_classes)}개 ID")
    print(f"Val   Set ID: {len(val_ships)} ships + {len(buoy_classes)} buoys = {len(val_classes)}개 ID")
    print(f"Test  Set ID: {len(test_ships)} ships + {len(buoy_classes)} buoys = {len(test_classes)}개 ID")
    
    # --- 파일 목록 생성 ---
    print("\n--- 파일 목록 생성 시작 ---")
    
    # Train
    write_file_list(
        train_classes,
        os.path.join(OUTPUT_DIR, 'real_ship_reid_train.txt'),
        DATA_ROOT
    )
    
    # Validation
    write_file_list(
        val_classes,
        os.path.join(OUTPUT_DIR, 'real_ship_reid_val.txt'),
        DATA_ROOT
    )
    
    # Test
    write_file_list(
        test_classes,
        os.path.join(OUTPUT_DIR, 'real_ship_reid_test.txt'),
        DATA_ROOT
    )
    
    print("\n[Success] ReID용 데이터 분할(Step 1) 완료!")

if __name__ == "__main__":
    main()