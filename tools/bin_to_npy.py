import os
import shutil

# ========== ❗️ 사용자 설정 영역 시작 ❗️ ==========

# NPY 파일이 현재 저장되어 있는 루트 폴더 경로 (수정된 올바른 경로 사용)
root_dir = '/home/jslee/Junseo/MAE_Marine/data/Ship_Real_ext'

# NPY 파일 이름
target_npy_filename = 'pts_xyz.npy'

# ========== ❗️ 사용자 설정 영역 끝 ❗️ ==========


def move_npy_up_and_clean_unique(root_directory, target_filename):
    """
    원본 폴더의 이름을 접두사로 사용하여 .npy 파일을 한 단계 상위 폴더로 이동시키고,
    파일이 비게 된 하위 폴더를 삭제합니다. (덮어쓰기 방지)
    """
    moved_count = 0
    deleted_folders = []

    # 디버깅: 루트 폴더 존재 여부 확인
    if not os.path.exists(root_directory):
        print(f"❌ 오류: 지정된 루트 폴더가 존재하지 않습니다: {root_directory}")
        return

    print(f"✅ 탐색 시작 폴더: {root_directory}")
    print("-" * 50)

    # os.walk를 깊은 곳부터 탐색 (topdown=False)
    for dirpath, dirnames, filenames in os.walk(root_directory, topdown=False):

        # 1. 파일 이동
        if target_filename in filenames:
            
            current_path = os.path.join(dirpath, target_filename)
            parent_dir = os.path.dirname(dirpath)
            
            # --- ⭐️ 파일 이름 충돌 방지 로직 추가 ⭐️ ---
            # 원본 폴더 이름(dir_name)을 추출
            dir_name = os.path.basename(dirpath)
            
            # 새 파일 이름 생성: '폴더이름_pts_xyz.npy'
            unique_npy_filename = f"{dir_name}_{target_filename}"
            
            # 새 파일 경로 설정 (상위 폴더로 이동)
            new_path = os.path.join(parent_dir, unique_npy_filename)
            
            # ----------------------------------------
            
            try:
                # shutil.move로 파일 이동
                shutil.move(current_path, new_path)
                print(f"✅ 파일 이동 및 이름 변경 완료:")
                print(f"   원본: {current_path}")
                print(f"   변경: {new_path}")
                moved_count += 1
                
            except Exception as e:
                print(f"❌ 파일 이동 오류 발생: {current_path} -> {e}")
        
        # 2. 빈 폴더 삭제
        # 현재 경로(dirpath)가 루트 경로가 아닌 하위 폴더인 경우에만 삭제 시도
        if dirpath != root_directory:
            try:
                # 폴더가 비어 있어야만 삭제됩니다.
                os.rmdir(dirpath)
                deleted_folders.append(dirpath)
                print(f"🗑️ 빈 폴더 삭제 완료: {dirpath}")
            except OSError as e:
                # 폴더가 비어있지 않거나 권한이 없는 경우 (무시)
                pass 

    print("-" * 50)
    print(f"✨ 최종 작업 요약:")
    print(f"   - 총 {moved_count}개의 {target_filename} 파일이 고유한 이름으로 이동되었습니다.")
    print(f"   - 총 {len(deleted_folders)}개의 빈 하위 폴더가 삭제되었습니다.")
    print(f"   - **주의:** 최상위 폴더 ({root_directory})는 삭제되지 않습니다.")

# 함수 실행
move_npy_up_and_clean_unique(root_dir, target_npy_filename)