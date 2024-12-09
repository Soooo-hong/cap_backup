import os
import random
from pathlib import Path
import re

# 경로 설정
def extract_number(filename):
    match = re.search(r'\d+', filename.stem)  # 숫자 부분 추출
    if match:
        # 추출된 숫자를 정수로 변환 후 다시 문자열로 변환하여 앞의 0을 제거
        return str(int(match.group(0)))  # int로 변환하여 앞의 0 제거
    return None

images_dir = Path("data/omni3d")
for day in images_dir.iterdir() : 
    jpg_dir = day/'images'
    
    print(jpg_dir)
    # 이미지 파일 목록 가져오기
    jpg_files = sorted([f for f in jpg_dir.iterdir() if f.suffix == '.jpg'])

    # 랜덤으로 섞기
    random.shuffle(jpg_files)

    # 전체 파일 수
    total_files = len(jpg_files)

    # 80:20 비율로 나누기
    split_index = int(total_files * 0.8)
    train_files = jpg_files[:split_index]
    test_files = jpg_files[split_index:]

    # 인덱스 저장
    train_indices = [extract_number(f) for f in train_files]  # 파일 이름에서 숫자 추출
    test_indices = [extract_number(f) for f in test_files] 

    train_file_path = day/"train_sparse.txt"
    test_file_path = day/"test_sparse.txt"
    
    # 파일에 인덱스 저장
    with open(train_file_path, "w") as train_file:
        for index in train_indices:
            train_file.write(f"{index}\n")

    with open(test_file_path, "w") as test_file:
        for index in test_indices:
            test_file.write(f"{index}\n")

print("train_sparse.txt와 test_sparse.txt가 생성되었습니다.")
