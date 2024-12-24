import os

def rename_images_in_directory(directory):
    # 디렉토리에 있는 파일 가져오기
    files = [f for f in os.listdir(directory) if f.endswith('.jpg')]

    # 파일 이름 정렬 (숫자 기준)
    files.sort(key=lambda x: int(x.split('.')[0]))

    # 파일 이름 재지정
    for i, file in enumerate(files):
        new_name = f"{i:05}.jpg"  # 00000 형식으로 이름 설정
        old_path = os.path.join(directory, file)
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {file} -> {new_name}")

# 사용 예시
directory_path = "/home/soohong/cap_backup/flash3d_2/data/sub/peanut_009/standard/mask"  # 디렉토리 경로를 여기에 입력하세요
rename_images_in_directory(directory_path)
