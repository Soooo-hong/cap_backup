import subprocess

# 텍스트 파일 경로
category_file = "category_list.txt"  # 수정 필요

# 기본 명령어 템플릿
base_command = "openxlab dataset download --dataset-repo omniobject3d/OmniObject3D-New --source-path {category} --target-path /home/soohong/cap_backup/flash3d_2/data/sub "

# 텍스트 파일 읽기
try:
    with open(category_file, "r") as file:
        categories = [line.strip() for line in file if line.strip()]
except FileNotFoundError:
    print(f"File not found: {category_file}")
    exit(1) 

# 각 카테고리별로 데이터셋 다운로드
for category in categories:
    command = base_command.format(category=category)
    print(f"Executing: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading category '{category}': {e}")
