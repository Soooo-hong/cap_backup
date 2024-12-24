import os
import cv2
import numpy as np

def chage_matting2mask(dir) : 
    origin = os.path.join(dir,'matting')
    if os.path.isdir(origin) : 
        new_dir = os.path.join(dir,'mask')
        os.rename(origin,new_dir)
        
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
    print(f"Change_rename")


def apply_mask_and_save_images(image_dir, mask_dir, output_dir):
    # 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 이미지와 마스크 파일 처리
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    for image_file in image_files:
        # 이미지와 매칭되는 마스크 파일 이름
        mask_file = image_file  # 동일한 이름의 마스크 파일이 있다고 가정

        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, mask_file)

        if not os.path.exists(mask_path):
            print(f"Mask not found for {image_file}. Skipping...")
            continue

        # 이미지와 마스크 읽기
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 배경 흰색 만들기
        background = np.full_like(image, 255)  # 흰색 배경
        mask_3channel = cv2.merge([mask, mask, mask])
        # 마스크를 이용하여 원본 이미지에서 객체 추출
        mask_inv = cv2.bitwise_not(mask_3channel)
        masked_image = cv2.bitwise_and(image, mask_3channel)
        
        # 객체와 흰색 배경 합성
        result = cv2.add(masked_image,mask_inv)
        

        # 결과 저장
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, result)
    print(f"Saved processed image")


category = 'peanut_031'
directory_path = f"/home/soohong/cap_backup/flash3d_2/data/sub/{category}/standard"  # 디렉토리 경로를 여기에 입력하세요
# chage_matting2mask(directory_path)

image_dir = f"/home/soohong/cap_backup/flash3d_2/data/sub/{category}/standard/images"  # 원본 이미지 디렉토리 경로
mask_dir = f"/home/soohong/cap_backup/flash3d_2/data/sub/{category}/standard/mask"  # 마스크 이미지 디렉토리 경로
output_dir = f"/home/soohong/cap_backup/flash3d_2/data/sub/{category}/standard/output"  # 결과 저장할 디렉토리 경로
# apply_mask_and_save_images(image_dir, mask_dir, output_dir)

rename_images_in_directory(image_dir)
rename_images_in_directory(output_dir)
rename_images_in_directory(mask_dir)

