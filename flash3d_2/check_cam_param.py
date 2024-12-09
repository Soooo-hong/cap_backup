import cv2
import numpy as np

def verify_camera_parameters(camera_matrix, dist_coeffs, rvec, tvec, points_3d, points_2d, image_size):
    """
    카메라 매개변수 검증.
    
    Parameters:
        camera_matrix (np.array): 카메라 내부 파라미터 행렬 (3x3).
        dist_coeffs (np.array): 왜곡 계수 (1x5 또는 1x8).
        rvec (np.array): 회전 벡터 (3x1).
        tvec (np.array): 변환 벡터 (3x1).
        points_3d (np.array): 3D 점 좌표 (Nx3).
        points_2d (np.array): 2D 이미지 점 좌표 (Nx2).
        image_size (tuple): 이미지 크기 (width, height).
    
    Returns:
        dict: 검증 결과 (투영 에러, 왜곡 결과, 내부 파라미터 확인 여부).
    """
    results = {}
    
    # 1. 3D -> 2D로 투영된 점 계산
    projected_points, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
    projected_points = projected_points.reshape(-1, 2)
    
    # 2. 투영 에러 계산
    projection_error = np.mean(np.linalg.norm(projected_points - points_2d, axis=1))
    results['projection_error'] = projection_error

    # 3. 왜곡 계수 확인
    if dist_coeffs is not None:
        undistorted_points = cv2.undistortPoints(points_2d, camera_matrix, dist_coeffs, None, camera_matrix)
        distortion_error = np.mean(np.linalg.norm(undistorted_points - points_2d, axis=1))
        results['distortion_error'] = distortion_error

    # 4. 내부 파라미터 확인 (예: 초점 거리와 주점 확인)
    focal_length_x = camera_matrix[0, 0]
    focal_length_y = camera_matrix[1, 1]
    principal_point_x = camera_matrix[0, 2]
    principal_point_y = camera_matrix[1, 2]

    results['focal_length'] = (focal_length_x, focal_length_y)
    results['principal_point'] = (principal_point_x, principal_point_y)
    
    # 결과 출력
    return results

# 예제 데이터
camera_matrix = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float64)
dist_coeffs = np.array([0.1, -0.05, 0, 0, 0], dtype=np.float64)
rvec = np.array([0.1, 0.2, 0.3], dtype=np.float64)
tvec = np.array([0, 0, 1000], dtype=np.float64)
points_3d = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64)
points_2d = np.array([[640, 360], [650, 370], [660, 380]], dtype=np.float64)
image_size = (1280, 720)

# 검증 실행
results = verify_camera_parameters(camera_matrix, dist_coeffs, rvec, tvec, points_3d, points_2d, image_size)
print("검증 결과:", results)
