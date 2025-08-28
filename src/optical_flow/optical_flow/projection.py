import cv2
import numpy as np

def undistort_pixels(features, K, D) -> np.ndarray:
    """
    points: Nx2 pixel coordinates
    K: camera intrinsic matrix
    D: distortion coefficients
    """
    features = np.array(features, dtype=np.float32).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(features, K, D, P=K)  # P=K keeps pixels in image coords
    return undistorted.reshape(-1, 2)


def project_points_to_ground(features, K, D, R, camera_height) -> np.ndarray:
    """
    Project 2D pixel coordinates to ground plane (z=0).

    Args:
        points (Nx2 ndarray): pixel coordinates (already undistorted!)
        K (3x3 ndarray): camera intrinsic matrix
        R (3x3 ndarray): rotation from camera to world
        camera_height (float): camera height above ground
    
    Returns:
        Nx3 ndarray: 3D ground coordinates in world frame
    """
    # 0. undistort pixels
    undist_features = undistort_pixels(features, K, D)
    
    # 1. Homogeneous pixel coords
    pix_h = np.hstack([undist_features, np.ones((undist_features.shape[0], 1))])  # (N,3)

    # 2. Back-project to camera rays
    K_inv = np.linalg.inv(K)
    d_c = (K_inv @ pix_h.T).T  # (N,3)
    d_c /= np.linalg.norm(d_c, axis=1, keepdims=True)  # normalize

    # 3. Rotate to world coordinates
    d_w = (R @ d_c.T).T  # (N,3)

    # 4. Intersect with ground plane z=0
    t = -camera_height / d_w[:, 2]  # (N,)
    camera_pos = np.array([0, 0, camera_height])
    ground_points = camera_pos + d_w * t[:, np.newaxis]  # (N,3)

    return ground_points