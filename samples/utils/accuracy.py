import numpy as np


def compute_accuracy(obj_pose, obj_points_in_calibord, extrinsic_matrix, benchmark_points_in_camera):
    T_caliboard_2_pcd = np.linalg.inv(extrinsic_matrix) @ obj_pose
    trans_points = np.dot(
        T_caliboard_2_pcd[:3, :3], benchmark_points_in_camera.T).T + T_caliboard_2_pcd[:3, 3]
    delta_points = trans_points - obj_points_in_calibord
    errors = np.linalg.norm(delta_points, axis=1)
    print(
        f"error min: {np.min(errors)}, max: {np.max(errors)}, quatile_0,5: {np.quantile(errors, 0.5)}")
    return errors
