import numpy as np

def compute_rigid_transform(src_pcd, dst_pcd):
    centroid_x = np.mean(dst_pcd, axis=0)
    centroid_y = np.mean(src_pcd, axis=0)
    H = np.dot((dst_pcd - centroid_x).T, src_pcd - centroid_y)
    U, S, VT = np.linalg.svd(H)
    V = VT.T
    UT = U.T
    R = np.dot(V, UT)
    det_R = np.linalg.det(R)
    if(det_R < 0):
        V[:, 2] *= -1
        R = np.dot(V, UT)
    T = centroid_y - np.dot(R, centroid_x)
    return R, T

if __name__ == "__main__":
    from IPython import embed
    points_in_cam_base = np.array([
        [-2.684260785161995955e-01, -6.090822204716630495e-01, 7.564390107317242595e-03],
        [-6.316223435044321377e-01, 8.467256630382992455e+01, 1.052906123330599186e-02],
        [1.405730905945689528e+02, -5.012562257102501917e-01, 1.246683159287487186e-02],
        [1.403232179019595378e+02, 8.442227666532673425e+01, -2.788688499375746588e-03],
        [7.005546500107323027e+01, 1.364942152988025725e+01, -8.114784683998843562e-03]
    ])
    points_in_robot_base = np.array([
        [0.966708, 0.29481, 0.599944],
        [0.967609, 0.21096, 0.599106],
        [0.826631, 0.293, 0.600649],
        [0.828223, 0.209574, 0.600807],
        [0.89765, 0.268, 0.600586]
    ]) * 1000
    R, T = compute_rigid_transform(points_in_cam_base, points_in_robot_base)
    T_cam_2_robot_base = np.eye(4)
    T_cam_2_robot_base[:3, :3] = R
    T_cam_2_robot_base[:3, 3] = T
    print(f"T_cam_2_robot_base: {T_cam_2_robot_base}")
    embed()