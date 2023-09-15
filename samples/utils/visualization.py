import numpy as np


def draw_plane(v, plane_normal, plane_center, xLength=0.5, yLength=0.5, color=[0, 1, 0]):
    z_axis = np.array([0, 0, 1])
    rotated_rad_half = np.arccos(np.dot(z_axis, plane_normal)) / 2
    q = [0, 0, 0, 1]
    if rotated_rad_half != 0:
        cross_prod = np.cross(z_axis, plane_normal)
        cross_prod /= np.linalg.norm(cross_prod)
        q = (cross_prod * np.sin(rotated_rad_half)
             ).tolist() + [np.cos(rotated_rad_half)]
    handle_plane = v.Plane(xLength, yLength, 1, 1, plane_center, q, color)
    return handle_plane
