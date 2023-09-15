import numpy as np
from typing import Tuple, Any


def to_plane_coeff(plane_center, plane_normal):
    coeff = np.zeros((4, ))
    coeff[:3] = plane_normal
    coeff[3] = -np.dot(plane_center, plane_normal)
    return coeff


def fit_surface_2nd(pcd):
    '''_summary_

    Args:
        pcd (_type_): _description_

    Returns:
        surface_coeff: f(surface_coeff) = [x**2, y**2, z**2, x * y, x * z, y * z, x, y, z, 1] * surface_coeff
    '''
    num_pts = pcd.shape[0]
    fit_error, coeff = float('inf'), np.zeros((10, ), dtype=np.float32)
    if num_pts < 3:
        return fit_error, coeff
    matrix = np.zeros((10, 10), dtype=np.float64)
    for xyz in pcd:
        x, y, z = xyz
        single = np.array([x**2, y**2, z**2, x * y, x * z, y * z, x, y, z, 1])
        matrix[0] += single * x**2
        matrix[1] += single * y**2
        matrix[2] += single * z**2
        matrix[3] += single * x * y
        matrix[4] += single * x * z
        matrix[5] += single * y * z
        matrix[6] += single * x
        matrix[7] += single * y
        matrix[8] += single * z
        matrix[9] += single

    eigen_values, eigen_vectors = np.linalg.eig(matrix)

    id = np.argmin(eigen_values)
    fit_error = np.sqrt(eigen_values[id] / num_pts)
    coeff = eigen_vectors[:, id]

    return fit_error, coeff


def intersect_l2p(plane_center, plane_normal, pt_in_line, line_vec) -> Tuple[bool, Any]:
    dot_pl = np.dot(plane_normal, line_vec)
    is_parallel = dot_pl == 0
    if is_parallel:
        is_pt_in_plane = (dot_pl - np.dot(plane_center, plane_normal)) == 0
        if is_pt_in_plane:
            return True, pt_in_line
        return False, None
    d = np.dot(plane_center - pt_in_line, plane_normal) / dot_pl
    intersect_pt = d * line_vec + pt_in_line
    return True, intersect_pt
