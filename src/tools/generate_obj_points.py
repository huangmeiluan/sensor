import numpy as np
import cv2


def generate_obj_points_rvmxn(circle_center_distance_mm, pattern_per_rows, pattern_per_cols):
    obj_pts = np.zeros(
        (pattern_per_rows*pattern_per_cols, 3), dtype=np.float32)
    for i in range(pattern_per_cols):
        for j in range(pattern_per_rows):
            obj_pts[pattern_per_rows * i + j] = (i * circle_center_distance_mm / 2,
                                                 circle_center_distance_mm * j + 0.5 * circle_center_distance_mm * (i % 2), 0)

    return obj_pts


if __name__ == "__main__":
    from RVBUST import Vis
    from IPython import embed

    v = Vis.View()

    pattern_per_rows = 4
    pattern_per_cols = 11
    circle_center_distance_mm = 20

    obj_pts = generate_obj_points_rvmxn(
        circle_center_distance_mm, pattern_per_rows, pattern_per_cols)
    v.Point(obj_pts.flatten(), 3, [1, 0, 0])

    embed()
