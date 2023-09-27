import numpy as np
import cv2
from IPython import embed
from scipy.spatial import KDTree
from src.tools.find_key_point_2d import CircleDetector, CircleDetectorParams


def find_image_points_rvmxn(image, pattern_per_rows, pattern_per_cols, circle_detect_param=CircleDetectorParams()):
    detector = CircleDetector(param=circle_detect_param)
    key_pts, circle_radius_list, binary_image = detector.detect(image)

    ratio = 7  # rvbust uncoded calibboard: circle_center_distance / circle_radius < 6
    min_num_neighbor = 4
    tree = KDTree(key_pts)

    candidate_pts = []
    for pt, circle_radius in zip(key_pts, circle_radius_list):
        neighbor_idx = tree.query_ball_point(pt, circle_radius * ratio)
        if len(neighbor_idx) >= min_num_neighbor:
            candidate_pts.append(pt)
    print(
        f"number: src detected center: {len(key_pts)}, after filter: {len(candidate_pts)}")

    key_pts = np.array(candidate_pts).astype(np.float32).reshape((-1, 1, 2))

    ret, pts = cv2.findCirclesGrid(key_pts, (pattern_per_rows, pattern_per_cols),
                                   flags=(cv2.CALIB_CB_ASYMMETRIC_GRID),
                                   blobDetector=None,
                                   parameters=None)
    if ret:
        pts = pts.reshape((-1, 2))
    else:
        pts = key_pts.reshape((-1, 2))
        cv2.imwrite("binary_image.png", binary_image)

    return ret, pts


if __name__ == "__main__":
    import line_profiler
    image_path = "./data/K43276266/2.png"
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    circle_detector = CircleDetector()

    prof = line_profiler.LineProfiler(circle_detector.detect)
    prof.enable()
    circle_centers, binary_image = circle_detector.detect(image)
    prof.disable()
    prof.print_stats()

    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for center in circle_centers:
        cv2.circle(image_color, np.int0(center), 5, (0, 255, 0), 1)

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("binary", cv2.WINDOW_NORMAL)
    cv2.imshow("image", image_color)
    cv2.imshow("binary", binary_image)
    cv2.waitKey(0)
