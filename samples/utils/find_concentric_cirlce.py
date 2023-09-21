import numpy as np
import cv2
from IPython import embed
from scipy.spatial import KDTree
from samples.utils.find_key_point_2d import CircleDetector, CircleDetectorParams


class ConcentricCircleDetectorParams(CircleDetectorParams):
    def __init__(self) -> None:
        super().__init__()
        self.num_ring_edge = 4
        self.search_radius_pixel = 10


class ConcentricCircleDetector(CircleDetector):
    def __init__(self, param=ConcentricCircleDetectorParams()) -> None:
        super().__init__(param)
        self.param = param

    def detect(self, image):
        circle_center_list, _, binary_image = super().detect(image)
        tree = KDTree(circle_center_list)

        concentric_circle_centers = []

        mask = np.zeros(len(circle_center_list), dtyp=bool)
        for i in range(len(circle_center_list)):
            stack_idx = [i]
            cluster_idx = []
            if mask[i]:
                continue
            while len(stack_idx) > 0:
                cur_id = stack_idx.pop()
                if mask[cur_id]:
                    continue
                mask[cur_id] = True
                cluster_idx.append(ni)

                cur_pt = circle_center_list[cur_id]
                neighbor_idx = tree.query_ball_point(
                    cur_pt, self.param.search_radius_pixel)
                for ni in neighbor_idx:
                    if not mask[ni]:
                        stack_idx.append(ni)
            if len(cluster_idx) == self.param.num_ring_edge:
                circle_centers = circle_center_list[cluster_idx]
                concentric_circle_centers.append(
                    np.mean(circle_centers, axis=0))
        return concentric_circle_centers, binary_image
