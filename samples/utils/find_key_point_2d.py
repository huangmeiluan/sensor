import cv2
import numpy as np
from scipy.spatial import KDTree
from IPython import embed


class CircleDetectorParams:
    def __init__(self) -> None:
        self.min_circle_radius_pixel = 5
        self.max_circle_raidius_pixel = 150
        self.min_circularity = 0.6

        self.adaptive_mathod = cv2.ADAPTIVE_THRESH_MEAN_C
        self.adaptive_block_radius_pixel = 50  # circle center distance is recommend
        self.adaptive_C = -10


class CircleDetector:
    def __init__(self, param=CircleDetectorParams()) -> None:
        self.param = param

    def detect(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary_image = cv2.adaptiveThreshold(image, 255, self.param.adaptive_mathod, cv2.THRESH_BINARY,
                                             self.param.adaptive_block_radius_pixel * 2 + 1, self.param.adaptive_C)

        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        circle_centers = []
        circle_radius_list = []

        min_contour_length = 2 * np.pi * self.param.min_circle_radius_pixel
        max_contour_length = 2 * np.pi * self.param.max_circle_raidius_pixel * 2
        min_contour_area = np.pi * self.param.min_circle_radius_pixel**2
        max_contour_area = np.pi * self.param.max_circle_raidius_pixel**2

        debug_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

        for contour in contours:
            contour_length = len(contour)
            if contour_length < min_contour_length or contour_length > max_contour_length:
                continue

            area = cv2.contourArea(contour)
            if area < min_contour_area or area > max_contour_area:
                continue

            center, radius = cv2.minEnclosingCircle(contour)
            circularity = area / (np.pi * radius**2)
            if circularity < self.param.min_circularity:
                continue

            rrt = cv2.fitEllipse(contour)
            cv2.ellipse(debug_image, rrt, (0, 0, 255), 1)
            center = rrt[0]
            radius = np.mean(rrt[1]) / 2
            # print(f"center: {center}, radius: {rrt[1]}, radius: {radius}")
            if radius > self.param.max_circle_raidius_pixel:
                continue

            circle_centers.append(center)
            circle_radius_list.append(radius)
        circle_centers = np.array(circle_centers)
        circle_radius_list = np.array(circle_radius_list)
        cv2.imwrite("debug_image.png", debug_image)
        return circle_centers, circle_radius_list, binary_image


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
        circle_center_list, circle_radius_list, binary_image = super().detect(image)
        if len(circle_center_list) == 0:
            return np.array([]), binary_image
        tree = KDTree(circle_center_list)

        concentric_circle_centers = []

        mask = np.zeros(len(circle_center_list), dtype=bool)
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
                cluster_idx.append(cur_id)

                cur_pt = circle_center_list[cur_id]
                neighbor_idx = tree.query_ball_point(
                    cur_pt, self.param.search_radius_pixel)
                for ni in neighbor_idx:
                    if not mask[ni]:
                        stack_idx.append(ni)
            if len(cluster_idx) >= self.param.num_ring_edge:
                radius_list = circle_radius_list[cluster_idx]
                circle_centers = circle_center_list[cluster_idx]
                sorted_idx = np.argsort(radius_list)
                circle_centers = circle_centers[sorted_idx[:self.param.num_ring_edge]]
                concentric_circle_centers.append(
                    np.mean(circle_centers, axis=0))
        concentric_circle_centers = np.array(concentric_circle_centers)
        return concentric_circle_centers, binary_image
