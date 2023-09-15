import numpy as np
import cv2
from IPython import embed


class CircleDetectorParams:
    def __init__(self) -> None:
        self.min_circle_radius_pixel = 10
        self.max_circle_raidius_pixel = 50
        self.min_circularity = 0.5

        self.adaptive_mathod = cv2.ADAPTIVE_THRESH_MEAN_C
        self.adaptive_block_radius_pixel = 100  # circle center distance is recommend
        self.adaptive_C = -10


class CircleDetector:
    def __init__(self, param=CircleDetectorParams()) -> None:
        self.param = param

    def detect(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("image.png", image)
        binary_image = cv2.adaptiveThreshold(image, 255, self.param.adaptive_mathod, cv2.THRESH_BINARY,
                                             self.param.adaptive_block_radius_pixel * 2 + 1, self.param.adaptive_C)
        cv2.imwrite("binary_image.png", binary_image)

        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        circle_centers = []

        min_contour_length = 2 * np.pi * self.param.min_circle_radius_pixel
        max_contour_length = 2 * np.pi * self.param.max_circle_raidius_pixel * 2
        min_contour_area = np.pi * self.param.min_circle_radius_pixel**2
        max_contour_area = np.pi * self.param.max_circle_raidius_pixel**2

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

            circle_centers.append(center)
        circle_centers = np.array(circle_centers)
        return circle_centers, binary_image


def find_image_points_rvmxn(image, pattern_per_rows, pattern_per_cols, circle_detect_param=CircleDetectorParams()):
    detector = CircleDetector(param=circle_detect_param)
    key_pts, _ = detector.detect(image)
    key_pts = key_pts.astype(np.float32).reshape((-1, 1, 2))

    ret, pts = cv2.findCirclesGrid(key_pts, (pattern_per_rows, pattern_per_cols),
                                   flags=(cv2.CALIB_CB_ASYMMETRIC_GRID),
                                   blobDetector=None,
                                   parameters=None)
    if ret:
        pts = pts.reshape((-1, 2))
    else:
        pts = key_pts.reshape((-1, 2))

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
