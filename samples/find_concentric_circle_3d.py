from src.tools.find_key_point_2d import ConcentricCircleDetector, ConcentricCircleDetectorParams
import numpy as np
from src.sensor.frame import Frame
from typing import Tuple
import cv2
import copy
from src.tools.geometry import intersect_l2p


class ConcentricCircleDetectorParams_3d(ConcentricCircleDetectorParams):
    def __init__(self) -> None:
        super().__init__()
        self.calc_plane_radius_pixel = 10
        self.num_min_point = 4


class ConcentricCircleDetector_3d(ConcentricCircleDetector):
    def __init__(self, param=ConcentricCircleDetectorParams_3d()) -> None:
        super().__init__(param)
        self.param = param

    def compute_obj_points(self, frame: Frame, image_pts, radius_pixel=5) -> Tuple[bool, np.ndarray]:
        if len(image_pts) == 0:
            return False, np.array([], dtype=np.float32).reshape((-1, 3))
        undistort_pts = cv2.undistortPoints(
            image_pts, frame.intrinsic_matrix, frame.distortion_cv).reshape((-1, 2))
        h, w = frame.image.shape[:2]
        mask_pm = ~np.isnan(frame.pm[:, :, 2])
        mask_image = np.zeros((h, w), dtype=np.uint8)
        for image_pt in image_pts:
            cv2.circle(mask_image, np.int0(image_pt), radius_pixel, 255, -1)

        mask = np.logical_and(mask_pm, mask_image > 0)
        num_pts = image_pts.shape[0]

        ret, obj_pts = False, np.zeros((num_pts, 3), dtype=np.float32)
        T_pcd_2_cam = frame.extrinsic_matrix
        T_cam_2_pcd = np.linalg.inv(T_pcd_2_cam)
        for i in range(num_pts):
            image_pt = np.int0(image_pts[i])
            c0, r0 = image_pt - radius_pixel - 1
            c1, r1 = image_pt + radius_pixel + 2
            r0 = max(r0, 0)
            c0 = max(c0, 0)
            r1 = min(r1, h)
            c1 = min(w, c1)
            pm_roi = frame.pm[r0:r1, c0:c1]
            mask_roi = mask[r0:r1, c0:c1]

            if np.count_nonzero(mask_roi) < self.param.num_min_point:
                ret = False
                break

            plane_center, plane_normal, rms_error = cv2.fitPlane(copy.deepcopy(
                pm_roi.astype(np.float32)), copy.deepcopy(mask_roi.astype(np.uint8)))

            undistort_pt = undistort_pts[i]
            pt_in_line = np.array([undistort_pt[0], undistort_pt[1], 1])
            plane_center_in_cam = np.dot(
                T_pcd_2_cam[:3, :3], plane_center) + T_pcd_2_cam[:3, 3]
            plane_normal_in_cam = np.dot(T_pcd_2_cam[:3, :3], plane_normal)
            ret, intersect_pt = intersect_l2p(
                plane_center_in_cam, plane_normal_in_cam, pt_in_line, pt_in_line)
            if not ret:
                break
            obj_pts[i] = np.dot(T_cam_2_pcd[:3, :3],
                                intersect_pt) + T_cam_2_pcd[:3, 3]

        return ret, obj_pts

    def detect(self, frame):
        image = frame.image
        centers_2d, _ = super().detect(image)

        ret, centers_3d = self.compute_obj_points(
            frame, centers_2d, self.param.calc_plane_radius_pixel)
        return ret, centers_3d, centers_2d


if __name__ == "__main__":
    from src.sensor.sensor_manager import SensorManager
    from src.tools.common import get_common_parse
    from RVBUST import Vis
    import sys
    from IPython import embed

    parse = get_common_parse("find concentric circle center")
    args = parse.parse_args()

    sn = args.sn
    config_sensor_path = args.config_sensor_path

    param = ConcentricCircleDetectorParams_3d()
    param.num_ring_edge = 4
    param.adaptive_C = -10
    param.adaptive_block_radius_pixel = 50

    circle_detector = ConcentricCircleDetector_3d(param=param)

    sensor_manager = SensorManager(config_sensor_path)
    sensor_manager.list_device(
        list_sensor_type=args.list_sensor_type, rvc_list_device_type=args.rvc_list_device_type)
    flag = sensor_manager.open(sn_list=[sn])
    if not flag:
        sys.exit()

    v = Vis.View()
    sensor = sensor_manager.sensor_dict[sn]
    while True:
        v.Clear()
        flag, frame = sensor.capture()
        if not flag:
            print("capture failed")
            sys.exit()

        ret, centers_3d, centers_2d = circle_detector.detect(frame=frame)
        print(f"centers_3d (mm): {centers_3d}")
        print(f"centers_2d: {centers_2d}")

        v.Axes(np.eye(4).flatten(), 1000, 1)
        v.Point(centers_3d, 3, [0, 1, 0])
        v.Point(frame.get_valid_points(), 1, [.5, .5, .5])

        debug_image = cv2.cvtColor(frame.image, cv2.COLOR_GRAY2BGR)
        for pt in centers_2d:
            cv2.circle(debug_image, np.int0(pt), 5, (0, 0, 255), 2)

        win_name = f"{sensor.param.sensor_name}_{sensor.param.sn}"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, debug_image)
        key = chr(cv2.waitKey(0) & 0xff)
        if key == 'e':
            embed()
        if key == 'q':
            break
        if key == 's':
            sensor = sensor_manager.sensor_dict[sn]
            if args.save_as_file_camera and hasattr(sensor, "save_as_file_camera"):
                sensor.save_as_file_camera(
                    config_sensor_path, args.file_camera_dir)
                print(
                    f"create file camera, raw data save in {args.file_camera_dir}/{sn}")
            if args.save_config and hasattr(sensor, "save_config"):
                sensor.save_config(args.config_sensor_path)
                print(f"save sensor config in {args.config_sensor_path}")
