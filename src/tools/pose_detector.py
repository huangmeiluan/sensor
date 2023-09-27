import copy
from src.sensor.frame import Frame
from IPython import embed
from src.tools.optimization import compute_rigid_transform
from src.tools.geometry import to_plane_coeff, intersect_l2p
from src.tools.generate_obj_points import generate_obj_points_rvmxn
from src.tools.find_image_points import find_image_points_rvmxn, CircleDetectorParams
from enum import Enum
import numpy as np
import cv2
from typing import Tuple
import sys
sys.path.append("./")


class CalibBoardType(Enum):
    Unknown = 0
    RV_Undecoded_Circle = 1


class PoseDetector:
    def __init__(self) -> None:
        self.calib_board_type = CalibBoardType.RV_Undecoded_Circle
        self.circle_center_distance_mm = 7
        self.pattern_per_rows = 4
        self.pattern_per_cols = 11

        self.calc_plane_radius_pixel = 10

        self.obj_pts = None
        self.updata_obj_points()

        self.circle_detector_param = CircleDetectorParams()

    def updata_obj_points(self):
        if self.calib_board_type == CalibBoardType.RV_Undecoded_Circle:
            self.obj_pts = generate_obj_points_rvmxn(
                self.circle_center_distance_mm, self.pattern_per_rows, self.pattern_per_cols)

    def find_image_points(self, image) -> Tuple[bool, np.ndarray]:
        ret, image_pts = False, np.array([])
        if self.calib_board_type == CalibBoardType.RV_Undecoded_Circle:
            ret, image_pts = find_image_points_rvmxn(
                image, self.pattern_per_rows, self.pattern_per_cols, self.circle_detector_param)

        return ret, image_pts

    def compute_obj_points(self, frame: Frame, image_pts, radius_pixel=5) -> Tuple[bool, np.ndarray]:
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

            if np.all(mask_roi == False):
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

    def compute_obj_pose(self, frame: Frame, use_pnp=True) -> Tuple[bool, np.ndarray]:
        ret, T_obj_2_cam = False, np.eye(4)
        ret, image_pts = self.find_image_points(frame.image)
        detected_obj_pts = np.array([])
        if ret and self.obj_pts is not None and self.obj_pts.shape[0] == image_pts.shape[0]:
            if use_pnp or frame.pm is None:
                ret, rvec, tvec = cv2.solvePnP(
                    self.obj_pts, image_pts, frame.intrinsic_matrix, frame.distortion_cv)
                if ret:
                    R, _ = cv2.Rodrigues(rvec)
                    T_obj_2_cam[:3, :3] = R
                    T_obj_2_cam[:3, 3] = tvec[:, 0]
            else:
                ret, detected_obj_pts = self.compute_obj_points(
                    frame, image_pts, self.calc_plane_radius_pixel)
                if ret:
                    R, T = compute_rigid_transform(
                        detected_obj_pts, self.obj_pts)
                    T_obj_2_cam[:3, :3] = R
                    T_obj_2_cam[:3, 3] = T
                    T_obj_2_cam = frame.extrinsic_matrix @ T_obj_2_cam
        else:
            ret = False
        return ret, T_obj_2_cam, image_pts, detected_obj_pts

    def draw(self, frame: Frame, image_pts=None, pose=None, axis_ratio=1, pattern_was_found=True) -> np.ndarray:
        image = frame.image
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image_pts is not None:
            cv2.drawChessboardCorners(
                image, (self.pattern_per_rows, self.pattern_per_cols), image_pts, pattern_was_found)
            if pattern_was_found and pose is not None:
                axis = np.float32([[self.circle_center_distance_mm * axis_ratio, 0, 0],
                                   [0, self.circle_center_distance_mm * axis_ratio, 0],
                                   [0, 0, self.circle_center_distance_mm * axis_ratio]]).reshape(-1, 3)

                rvec, _ = cv2.Rodrigues(pose[:3, :3])
                tvec = pose[:3, 3]
                project_pts, _ = cv2.projectPoints(
                    axis, rvec, tvec, frame.intrinsic_matrix, frame.distortion_cv)
                project_pts = project_pts.astype(np.int)
                origin = np.int0(image_pts[0].ravel())
                cv2.line(image, origin, tuple(
                    project_pts[0].ravel()), (0, 0, 255), 5)
                cv2.line(image, origin, tuple(
                    project_pts[1].ravel()), (0, 255, 0), 5)
                cv2.line(image, origin, tuple(
                    project_pts[2].ravel()), (255, 0, 0), 5)
        return image


if __name__ == "__main__":
    from sensor.sensor_hik_2d import SensorHik_2D, HikConfigName
    # from sensor.sensor_file3d import SensorFile

    from IPython import embed
    # from RVBUST import Vis

    run_online = False
    if run_online:
        sn = "K43276266"
        ret, dev_info = SensorHik_2D.find_device(sn=sn)
        if not ret:
            print(f"camera sn = {sn} not found")
            sys.exit()
        sensor = SensorHik_2D()
        ret = sensor.connect(hik_device_info=dev_info)
        if not ret:
            print(f"connect to camera sn = {sn} failed")
            sys.exit()
        ret = sensor.open()
        if not ret:
            print(f"open camera sn = {sn} failed")
            sys.exit()

        ret = sensor.set_config(ExposureTime=25000)
        ret, exposure_time_us = sensor.get_config(
            key=HikConfigName.ExposureTime_us)
        print(f"current exposure time (us): {exposure_time_us}")
    # else:
    #     sensor = SensorFile()
    #     sensor.image_dir = "/home/rvbust/Rvbust/Sources/FlexCargo/DysonDemo/data/K43276266"
    #     sn = "file"

    sensor.status.frame.load_from_file(intrinsic_path="/home/rvbust/Rvbust/Sources/FlexCargo/DysonDemo/data/K43276266/intrinsic_matrix.txt",
                                       distortion_path="/home/rvbust/Rvbust/Sources/FlexCargo/DysonDemo/data/K43276266/distortion_coeff.txt")

    pose_detector = PoseDetector()
    pose_detector.calib_board_type = CalibBoardType.RV_Undecoded_Circle
    pose_detector.pattern_per_rows = 4
    pose_detector.pattern_per_cols = 11
    pose_detector.circle_center_distance_mm = 20
    pose_detector.updata_obj_points()

    # v = Vis.View()
    # v.Axes(np.eye(4).flatten(), pose_detector.circle_center_distance_mm * 3, 5)
    while True:
        ret, frame = sensor.capture()
        if not ret:
            print(f"camera sn = {sn} capture failed")
            break

        ret, obj_pose, image_pts, _ = pose_detector.compute_obj_pose(frame)
        # if ret:
        #     v.Axes(obj_pose.flatten(), pose_detector.circle_center_distance_mm, 1)
        image = frame.image
        if ret:
            image = pose_detector.draw(
                frame=frame, image_pts=image_pts, pose=obj_pose)
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow("image", image)
        key = chr(cv2.waitKey(run_online * 30) & 0xff)

        if key == 'q':
            break
        if key == 'e':
            embed()

    embed()
    sensor.close()
    sensor.disconnect()
