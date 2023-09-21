from src.sensor.sensor_manager import SensorManager
from samples.utils.pose_detector import PoseDetector
import numpy as np
import copy
import cv2


class PositionInfos:
    def __init__(self, image_points=None, pose_robot_end=np.eye(4), frame=None) -> None:
        self.image_points = image_points
        self.pose_robot_end = pose_robot_end
        self.frame = frame


class HandEyeCalibrationManager_2d:
    def __init__(self) -> None:
        self.pose_detector = PoseDetector()

        self.position_infos_list = []  # list of PositionInfos
        self.image_shape = None

    def set_calibboard_circle_center_distance(self, circle_center_distance_mm):
        self.pose_detector.circle_center_distance_mm = circle_center_distance_mm
        self.pose_detector.updata_obj_points()

    def _get_debug_image(self, ret, frame, image_pts):
        image = frame.image
        if ret:
            debug_image = self.pose_detector.draw(
                frame=frame, image_pts=image_pts)
        else:
            if len(image.shape) == 2:
                debug_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                debug_image = copy.deepcopy(image)
            for pt in image_pts:
                cv2.circle(debug_image, np.int0(pt), 5, (0, 0, 255), 2)
        return debug_image

    def detect_calib_points(self, frame):
        image = frame.image
        ret, image_pts = self.find_image_points(image)
        debug_image = self._get_debug_image(ret, frame, image_pts)
        return ret, image_pts, debug_image

    def add_data(self, frame, pose_robot_end, keep_src_frame=True):
        ret, image_pts = self.pose_detector.find_image_points(frame.image)
        if ret:
            src_frame = None
            if keep_src_frame:
                src_frame = copy.deepcopy(frame)
            position_infos = PositionInfos(
                image_points=image_pts, pose_robot_end=pose_robot_end, frame=src_frame)
            self.position_infos_list.append(position_infos)

            self.image_shape = frame.image.shape
        return ret, image_pts

    def _calibrate_camera_2d(self, src_image_pts_list, src_obj_pts, image_shape):
        image_pts_list = np.array(src_image_pts_list)
        image_pts_list = image_pts_list.reshape(
            (image_pts_list.shape[0], image_pts_list.shape[1], 1, image_pts_list.shape[2]))
        obj_pts = src_obj_pts.reshape(
            (1, src_obj_pts.shape[0], src_obj_pts.shape[1]))
        obj_pts_list = np.repeat(
            obj_pts, image_pts_list.shape[0], axis=0).astype(np.float32)
        ret, intrinsic_matrix, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(
            obj_pts_list, image_pts_list, image_shape, None, None)
        dist_coeff_rv = np.array(
            [dist_coeff[0], dist_coeff[1], dist_coeff[4], dist_coeff[2], dist_coeff[3]])
        return ret, intrinsic_matrix, dist_coeff_rv

    def calibrate(self, eye_in_hand: bool, calib_sensor=True, intrinsic_matrix=None, dist_coeff_rv=None):
        hand_eye_result = np.eye(4)
        if calib_sensor:
            image_pts_list = [
                position_infos.image_points for position_infos in self.position_infos_list]
            ret, intrinsic_matrix, dist_coeff_rv = self._calibrate_camera_2d(
                image_pts_list, self.pose_detector.obj_pts, self.image_shape)
        elif intrinsic_matrix is None or dist_coeff_rv is None:
            raise ValueError(
                f"intrinsic_matrix and dist_coeff_rv cannot be None when calib_sensor is False")
        calib_pose_list = []
        for position_infos in self.position_infos_list:
            image_pts = position_infos.image_points
            ret, T_obj_2_cam, _, _ = self.pose_detector.compute_obj_pose(
                image_pts, use_pnp=True)
            if not ret:
                return False, intrinsic_matrix, dist_coeff_rv, hand_eye_result
            calib_pose_list.append(T_obj_2_cam)

        T_gripper_2_base = np.array(
            [position_infos.pose_robot_end for position_infos in self.position_infos_list])
        if not eye_in_hand:
            for i in range(T_gripper_2_base.shape[0]):
                T_gripper_2_base[i] = np.linalg.inv(T_gripper_2_base[i])
        T_target_2_cam = np.array(calib_pose_list)
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            T_gripper_2_base[:, :3, :3], T_gripper_2_base[:, :3, 3], T_target_2_cam[:, :3, :3], T_target_2_cam[:, :3, 3])
        hand_eye_result[:3, :3] = R_cam2gripper
        hand_eye_result[:3, 3] = t_cam2gripper
        return True, intrinsic_matrix, dist_coeff_rv, hand_eye_result


class HandEyeCalibrationManager_Mix(HandEyeCalibrationManager_2d):
    def __init__(self) -> None:
        super().__init__()

        self.calib_pose_out_hand = np.eye(4)
        self.calib_pose_in_hand = np.eye(4)

    def detect_calib_pose(self, frame):
        ret, obj_pose, image_pts, obj_pts = self.pose_detector.compute_obj_pose(
            frame, use_pnp=frame.pm is None)
        debug_image = self._get_debug_image(ret, frame, image_pts)
        return ret, obj_pose, obj_pts, debug_image

    def calibrate_camera_out_hand(self, robot_pose, calib_pose_in_hand, calib_pose_out_hand):
        T_cam_2_base = np.eye(4)
        return T_cam_2_base
