from src.tools.pose_detector import PoseDetector
import numpy as np
import copy
import cv2
from sensor.frame import Frame


def detect_caliboard_pose(frame, **kwargs):
    if "circle_center_distance" in kwargs:
        pose_detector = PoseDetector()
        pose_detector.circle_center_distance_mm = kwargs["circle_center_distance"]
    elif "pose_detector" in kwargs:
        pose_detector = kwargs["pose_detector"]
    else:
        raise ValueError(
            "'circle_center_distance' and 'pose_detector' is key word input, must input one of them")
    ret, T_target_2_cam, points_2d, points_3d = pose_detector.compute_obj_pose(
        frame, use_pnp=frame.pm is None)
    T_cam_2_cam_base = frame.pose
    T_target_2_cam_base = T_cam_2_cam_base @ T_target_2_cam
    return ret, T_target_2_cam_base, points_2d, points_3d


class PositionInfos_2d:
    def __init__(self, image_points=None, T_end_2_base=np.eye(4), image=None) -> None:
        self.image_points = image_points
        self.T_end_2_base = T_end_2_base
        self.image = image


class HandEyeCalibrateManager_2d:
    def __init__(self) -> None:
        self.pose_detector = PoseDetector()
        self.position_infos_list = []  # list of PositionInfos_2d
        self._image_shape = None

    def set_circle_center_distance(self, circle_center_distance_mm):
        self.pose_detector.circle_center_distance_mm = circle_center_distance_mm
        self.pose_detector.updata_obj_points()

    def detect_calib_points(self, image):
        ret, image_pts = self.pose_detector.find_image_points(image)
        return ret, image_pts

    def draw_points(self, image, pattern_was_found, image_pts):
        frame = Frame(image=image)
        image = self.pose_detector.draw(
            frame=frame, image_pts=image_pts, pattern_was_found=pattern_was_found)
        return image

    def add_data(self, image, pose_robot_end):
        position_infos = PositionInfos_2d()
        if self._image_shape is not None and image.shape != self._image_shape:
            print(
                f"image size is difference, {self._image_shape} vs {image.shape}, try calling clear() !!!")
            return False, position_infos
        ret, image_pts = self.pose_detector.find_image_points(image)
        if ret:
            position_infos = PositionInfos_2d(
                image_points=image_pts, T_end_2_base=pose_robot_end, image=copy.deepcopy(image))
            self.position_infos_list.append(position_infos)

            self._image_shape = image.shape
        return ret, position_infos

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
                image_pts_list, self.pose_detector.obj_pts, self._image_shape)
        elif intrinsic_matrix is None or dist_coeff_rv is None:
            raise ValueError(
                f"intrinsic_matrix and dist_coeff_rv cannot be None when calib_sensor is False")
        frame = Frame()
        frame.distortion = dist_coeff_rv
        frame.intrinsic_matrix = intrinsic_matrix
        calib_pose_list = []
        for position_infos in self.position_infos_list:
            image_pts = position_infos.image_points
            ret, rvec, tvec = cv2.solvePnP(
                self.pose_detector.obj_pts, image_pts, frame.intrinsic_matrix, frame.distortion_cv)
            if not ret:
                return False, intrinsic_matrix, dist_coeff_rv, hand_eye_result
            R, _ = cv2.Rodrigues(rvec)
            T_obj_2_cam = np.eye(4)
            T_obj_2_cam[:3, :3] = R
            T_obj_2_cam[:3, 3] = tvec[:, 0]
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

    def detect_calib_pose(self, frame, pose_robot_end=np.eye(4)):
        '''detect caliboard pose in robot base coordinate

        Args:
            frame (_type_): sensor frame, the calibrate result must be written into frame.pose
            pose_robot_end (_type_): use when eye in hand. 

        Returns:
            _type_: _description_
        '''
        ret, T_target_2_end, points_2d, _ = detect_caliboard_pose(
            frame, pose_detector=self.pose_detector)
        T_target_2_base = pose_robot_end @ T_target_2_end

        return ret, T_target_2_base, points_2d


class PositionInfos_3d:
    def __init__(self, calib_pose=None, T_end_2_base=np.eye(4), frame=None) -> None:
        self.calib_pose = calib_pose
        self.T_end_2_base = T_end_2_base
        self.frame = frame


class HandEyeCalibrateManager:
    def __init__(self) -> None:
        self.pose_detector = PoseDetector()
        self.position_infos_list = []  # list of PositionInfos_2d

    def set_circle_center_distance(self, circle_center_distance_mm):
        self.pose_detector.circle_center_distance_mm = circle_center_distance_mm
        self.pose_detector.updata_obj_points()

    def detect_calib_pose(self, frame, pose_in_cam: bool, pose_robot_end=np.eye(4)):

        ret, T_target_2_cam, points_2d, points_3d = self.pose_detector.compute_obj_pose(
            frame, use_pnp=frame.pm is None)
        pose = T_target_2_cam
        if not pose_in_cam:
            pose = pose_robot_end @ frame.pose @ T_target_2_cam
        return ret, pose, points_2d, points_3d

    def draw_points(self, frame, pose, pattern_was_found, image_pts):
        image = self.pose_detector.draw(
            frame=frame, image_pts=image_pts, pose=pose, pattern_was_found=pattern_was_found)
        return image

    def add_data(self, frame, pose_robot_end):
        position_infos = PositionInfos_3d()
        ret, pose, points_2d, points_3d = self.detect_calib_pose(
            frame, pose_in_cam=True)
        if ret:
            position_infos = PositionInfos_3d(
                calib_pose=pose, pose_robot_end=pose_robot_end, frame=copy.deepcopy(frame))
            self.position_infos_list.append(position_infos)

        return ret, position_infos

    def calibrate(self, eye_in_hand: bool):
        T_gripper_2_base = np.array(
            [position_infos.pose_robot_end for position_infos in self.position_infos_list])
        if not eye_in_hand:
            for i in range(T_gripper_2_base.shape[0]):
                T_gripper_2_base[i] = np.linalg.inv(T_gripper_2_base[i])
        T_target_2_cam = np.array(
            [position_infos.calib_pose for position_infos in self.position_infos_list])
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            T_gripper_2_base[:, :3, :3], T_gripper_2_base[:, :3, 3], T_target_2_cam[:, :3, :3], T_target_2_cam[:, :3, 3])
        hand_eye_result = np.eye(4)
        hand_eye_result[:3, :3] = R_cam2gripper
        hand_eye_result[:3, 3] = t_cam2gripper
        return True, hand_eye_result


if __name__ == "__main__":
    from src.sensor.sensor_manager import SensorManager
