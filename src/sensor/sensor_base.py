#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) RVBUST, Inc - All rights reserved.

from enum import IntEnum
import cv2
import numpy as np
import copy
import time


class SensorType(IntEnum):
    Unknown = 1 << 0
    File = 1 << 1
    RVC = 1 << 2
    Hik_2D = 1 << 3,


class DeviceInfos:
    def __init__(self) -> None:
        self.sn = ""
        self.sensor_name = ""
        self.support_x_mode = []


class SensorBase:
    def __init__(self) -> None:
        self.type = SensorType.Unknown
        pass

    def connect(self, **kwargs) -> bool:
        return True

    def disconnect(self):
        pass

    def is_connected(self):
        return False

    def open(self, **kwargs):
        pass

    def is_opened(self):
        return False

    def close(self):
        pass

    def capture(self):
        pass

    def set_config(self, **kwargs):
        pass

    def get_config(self, key):
        pass

    def post_process(self, frame, post_process_config):
        src_mask = copy.deepcopy(frame.mask)
        for key in post_process_config:
            t0 = time.time()
            if key == "crop_by_plane":
                param = post_process_config[key]
                if param.get("enable", True):
                    plane_center_mm = np.array(
                        param.get("plane_center_mm", [0, 0, 0]))
                    plane_normal = np.array(
                        param.get("plane_normal", [0, 0, 1]))
                    crop_distance_range_mm = np.array(param.get(
                        "crop_distance_range_mm", [-5000, 5000]))
                    cv2.cropByPlane(frame.pm, frame.mask, plane_center_mm,
                                    plane_normal, crop_distance_range_mm)
            if key == "dbscan_removal":
                param = post_process_config[key]
                if param.get("enable", True):
                    distance_threshold_mm = param.get(
                        "distance_threshold_mm", 3)
                    min_points = param.get("min_points", 3)
                    min_num_cluster = param.get("min_num_cluster", 100)

                    cv2.pointmapRemoveClusterOutliers(
                        frame.pm, frame.mask, distance_threshold_mm, min_points, min_num_cluster)
            if key == "erode":
                param = post_process_config[key]
                if param.get("enable", True):
                    kernel_size = (param.get("kernel_w", 3),
                                   param.get("kernel_h", 3))
                    kernel = np.ones(kernel_size, dtype=np.uint8)
                    iterations = param.get("iterations", 1)
                    frame.mask = cv2.erode(
                        frame.mask, kernel, iterations=iterations)
            if key == "dilate":
                if param.get("enable", True):
                    param = post_process_config[key]
                    kernel_size = (param.get("kernel_w", 3),
                                   param.get("kernel_h", 3))
                    kernel = np.ones(kernel_size, dtype=np.uint8)
                    iterations = param.get("iterations", 1)
                    frame.mask = cv2.dilate(
                        frame.mask, kernel, iterations=iterations)
            if key == "area_filter_2d":
                param = post_process_config[key]
                if param.get("enable", True):
                    area_threshold_lower = param.get("area_threshold_lower", 1)
                    area_threshold_upper = param.get(
                        "area_threshold_upper", np.float('inf'))

                    contours, _hierarchy = cv2.findContours(
                        frame.mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area < area_threshold_lower or area > area_threshold_upper:
                            cv2.drawContours(frame.mask, [contour], 0, 0, -1)

            if key == "estimate_normal":
                param = post_process_config[key]
                if param.get("enable", True):
                    T_cam_2_pcd = np.linalg.inv(frame.extrinsic_matrix)
                    frame.normals = cv2.estimatePointMapNormal(
                        frame.pm, frame.mask, param["search_radius_pixel"], T_cam_2_pcd[:3, 3])
            t1 = time.time()
            print(f"{key} times(s): {t1 - t0}")
        frame.mask[src_mask == 0] = 0
        return frame
