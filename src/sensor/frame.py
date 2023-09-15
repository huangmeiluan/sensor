#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) RVBUST, Inc - All rights reserved.

import numpy as np
import cv2
import open3d as o3d
import os
import glob
import copy


class Frame:
    def __init__(self, pm=None, image=None, pose=np.eye(4)) -> None:
        self.pm = pm
        self.image = image
        self.extrinsic_matrix = np.eye(4)
        self.intrinsic_matrix = np.eye(3)
        self.distortion = np.zeros((5, ))
        self.pose = pose
        self.mask = None
        self.normals = None

    def extract_camera_param(self, param_a, w, h):
        if param_a[0] < 0:
            check_cam_param_num = np.array(
                [1, 2, 11, 13, 15, 17, 22, 23, 24, 25, 26, 27, 29, 30]) - 1
            param_a[check_cam_param_num] *= -1
            param_a[4] = h - 1 - param_a[4]
        self.intrinsic_matrix = np.eye(3)
        self.intrinsic_matrix[0, 0] = param_a[0]
        self.intrinsic_matrix[0, 1] = param_a[2]
        self.intrinsic_matrix[0, 2] = param_a[3]
        self.intrinsic_matrix[1, 1] = param_a[1]
        self.intrinsic_matrix[1, 2] = param_a[4]

        self.extrinsic_matrix = np.eye(4)
        self.extrinsic_matrix[0, :3] = param_a[18:21]
        self.extrinsic_matrix[1, :3] = param_a[21:24]
        self.extrinsic_matrix[2, :3] = param_a[24:27]
        self.extrinsic_matrix[:3, 3] = param_a[27:30]

        self.distortion = np.zeros((5, ))
        self.distortion[:3] = param_a[5:8]
        self.distortion[3] = param_a[11]
        self.distortion[4] = param_a[10]

    @property
    def distortion_cv(self):
        return np.array([self.distortion[0], self.distortion[1], self.distortion[3], self.distortion[4], self.distortion[2]])

    def load_from_file(self, pm_path=None, image_path=None, intrinsic_path=None, distortion_path=None, extrinsic_path=None, pose_path=None, param_535A_path=None, unit_scale=1, w=-1, h=-1, remove_nan_points=False):
        ret = True
        if pm_path is not None:
            pcd_o3d = o3d.io.read_point_cloud(
                pm_path, remove_nan_points=remove_nan_points)
            self.pm = np.asarray(pcd_o3d.points) * unit_scale
            self.pm[np.all(self.pm == 0, axis=1)] = np.nan  # TODO: delete
            self.pm = self.pm.astype(np.float32)

            if pcd_o3d.has_normals():
                self.normals = np.asarray(pcd_o3d.normals).astype(np.float32)

            self.mask = (~np.isnan(self.pm[:, 2])).astype(np.uint8)

        if image_path is not None:
            self.image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            h, w = self.image.shape[:2]
        if self.pm is not None and h > 0 and w > 0:
            self.pm = self.pm.reshape((h, w, 3))
            self.mask = self.mask.reshape((h, w))
            if self.normals is not None:
                self.normals = self.normals.reshape((h, w, 3))

        if param_535A_path is not None:
            param_a = np.loadtxt(param_535A_path)
            self.extract_camera_param(param_a, w, h)
        else:
            if intrinsic_path is not None:
                self.intrinsic_matrix = np.loadtxt(intrinsic_path)
            if extrinsic_path is not None:
                self.extrinsic_matrix = np.loadtxt(extrinsic_path)
            if distortion_path is not None:
                self.distortion = np.loadtxt(distortion_path)
        if pose_path is not None:
            self.pose = np.loadtxt(pose_path)
        return ret

    def save(self, save_dir="./", id=0, save_param=False, remove_invalid_point=False):
        os.makedirs(save_dir, exist_ok=True)
        if isinstance(id, int) and id < 0:
            id = np.max([len(glob.glob(f"{save_dir}/*.ply")),
                         len(glob.glob(f"{save_dir}/*.png"))])
        if self.pm is not None:
            pm = self.pm
            if remove_invalid_point:
                mask = self.mask.astype(bool)
                pm = copy.deepcopy(self.pm)
                pm[~mask] = np.nan

            pcd_o3d = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(pm.reshape((-1, 3))))
            if self.normals is not None:
                pcd_o3d.normals = o3d.utility.Vector3dVector(
                    self.normals.reshape((-1, 3)))
            o3d.io.write_point_cloud(f"{save_dir}/{id}.ply", pcd_o3d)
        if self.image is not None:
            cv2.imwrite(f"{save_dir}/{id}.png", self.image)

        if save_param:
            self.save_param(save_dir)

    def save_param(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        np.savetxt(f"{save_dir}/intrinsic_matrix.txt",
                   self.intrinsic_matrix)
        np.savetxt(f"{save_dir}/distortion.txt", self.distortion)
        np.savetxt(f"{save_dir}/extrinsic_matrix.txt",
                   self.extrinsic_matrix)
        np.savetxt(f"{save_dir}/pose.txt", self.pose)

    def get_transform_pm(self, T_b_w=np.eye(4), remove_invalid_points=True):
        new_pose = T_b_w @ self.pose @ self.extrinsic_matrix
        if remove_invalid_points:
            pcd = self.get_valid_points()
        else:
            pcd = self.pm.reshape((-1, 3))
        new_pcd = (np.dot(new_pose[:3, :3], pcd.T).T +
                   new_pose[:3, 3]).astype(np.float32)
        return new_pcd

    def get_valid_points(self):
        if self.pm is None:
            return np.array([], dtype=np.float32).reshape((-1, 3))
        if self.mask is None:
            self.mask = (~np.isnan(self.pm[:, :, 2])).astype(np.uint8)
        return self.pm[(self.mask).astype(bool)]

    def transform(self, T_b_w=np.eye(4)):
        T_cam_2_base = T_b_w @ self.pose
        T_pcd_2_base = T_cam_2_base @ self.extrinsic_matrix
        frame = copy.deepcopy(self)
        if frame.pm is not None:
            shape = frame.pm.shape
            points = self.pm.reshape((-1, 3))
            frame.pm = (np.dot(T_pcd_2_base[:3, :3], points.T).T +
                        T_pcd_2_base[:3, 3]).astype(np.float32).reshape(shape)
        if frame.normals is not None:
            shape = frame.normals.shape
            normals = self.normals.reshape((-1, 3))
            frame.normals = (np.dot(T_pcd_2_base[:3, :3], normals.T).T).astype(
                np.float32).reshape(shape)
        frame.pose = T_cam_2_base
        frame.extrinsic_matrix = np.linalg.inv(T_cam_2_base)
        return frame
