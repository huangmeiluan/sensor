#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) RVBUST, Inc - All rights reserved.

import time
import socket
import struct
import numpy as np
from scipy.spatial.transform import Rotation

np.set_printoptions(precision=4, suppress=True)

__all__ = [
    "RPController"
]


class RPController:
    """Rotary Platform Controller
    NOTE:
    Position unit: degree of the Rotary Platform
    Speed unit: degree / second of the Rotary Platform
    """
    PREFIX = struct.pack("7B", 0, 0, 0, 0, 0, 0x06, 0x01)
    READ_MOTOR_STATUS = PREFIX + \
        struct.pack("5B", 0x03, 0x00, 0x01, 0x00, 0x01)
    READ_MOTOR_PULSES = PREFIX + \
        struct.pack("5B", 0x03, 0x00, 0x08, 0x00, 0x02)
    FORWARD_MOVE = PREFIX + struct.pack("5B", 0x06, 0x00, 0x12, 0x00, 0x03)
    BACKWARD_MOVE = PREFIX + struct.pack("5B", 0x06, 0x00, 0x12, 0x00, 0x04)
    STOP_MOVE = PREFIX + struct.pack("5B", 0x06, 0x00, 0x12, 0x00, 0x05)
    STEP_FORWARD_MOVE = struct.pack("12B", 0x00, 0x00, 0x00,
                                    0x00, 0x00, 0x06, 0x01, 0x06, 0x00, 0x12, 0x00, 0x01)
    STEP_BACKWARD_MOVE = struct.pack("12B", 0x00, 0x00, 0x00,
                                     0x00, 0x00, 0x06, 0x01, 0x06, 0x00, 0x12, 0x00, 0x02)

    ##
    LIGHT_START = PREFIX + struct.pack("5B", 0x06, 0x00, 0x42, 0x00, 0x10)
    LIGHT_STOP = PREFIX + struct.pack("5B", 0x06, 0x00, 0x42, 0x00, 0x00)

    # 180000 pulses per table loop, note no need to multiply REDUCTION_RATIO again
    PULSES_X_LOOP = 180000
    REDUCTION_RATIO = 45

    def __init__(self, ip_addr: str = "192.168.0.104", port: int = 502,
                 base_pose: np.ndarray = np.eye(4), axis: np.ndarray = np.array([0., 0., 1.])):
        """Initialize the RotaryPlatform

        Args:
            ip_addr (str, optional): IP address of the Rotary Platform Modbus driver. Defaults to "192.168.0.104".
            port (int, optional): Modbus comunication port. Defaults to 502.
            base_pose ([type], optional): rotary platform base pose in world coordinate. Defaults to np.eye(4).
            axis ([type], optional): rotary axis in local coordinate, the axis is assumed to cross the origin of local frame. Defaults to np.array([0., 0., 1.]).
        """
        self.ip_addr = ip_addr
        self.port = port
        self.__base_pose = np.asarray(base_pose)
        self.__axis = np.asarray(axis)

        self.sock = None
        self.m_is_connected = False
        self.m_is_working = False

    def __del__(self):
        self.disconnect()

    @property
    def base_pose(self):
        return self.__base_pose

    def connect(self, ip_addr=None):
        # self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # ret = self.sock.connect((self.ip_addr, self.port))
        # return ret
        if ip_addr is not None:
            self.ip_addr = ip_addr

        if self.m_is_connected:
            return True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5)
        try:
            self.sock.connect((self.ip_addr, self.port))
        except Exception as error:
            print(error)
            self.sock.close()
            self.sock = None
            return False
        self.m_is_connected = True
        return True

    def disconnect(self):
        if not self.m_is_connected:
            return True
        if self.sock is not None:
            self.stop()
            self.sock.close()
            self.sock = None
        self.m_is_connected = False
        return True

    def is_connected(self):
        return self.m_is_connected

    def is_working(self):
        return self.m_is_working

    def get_pose(self):
        """Get rotary platform pose in world coordinate

        Returns:
            np.ndarray: 4x4 transform
        """
        pos_rad = np.deg2rad(self.read_rp_angle())
        R = Rotation.from_rotvec(pos_rad * self.__axis)
        pose = np.array(self.base_pose)  # copy
        pose[:3, :3] = pose[:3, :3] @ R.as_matrix()
        return pose

    def enable_light(self, able=False):
        if not self.m_is_connected:
            return
        if able:
            self.sock.send(self.LIGHT_START)
        else:
            self.sock.send(self.LIGHT_STOP)

        try:
            self.sock.recv(12)
        except Exception as ee:
            print("StartLight fail: {}".format(ee))

    def start(self, reverse=False):
        """start continuous movement

        Args:
            reverse (bool, optional): True: anti-clockwise. False: clock-wise

        Returns:
            [type]: [description]
        """
        if not self.m_is_connected:
            return None
        if reverse:
            self.sock.send(self.BACKWARD_MOVE)
        else:
            self.sock.send(self.FORWARD_MOVE)
        return self.sock.recv(12)

    def stop(self):
        """stop movement

        Returns:
            [type]: [description]
        """
        if not self.m_is_connected:
            return None
        self.sock.send(self.STOP_MOVE)
        data = None
        try:
            data = self.sock.recv(12)
        except Exception as ee:
            print("Rot stop fail: {}".format(ee))
        return data

    def set_jog_speed(self, speed: float):
        """Set table rotation speed

        Args:
            speed (float): degree / second

        Returns:
            [type]: [description]
        """
        if not self.m_is_connected:
            return None
        if speed < 0:
            raise ValueError(f"speed {speed} is < 0 !")
        rpm_motor = round(speed * self.REDUCTION_RATIO * 60 / 360)
        signal = self.PREFIX + \
            struct.pack("3B", 0x06, 0x00, 0x4D) + struct.pack(">H", rpm_motor)
        self.sock.send(signal)
        return self.sock.recv(12)

    def set_step_speed(self, speed: float):
        """Set rotation speed

        Args:
            speed (float): degree / second

        Returns:
            [type]: [description]
        """
        if not self.m_is_connected:
            return None
        if speed < 0:
            raise ValueError(f"speed {speed} is < 0 !")
        rpm_motor = round(speed * self.REDUCTION_RATIO * 60 / 360)
        signal = self.PREFIX + \
            struct.pack("3B", 0x06, 0x00, 0x48) + struct.pack(">H", rpm_motor)
        self.sock.send(signal)
        data = None
        try:
            data = self.sock.recv(12)
        except Exception as ee:
            print("rot set_step_speed fail: {}".format(ee))
        return data

    def read_rp_angle(self, force_range=True):
        """Read current table position

        Returns:
            [type]: [description]
        """
        if not self.m_is_connected:
            return None
        self.sock.send(self.READ_MOTOR_PULSES)
        res = self.sock.recv(13)
        pulses_pos = struct.unpack(">i", res[-2:] + res[-4:-2])[0]

        angle_deg = pulses_pos / self.PULSES_X_LOOP * 360
        if force_range:
            angle_deg = angle_deg % 360
            tmp = angle_deg % 360
            angle_deg = tmp if abs(tmp) < abs(tmp - 360) else tmp - 360
        return angle_deg

    def is_moving(self):
        if not self.m_is_connected:
            return None
        self.sock.send(self.READ_MOTOR_STATUS)
        res = self.sock.recv(11)
        status = int.from_bytes(res[-2:], 'big')
        return bool(status & 0b1000)

    def move_home(self):
        # Home position is the position when power on
        return self.step_move(-self.read_rp_angle())

    def step_move(self, angle_in_deg: float, wait: bool = True):
        if not self.m_is_connected:
            return None

        self.m_is_working = True
        pulses = int(angle_in_deg / 360 * self.PULSES_X_LOOP)
        pulses_bytes = struct.pack(">i", pulses)
        high_digits = struct.pack(
            "10B", 0, 0, 0, 0, 0, 0x06, 0x01, 0x06, 0x00, 0x4A) + pulses_bytes[:2]
        low_digits = struct.pack(
            "10B", 0, 0, 0, 0, 0, 0x06, 0x01, 0x06, 0x00, 0x49) + pulses_bytes[2:]
        self.sock.send(high_digits)
        self.sock.recv(12)
        self.sock.send(low_digits)
        self.sock.recv(12)
        pos1 = self.read_rp_angle()
        if angle_in_deg < 0:
            self.sock.send(self.STEP_BACKWARD_MOVE)
        else:
            self.sock.send(self.STEP_FORWARD_MOVE)
        self.sock.recv(12)
        if not wait:
            self.m_is_working = False
            return True
        count = 0
        while self.is_moving():
            time.sleep(0.2)
            count += 1
            if count > 20 * 5:
                break
        pos2 = self.read_rp_angle()

        self.m_is_working = False
        return abs((pos2 - pos1) - angle_in_deg) < 1e-5

if __name__ == "__main__":
    from IPython import embed

    c = RPController(ip_addr="10.10.10.10")
    embed()
