
import argparse
import os
from src.sensor.sensor_base import SensorType
import PyRVC as RVC


def get_common_parse(description="sensor utils"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config_sensor_path", help="the path of config_sensor.json, default = %(default)s", default="./data/sensor/config_sensor.json")
    parser.add_argument(
        "sn_list", help="used sensor serial number", nargs="+")
    parser.add_argument(
        "--file_camera_dir", help="file camera directory, default = %(default)s", default=f'''{os.environ["HOME"]}/Rvbust/Data/sensor''')
    parser.add_argument(
        "--list_sensor_type", help=f"list sensor type, {SensorType.RVC.value}: {SensorType.RVC.name}, {SensorType.File.value}: {SensorType.File.name}, {SensorType.Hik_2D.value}: {SensorType.Hik_2D.name}", type=int, default=0xfffff)
    parser.add_argument(
        "--rvc_list_device_type", help=f"rvc list device type, all: {RVC.SystemListDeviceTypeEnum.All.name}, usb: {RVC.SystemListDeviceTypeEnum.USB.name}, gige: {RVC.SystemListDeviceTypeEnum.GigE.name}", default=RVC.SystemListDeviceTypeEnum.All.name)

    return parser


if __name__ == "__main__":
    from IPython import embed
    parser = get_common_parse("test")
    parser.add_argument("circle_center_distance_mm",
                        help="the distance of circle center", type=float)
    args = parser.parse_args()
    embed()
