

import os

import numpy as np

from robot.robot.base_robot import Robot

from robot.controller.Piper_controller import PiperController
from robot.sensor.Realsense_sensor import RealsenseSensor

from robot.data.collect_any import CollectAny

# Configure RealSense serial numbers through environment variables before running
# real-robot inference. Do not commit machine-specific serial numbers.
CAMERA_SERIALS = {
    'head': os.environ.get('PIPER_CAM_HEAD_SERIAL', 'REPLACE_WITH_HEAD_CAMERA_SERIAL'),
    'left_wrist': os.environ.get('PIPER_CAM_LEFT_WRIST_SERIAL', 'REPLACE_WITH_LEFT_WRIST_CAMERA_SERIAL'),
    'right_wrist': os.environ.get('PIPER_CAM_RIGHT_WRIST_SERIAL', 'REPLACE_WITH_RIGHT_WRIST_CAMERA_SERIAL'),
}

# Define start position (in degrees)
START_POSITION_ANGLE_LEFT_ARM = [
    0,   # Joint 1
    0,    # Joint 2
    0,  # Joint 3
    0,   # Joint 4
    0,  # Joint 5
    0,    # Joint 6
]

# Define start position (in degrees)
START_POSITION_ANGLE_RIGHT_ARM = [
    0,   # Joint 1
    0,    # Joint 2
    0,  # Joint 3
    0,   # Joint 4
    0,  # Joint 5
    0,    # Joint 6
]

condition = {
    "save_path": "./save/",
    "task_name": "test",
    "save_format": "hdf5",
    "save_freq": 10, 
}

class PiperDual(Robot):
    def __init__(self, condition=condition, move_check=True, start_episode=0):
        super().__init__(condition=condition, move_check=move_check, start_episode=start_episode)

        self.controllers = {
            "arm":{
                "left_arm": PiperController("left_arm"),
                "right_arm": PiperController("right_arm"),
            }
        }
        self.sensors = {
            "image": {
                "cam_head": RealsenseSensor("cam_head"),
                "cam_left_wrist": RealsenseSensor("cam_left_wrist"),
                "cam_right_wrist": RealsenseSensor("cam_right_wrist"),
            },
        }

    def reset(self):
        self.controllers["arm"]["left_arm"].reset(START_POSITION_ANGLE_LEFT_ARM)
        self.controllers["arm"]["right_arm"].reset(START_POSITION_ANGLE_RIGHT_ARM)

    def set_up(self):
        super().set_up()
        self.controllers["arm"]["left_arm"].set_up("can_left")#can0和can1在另一个工具里设置，注意和Piper_controller.py里一致
        self.controllers["arm"]["right_arm"].set_up("can_right")

        self.sensors["image"]["cam_head"].set_up(CAMERA_SERIALS['head'], is_depth=False)
        self.sensors["image"]["cam_left_wrist"].set_up(CAMERA_SERIALS['left_wrist'], is_depth=False)
        self.sensors["image"]["cam_right_wrist"].set_up(CAMERA_SERIALS['right_wrist'], is_depth=False)

        self.set_collect_type({"arm": ["joint","qpos","gripper"],
                               "image": ["color"]
                               })
        print("set up success!")

if __name__ == "__main__":
    import time
    
    robot = PiperDual()
    robot.set_up()

    # collection test
    data_list = []
    for i in range(100):
        print(i)
        data = robot.get()
        robot.collect(data)
        time.sleep(0.1)
    robot.finish()
    
    # moving test
    move_data = {
        "arm":{
            "left_arm":{
            "qpos":[0.057, 0.0, 0.216, 0.0, 0.085, 0.0, 0.057, 0.0, 0.216, 0.0, 0.085, 0.0],
            "gripper":0.8,
            },
        }
    }
    robot.move(move_data)
    
    move_data = {
        "arm":{
            "left_arm":{
            "qpos":[0.057, 0.0, 0.216, 0.0, 0.085, 0.0, 0.057, 0.0, 0.216, 0.0, 0.085, 0.0],
            "gripper":0.8,
            },
        }
    }
    robot.move(move_data)
