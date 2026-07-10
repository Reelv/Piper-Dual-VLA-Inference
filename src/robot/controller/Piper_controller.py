

import importlib
import os
import sys

from robot.controller.arm_controller import ArmController
import numpy as np
import time

SDK_ROOT_CANDIDATES = [
    os.environ.get("PIPER_SDK_ROOT"),
    os.path.expanduser("~/Lzr/piper_sdk"),
    os.path.expanduser("~/Lzr/piper_sdk/piper_sdk"),
    os.path.expanduser("~/piper_sdk"),
    os.path.expanduser("~/piper_sdk/piper_sdk"),
]
for path in SDK_ROOT_CANDIDATES:
    if path and os.path.isdir(path) and path not in sys.path:
        sys.path.append(path)

try:
    _piper_sdk = importlib.import_module("piper_sdk")
    C_PiperInterface_V2 = _piper_sdk.C_PiperInterface_V2
except Exception as exc:
    raise ImportError(
        "Unable to import piper_sdk. Install it with pip or add SDK path to SDK_ROOT_CANDIDATES."
    ) from exc

'''
Piper base code from:
https://github.com/agilexrobotics/piper_sdk.git
'''

class PiperController(ArmController):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.controller_type = "user_controller"
        self.controller = None
        self.min_eef_z = 0.1275
    
    def set_up(self, can:str):
        piper = C_PiperInterface_V2(can)
        piper.ConnectPort()
        piper.EnableArm(7)
        enable_fun(piper=piper)
        self.controller = piper

    def reset(self, start_state):
        try:
            self.set_joint(start_state)
        except :
            print(f"reset error")
        return

    # 返回单位为米
    def get_state(self):
        state = {}
        eef = self.controller.GetArmEndPoseMsgs()
        joint = self.controller.GetArmJointMsgs()
        
        state["joint"] = np.array([joint.joint_state.joint_1, joint.joint_state.joint_2, joint.joint_state.joint_3,\
                                   joint.joint_state.joint_4, joint.joint_state.joint_5, joint.joint_state.joint_6]) * 0.001 / 180 * 3.1415926
        state["qpos"] = np.array([eef.end_pose.X_axis, eef.end_pose.Y_axis, eef.end_pose.Z_axis, \
                                  eef.end_pose.RX_axis, eef.end_pose.RY_axis, eef.end_pose.RZ_axis]) * 0.001 / 1000
        state["gripper"] = self.controller.GetArmGripperMsgs().gripper_state.grippers_angle * 0.001 / 70
        return state

    # All returned values are expressed in meters,if the value represents an angle, it is returned in radians
    def set_position(self, position):
        pos = np.array(position, dtype=np.float32).copy()
        if pos.shape[0] >= 3 and pos[2] < 0.1275:
            pos[2] = 0.1275

        x, y, z, rx, ry, rz = pos * 1000 * 1000
        x, y, z, rx, ry, rz = int(x), int(y), int(z), int(rx), int(ry), int(rz)

        self.controller.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self.controller.EndPoseCtrl(x, y, z, rx, ry, rz)
    
    def set_joint(self, joint):
        j1, j2, j3 ,j4, j5, j6 = joint * 57295.7795 #1000*180/3.1415926
        j1, j2, j3 ,j4, j5, j6 = int(j1), int(j2), int(j3), int(j4), int(j5), int(j6)
        
        self.controller.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        self.controller.JointCtrl(j1, j2, j3, j4, j5, j6)
        self._clamp_eef_z()

    # The input gripper value is in the range [0, 1], representing the degree of opening.
    def set_gripper(self, gripper):
        gripper = int(gripper * 70 * 1000)
        self.controller.GripperCtrl(gripper, 1000, 0x01, 0)

    def __del__(self):
        try:
            if hasattr(self, 'controller'):
                # Add any necessary cleanup for the arm controller
                pass
        except:
            pass

    def _clamp_eef_z(self):
        try:
            state = self.get_state()
            qpos = state.get("qpos")
            if qpos is None:
                return
            if qpos[2] < self.min_eef_z:
                qpos = np.array(qpos, dtype=np.float32)
                qpos[2] = self.min_eef_z
                self.set_position(qpos)
        except Exception:
            pass

def enable_fun(piper:C_PiperInterface_V2):
    enable_flag = False
    timeout = 5
    start_time = time.time()
    elapsed_time_flag = False
    while not (enable_flag):
        elapsed_time = time.time() - start_time
        print("--------------------")
        enable_flag = piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
        print("enable flag:",enable_flag)
        piper.EnableArm(7)
        piper.GripperCtrl(0,1000,0x01, 0)

        print("--------------------")
        if elapsed_time > timeout:
            print("time out....")
            elapsed_time_flag = True
            enable_flag = True
            break
        time.sleep(1)
        pass
    if(elapsed_time_flag):
        print("time out, exit!")
        exit(0)

if __name__=="__main__":
    controller = PiperController("test_piper")
    controller.set_up("can0")
    print(controller.get_state())
    
    controller.set_gripper(0.2)
    controller.set_joint(np.array([0.1,0.1,-0.2,0.3,-0.2,0.5]))
    time.sleep(1)
    print(controller.get_gripper())
    print(controller.get_state())

    controller.set_position(np.array([0.057, 0.0, 0.260, 0.0, 0.085, 0.0]))
    time.sleep(1)
    print(controller.get_gripper())
    print(controller.get_state())
