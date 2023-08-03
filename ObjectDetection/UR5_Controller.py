# Created by TCY on 2023.7.14


import sys
sys.path.append(r"C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\programming\zmqRemoteApi\clients\python")

from coppeliasim_zmqremoteapi_client import *
import time
import os
from typing import List

import numpy as np
import cv2

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

# Define some utility functions
def RAD2DEG(rad):
    return rad * 180 / np.pi

def DEG2RAD(deg):
    return deg * np.pi / 180


class UR5_Controller:
    """A class for controlling the UR5 robot in CoppeliaSim"""
    def __init__(self) -> None:
        # Set up the remote API
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')

        # When simulation is not running, ZMQ message handling could be a bit
        # slow, since the idle loop runs at 8 Hz by default. So let's make
        # sure that the idle loop runs at full speed for this program:
        self.defaultIdleFps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)

        # Start the simulation
        self.sim.startSimulation()

        # Get the handles
        self.ur5Handle: int = self.sim.getObject('/UR5')
        self.ur5jointHandles: List[int] = [self.sim.getObject('/joint', {'index': i}) for i in range(6)]
        self.rg2Handle: int = self.sim.getObject('/UR5/RG2')
        self.rg2ScriptHandle: int = self.sim.getScript(self.sim.scripttype_childscript, self.rg2Handle)
        self.camerargbHandle: int = self.sim.getObject('/rgb')
        self.cameradepthHandle: int = self.sim.getObject('/depth')
        self.tipHandle: int = self.sim.getObject('/UR5/TipDummy')
        self.targetHandle: int = self.sim.getObject('/TargetDummy')


        # Get the initial parameters
        self.jointConfig: List[float] = [0, 0, 0, 0, 0, 0]
        for i in range(6):
            self.jointConfig[i] = self.sim.getJointPosition(self.ur5jointHandles[i])

        [self.resX, self.resY] = self.sim.getVisionSensorResolution(self.camerargbHandle)

        self.tipPosition = self.sim.getObjectPosition(self.tipHandle, self.sim.handle_world)
        self.tipOrientation = self.sim.getObjectOrientation(self.tipHandle, self.sim.handle_world)

    def __del__(self):
        self.sim.stopSimulation()
        # If you need to make sure we really stopped:
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.1)

        # Restore the original idle loop frequency:
        self.sim.setInt32Param(self.sim.intparam_idle_fps, self.defaultIdleFps)

    def updateJointAngle_Absolute(self, newJointConfig: List[float]):
        newJointConfig = [DEG2RAD(i) for i in newJointConfig]
        self.jointConfig.clear()
        for i in range(6):
            self.jointConfig.append(newJointConfig[i])
            self.sim.setJointTargetPosition(self.ur5jointHandles[i], self.jointConfig[i])

    def updateJointAngle_Relative(self, newJointConfig: List[float]):
        newJointConfig = [DEG2RAD(i) for i in newJointConfig]
        for i in range(6):
            self.jointConfig[i] = self.jointConfig[i] + newJointConfig[i]
            self.sim.setJointTargetPosition(self.ur5jointHandles[i], self.jointConfig[i])

    def getJointAngle(self):
        return [RAD2DEG(i) for i in self.jointConfig]
    
    def rotateAJoint(self, jointIndex: int, angle: float):
        newConfig = [0, 0, 0, 0, 0, 0]
        newConfig[jointIndex] = angle
        self.updateJointAngle_Relative(newConfig)
    
    def getCameraImage(self):
        img_rgb, _ = self.sim.getVisionSensorImg(self.camerargbHandle)
        img_depth, _ = self.sim.getVisionSensorImg(self.cameradepthHandle)

        img_rgb = np.frombuffer(img_rgb, dtype=np.uint8).reshape(self.resY, self.resX, 3)
        img_depth = np.frombuffer(img_depth, dtype=np.uint8).reshape(self.resY, self.resX, 3)

        # In CoppeliaSim images are left to right (x-axis), and bottom to top (y-axis)
        # (consistent with the axes of vision sensors, pointing Z outwards, Y up)
        # and color format is RGB triplets, whereas OpenCV uses BGR:
        img_rgb = cv2.flip(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB), 0)
        img_depth = cv2.flip(cv2.cvtColor(img_depth, cv2.COLOR_BGR2RGB), 0)
        img_depth = cv2.cvtColor(img_depth, cv2.COLOR_RGB2GRAY)

        return img_rgb, img_depth
    
    def openRG2(self):
        self.sim.callScriptFunction('RG2Open', self.rg2ScriptHandle, [], [], [], b'')

    def closeRG2(self):
        self.sim.callScriptFunction('RG2Close', self.rg2ScriptHandle, [], [], [], b'')

    def goToPosition_Absolute(self, position: List[float]):
        origPos, destPos = np.array(self.tipPosition), np.array(position)
        round = 100
        linspacePos = np.linspace(origPos, destPos, round)
        for i in range(1, round, 1):
            self.sim.setObjectPosition(self.targetHandle, self.sim.handle_world, linspacePos[i].tolist())
        
        for i in range(6):
            self.jointConfig[i] = self.sim.getJointPosition(self.ur5jointHandles[i])
        self.tipPosition = self.sim.getObjectPosition(self.tipHandle, self.sim.handle_world)

    def goToPosition_Relative(self, position: List[float]):
        position = [self.tipPosition[i] + position[i] for i in range(3)]
        self.goToPosition_Absolute(position)

    def rotateToAngle_Absolute(self, orientation: List[float]):
        self.sim.setObjectOrientation(self.targetHandle, self.sim.handle_world, orientation)

        for i in range(6):
            self.jointConfig[i] = self.sim.getJointPosition(self.ur5jointHandles[i])
        self.tipPosition = self.sim.getObjectPosition(self.tipHandle, self.sim.handle_world)
        self.tipOrientation = self.sim.getObjectOrientation(self.tipHandle, self.sim.handle_world)

    def rotateToAngle_Relative(self, orientation: List[float]):
        orientation = [self.tipOrientation[i] + orientation[i] for i in range(3)]
        self.rotateToAngle_Absolute(orientation)

def main_ManualCtrl():
    # Create the controller
    controller = UR5_Controller()

    path = r"C:\Users\t-c-y\Desktop\VREP_PROJECT"

    velocity = 1
    
    # Get the display
    pygame.init()
    screen = pygame.display.set_mode((controller.resX, controller.resY))
    screen.fill((255,255,255))
    pygame.display.set_caption("V-REP Remote API Control")
    # 循环事件，按住一个键可以持续移动
    pygame.key.set_repeat(200,50)

    

    # # prevent the program from exiting (press enter to exit)
    # while (t := input()) != "":
    #     time.sleep(0.1)
    
    while True:
        imgTempPath = os.path.join(path, "imgTemp", "frame.jpg")
        img_rgb, img_depth = controller.getCameraImage()
        cv2.imwrite(imgTempPath, img_rgb)
        ig = pygame.image.load(imgTempPath)

        #robot.arrayToDepthImage()
        #ig = pygame.image.load("imgTempDep\\frame.jpg")

        screen.blit(ig, (0, 0))
        pygame.display.update()
        
        key_pressed = pygame.key.get_pressed()
        for event in pygame.event.get():
            # 关闭程序
            if event.type == pygame.QUIT:
                print("exit")
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    print("exit")
                    sys.exit()
                # joint 0
                elif event.key == pygame.K_q:
                    print("q pressed")
                    controller.rotateAJoint(0, velocity)
                elif event.key == pygame.K_w:
                    print("w pressed")
                    controller.rotateAJoint(0, -velocity)
                # joint 1
                elif event.key == pygame.K_a:
                    print("a pressed")
                    controller.rotateAJoint(1, velocity)
                elif event.key == pygame.K_s:
                    print("s pressed")
                    controller.rotateAJoint(1, -velocity)
                # joint 2
                elif event.key == pygame.K_z:
                    print("z pressed")
                    controller.rotateAJoint(2, velocity)
                elif event.key == pygame.K_x:
                    print("x pressed")
                    controller.rotateAJoint(2, -velocity)
                # joint 3
                elif event.key == pygame.K_e:
                    print("e pressed")
                    controller.rotateAJoint(3, velocity)
                elif event.key == pygame.K_r:
                    print("r pressed")
                    controller.rotateAJoint(3, -velocity)
                # joint 4
                elif event.key == pygame.K_d:
                    print("d pressed")
                    controller.rotateAJoint(4, velocity)
                elif event.key == pygame.K_f:
                    print("f pressed")
                    controller.rotateAJoint(4, -velocity)
                # joint 5
                elif event.key == pygame.K_c:
                    print("c pressed")
                    controller.rotateAJoint(5, velocity)
                elif event.key == pygame.K_v:
                    print("v pressed")
                    controller.rotateAJoint(5, -velocity)
                # close RG2
                elif event.key == pygame.K_t:
                    print("t pressed")
                    controller.closeRG2()
                # open RG2
                elif event.key == pygame.K_y:
                    print("y pressed")
                    controller.openRG2()
                # reset angle
                elif event.key == pygame.K_l:
                    print("l pressed")
                    controller.updateJointAngle_Absolute([0, 0, 0, 0, 0, 0])
                else:
                    print("Invalid input, no corresponding function for this key!")

def main_IK():
    # Create the controller
    controller = UR5_Controller()

    # Get handles of global camera
    camRGBHandle = controller.sim.getObject('/kinect/rgb')
    camDepthHandle = controller.sim.getObject('/kinect/depth')
    [resX, resY] = controller.sim.getVisionSensorResolution(camRGBHandle)

    def getCameraImage(controller: UR5_Controller, camRGBHandle, camDepthHandle):
        img_rgb, _ = controller.sim.getVisionSensorImg(camRGBHandle)
        img_depth, _ = controller.sim.getVisionSensorImg(camDepthHandle)

        img_rgb = np.frombuffer(img_rgb, dtype=np.uint8).reshape(resY, resX, 3)
        img_depth = np.frombuffer(img_depth, dtype=np.uint8).reshape(resY, resX, 3)

        # In CoppeliaSim images are left to right (x-axis), and bottom to top (y-axis)
        # (consistent with the axes of vision sensors, pointing Z outwards, Y up)
        # and color format is RGB triplets, whereas OpenCV uses BGR:
        img_rgb = cv2.flip(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB), 0)
        img_depth = cv2.flip(cv2.cvtColor(img_depth, cv2.COLOR_BGR2RGB), 0)
        img_depth = cv2.cvtColor(img_depth, cv2.COLOR_RGB2GRAY)
        return img_rgb, img_depth


    cupHandle: int = controller.sim.getObject('/Cup')
    cupPosition: List[float] = controller.sim.getObjectPosition(cupHandle, controller.sim.handle_world)

    tipPosition: List[float] = controller.tipPosition


    controller.goToPosition_Absolute([cupPosition[0], cupPosition[1], cupPosition[2] + 0.02])
    controller.closeRG2()
    time.sleep(2)
    controller.goToPosition_Absolute([tipPosition[0], tipPosition[1]+0.2, tipPosition[2]])
    # controller.openRG2()

    # imgRGB, imgDepth = getCameraImage(controller, camRGBHandle, camDepthHandle)
    # cv2.imwrite("exps\\imgTemp\\rgb.jpg", imgRGB)
    # cv2.imwrite("exps\\imgTemp\\depth.jpg", imgDepth)

    # cuboidHandle: int = controller.sim.getObject('/Cuboid')
    # cuboidPosition: List[float] = controller.sim.getObjectPosition(cuboidHandle, controller.sim.handle_world)
    # tipPosition = controller.tipPosition

    # controller.goToPosition_Absolute([cuboidPosition[0], cuboidPosition[1], cuboidPosition[2] + 0.02])
    # controller.closeRG2()
    # time.sleep(2)
    # controller.goToPosition_Absolute(tipPosition)
    # controller.openRG2()

    # prevent the program from exiting (press enter to exit)
    while (t := input()) != "":
        time.sleep(0.1)



if __name__ == '__main__':
    main_IK()