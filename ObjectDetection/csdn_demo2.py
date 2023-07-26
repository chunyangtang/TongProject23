# -*- coding:utf-8 -*-


import sys
import math
import time
import pygame
import sim
import numpy as np


class UR3_RG2:
    # variates
    resolutionX = 200   # 这里是pygame界面的窗口大小xy
    resolutionY = 200
    joint_angle = [0, 0, 0, 0, 0, 0]  # each angle of joint
    RAD2DEG = 180 / math.pi  # transform radian to degrees

    # Handles information
    jointNum = 6
    baseName = 'UR3'
    jointName = 'UR3_joint'

    # communication and read the handles
    def __init__(self):
        jointNum = self.jointNum
        baseName = self.baseName
        jointName = self.jointName

        print('Simulation started')

        try:

            sim.simxFinish(-1)  # 关掉之前连接
            clientID = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to CoppeliaSim
            if clientID != -1:
                print('connect successfully')
            else:
                sys.exit("Error: no se puede conectar")  # Terminar este script

        except:
            print('Check if CoppeliaSim is open')

        sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)  # 启动仿真
        print("Simulation start")

        # 读取Base和Joint的句柄
        jointHandle = np.zeros((jointNum, 1), dtype=np.int)
        for i in range(jointNum):
            _, returnHandle = sim.simxGetObjectHandle(clientID, jointName + str(i + 1), sim.simx_opmode_blocking)
            jointHandle[i] = returnHandle
            print(jointHandle[i])

        _, baseHandle = sim.simxGetObjectHandle(clientID, baseName, sim.simx_opmode_blocking)
        time.sleep(2)

        # 读取tip和target的句柄
        _, ikTipHandle = sim.simxGetObjectHandle(clientID, 'target', sim.simx_opmode_blocking)
        print('ikTipHandle:')
        print(ikTipHandle)
        _, connectionHandle = sim.simxGetObjectHandle(clientID, 'UR3_connection', sim.simx_opmode_blocking)
        print('connectionHandle:')
        print(connectionHandle)
        errorCode, targetPosition = sim.simxGetObjectPosition(clientID, ikTipHandle, -1,
                                                              sim.simx_opmode_blocking)
        print("target,targetPosition:")
        print(targetPosition)
        # 获取欧拉角
        errorCode, euler = sim.simxGetObjectOrientation(clientID, ikTipHandle, -1, sim.simx_opmode_blocking)

        # 读取每个关节角度
        jointConfig = np.zeros((jointNum, 1))
        for i in range(jointNum):
            _, jpos = sim.simxGetJointPosition(clientID, jointHandle[i], sim.simx_opmode_blocking)
            jointConfig[i] = jpos
            # print(jointConfig[i])

        self.clientID = clientID
        self.jointHandle = jointHandle
        self.ikTipHandle = ikTipHandle
        self.targetPosition = targetPosition
        self.jointConfig = jointConfig
        self.euler = euler

    def __del__(self):
        clientID = self.clientID
        sim.simxFinish(clientID)
        print('Simulation end')

    # show Handles information
    def showHandles(self):

        RAD2DEG = self.RAD2DEG
        jointNum = self.jointNum
        clientID = self.clientID
        jointHandle = self.jointHandle

        print('Handles available!')
        print("==============================================")
        print("Handles:  ")
        for i in range(len(jointHandle)):
            print("jointHandle" + str(i + 1) + ": " + jointHandle[i])
        print("===============================================")

    # show each joint's angle
    def showJointAngles(self):
        RAD2DEG = self.RAD2DEG
        jointNum = self.jointNum
        clientID = self.clientID
        jointHandle = self.jointHandle

        for i in range(jointNum):
            _, jpos = sim.simxGetJointPosition(clientID, jointHandle[i], sim.simx_opmode_blocking)
            print(round(float(jpos) * RAD2DEG, 2))
        print('\n')

    def StopSimulation(self):
        clientID = self.clientID
        sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)  # 关闭仿真
        sim.simxFinish(clientID)  # 关闭连接

    def move_x(self, length):                   # 沿着x正方向动一动
        clientID = self.clientID
        ikTipHandle = self.ikTipHandle
        targetPosition = self.targetPosition
        targetPosition[0] += length
        sim.simxSetObjectPosition(clientID, ikTipHandle, -1, targetPosition, sim.simx_opmode_oneshot)
        self.jointConfig = targetPosition

    def move_xx(self, length):                  # 沿着x负方向动一动
        clientID = self.clientID
        ikTipHandle = self.ikTipHandle
        targetPosition = self.targetPosition
        targetPosition[0] -= length
        sim.simxSetObjectPosition(clientID, ikTipHandle, -1, targetPosition, sim.simx_opmode_oneshot)
        self.jointConfig = targetPosition

    def move_y(self, length):
        clientID = self.clientID
        ikTipHandle = self.ikTipHandle
        targetPosition = self.targetPosition
        targetPosition[1] += length
        sim.simxSetObjectPosition(clientID, ikTipHandle, -1, targetPosition, sim.simx_opmode_oneshot)
        self.jointConfig = targetPosition

    def move_yy(self, length):
        clientID = self.clientID
        ikTipHandle = self.ikTipHandle
        targetPosition = self.targetPosition
        targetPosition[1] -= length
        sim.simxSetObjectPosition(clientID, ikTipHandle, -1, targetPosition, sim.simx_opmode_oneshot)
        self.jointConfig = targetPosition

    def move_z(self, length):
        clientID = self.clientID
        ikTipHandle = self.ikTipHandle
        targetPosition = self.targetPosition
        targetPosition[2] += length
        sim.simxSetObjectPosition(clientID, ikTipHandle, -1, targetPosition, sim.simx_opmode_oneshot)
        self.jointConfig = targetPosition

    def move_zz(self, length):
        clientID = self.clientID
        ikTipHandle = self.ikTipHandle
        targetPosition = self.targetPosition
        targetPosition[2] -= length
        sim.simxSetObjectPosition(clientID, ikTipHandle, -1, targetPosition, sim.simx_opmode_oneshot)
        self.jointConfig = targetPosition

    def move_rx(self, angle):                           # rx正方向转一转
        clientID = self.clientID
        ikTipHandle = self.ikTipHandle
        euler = self.euler
        euler[0] += angle
        sim.simxSetObjectOrientation(clientID, ikTipHandle, -1, euler, sim.simx_opmode_oneshot)

    def showtargetpositon(self):                        # 显示一下当前位置状态
        clientID = self.clientID
        ikTipHandle = self.ikTipHandle
        targetPosition = self.targetPosition

        _, ikTipHandle = sim.simxGetObjectHandle(clientID, 'target', sim.simx_opmode_blocking)
        errorCode, targetPosition = sim.simxGetObjectPosition(clientID, ikTipHandle, -1,
                                                              sim.simx_opmode_blocking)

        print("guoguoguo")
        print(targetPosition)


# control robot by keyboard
def main():
    robot = UR3_RG2()
    resolutionX = robot.resolutionX
    resolutionY = robot.resolutionY

    # angle = float(eval(input("please input velocity: ")))
    angle = 1
    length = 0.01

    # robot.initialize_target_position_tracking()

    pygame.init()
    screen = pygame.display.set_mode((resolutionX, resolutionY))
    screen.fill((255, 255, 255))
    pygame.display.set_caption("guo_Vrep")
    # 循环事件，按住一个键可以持续移动
    pygame.key.set_repeat(200, 50)

    while True:

        pygame.display.update()

        key_pressed = pygame.key.get_pressed()
        for event in pygame.event.get():
            # 关闭程序
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    robot.StopSimulation()
                    sys.exit()



                # 按键部分

                # 小键盘1，2对应x轴正负
                # 小键盘4，5对应y轴正负
                # 小键盘7，8对应z轴正负
                # 小键盘3对应rx正
                #
                # 后续需要把rx ry rz加上



                elif event.key == pygame.K_KP1:
                    robot.move_x(length)
                elif event.key == pygame.K_KP2:
                    robot.move_xx(length)

                elif event.key == pygame.K_KP4:
                    robot.move_y(length)
                elif event.key == pygame.K_KP5:
                    robot.move_yy(length)

                elif event.key == pygame.K_KP7:
                    robot.move_z(length)
                elif event.key == pygame.K_KP8:
                    robot.move_zz(length)
                elif event.key == pygame.K_KP3:
                    robot.move_rx(angle)
                elif event.key == pygame.K_g:
                    robot.showtargetpositon()


                else:
                    print("Invalid input, no corresponding function for this key!")


if __name__ == '__main__':
    main()
