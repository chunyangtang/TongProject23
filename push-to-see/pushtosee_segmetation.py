import sys
sys.path.append('.')
import os
os.chdir('./src/push_DQN')
import numpy as np
from src.push_DQN.robot import Robot
import src.push_DQN.utils as utils


# TODO FILE UNFINISHED

def main():
    min_num_obj = 26
    max_num_obj = 32
    heightmap_resolution = 0.002
    workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])
    # Initialize pick-and-place system (camera and robot)
    robot = Robot(min_num_obj, max_num_obj, workspace_limits)
    
    for i in range(30):
        print('Pushing action ', i)
        # Make sure simulation is still stable (if not, reset simulation)
        robot.check_sim()

        # Get latest RGB-D image
        color_img, depth_img_raw = robot.get_camera_data()
        # Detph scale is 1 for simulation!!
        depth_img = depth_img_raw * robot.cam_depth_scale # Apply depth scale from calibration

        # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
        color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, robot.cam_intrinsics,
                                                               robot.cam_pose, workspace_limits, heightmap_resolution)
        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0
