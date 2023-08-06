import sys
sys.path.append('.')
import os

os.chdir('./src/push_DQN')
import numpy as np
import torch
import cv2
import yaml
from src.mask_rg.mask_rg import MaskRG
from src.push_DQN.robot import Robot
# import src.push_DQN.utils as utils


# # TODO FILE UNFINISHED

# def main():
#     min_num_obj = 26
#     max_num_obj = 32
#     heightmap_resolution = 0.002
#     workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])
#     # Initialize pick-and-place system (camera and robot)
#     robot = Robot(min_num_obj, max_num_obj, workspace_limits)
    
#     for i in range(30):
#         print('Pushing action ', i)
#         # Make sure simulation is still stable (if not, reset simulation)
#         robot.check_sim()

#         # Get latest RGB-D image
#         color_img, depth_img_raw = robot.get_camera_data()
#         # Detph scale is 1 for simulation!!
#         depth_img = depth_img_raw * robot.cam_depth_scale # Apply depth scale from calibration

#         # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
#         color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, robot.cam_intrinsics,
#                                                                robot.cam_pose, workspace_limits, heightmap_resolution)
#         valid_depth_heightmap = depth_heightmap.copy()
#         valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

os.chdir('../..')

from src.push_DQN.main_connector import push_the_scene

CONF_PATH = './push-DQN_config.yaml'

class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

with open(CONF_PATH) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

mask_rg = MaskRG(config['detection_thresholds']['confidence_threshold'], config['detection_thresholds']['mask_threshold'])
# Initialize pick-and-place system (camera and robot)
robot = Robot(config['environment']['min_num_objects'], config['environment']['max_num_objects'], np.asarray(config['environment']['workspace_limits']))


def get_segmentation():
    print('Pushing the objects first...')
    push_the_scene(mask_rg, robot)
    print('Now segmenting the objects...')
    # Get latest RGB-D image
    color_img, depth_img_raw = robot.get_camera_data()
    # Detph scale is 1 for simulation!!
    depth_img = depth_img_raw * robot.cam_depth_scale # Apply depth scale from calibration
    # convert img values to [0, 1]
    img_for_train = torch.tensor(depth_img).float() / 255
    img_for_train = [img_for_train.permute(2, 0, 1).to(mask_rg.model.device)]
    img_pred = mask_rg.model.eval_single_img(img_for_train)

    img_pred = img_pred[0]
    boxes = img_pred['boxes']
    scores = img_pred['scores']
    masks = img_pred['masks']
    rectangles = []  # list of (x1, y1, x2, y2), i.e. each object's bounding box
    binary_masks = []  # list of binary masks, i.e. each object's segmentation mask
    for box, score, mask in zip(boxes, scores, masks):
        if score < 0.9:
            continue
        x1, y1, x2, y2 = box.to(torch.int32).detach().cpu().numpy()
        rectangles.append((x1, y1, x2, y2))
        # Convert mask to binary image
        _, binary_mask = cv2.threshold(mask[0].detach().cpu().numpy(), 0.5, 1, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.uint8)
        binary_masks.append(binary_mask)

    return rectangles, binary_masks

if __name__ == '__main__':
    rectangles, binary_masks = get_segmentation()


