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


os.chdir('../..')

from src.push_DQN.main_connector import push_the_scene

CONF_PATH = './push-DQN_config.yaml'

with open(CONF_PATH) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

mask_rg = MaskRG(config['detection_thresholds']['confidence_threshold'], config['detection_thresholds']['mask_threshold'])
# Initialize pick-and-place system (camera and robot)
robot = Robot(config['environment']['min_num_objects'], config['environment']['max_num_objects'], np.asarray(config['environment']['workspace_limits']))


def get_segmentation():
    """
    Get segmentation of the objects in the current scene base on trained Mask-RCNN model.
    Returns:
        rectangles: list of (x1, y1, x2, y2), i.e. each object's bounding box (left-top and right-bottom corners)
        binary_masks: list of binary masks, i.e. each object's segmentation mask (0 for background, 1 for object)
    """
    print('Pushing the objects first...')
    push_the_scene(mask_rg, robot, 0)
    print('Now segmenting the objects...')
    # Get the depth image
    color_m_rg, depth_m_rg, [segmentation_mask, num_objects] = robot.get_data_mask_rg()
    # Imitating the way to process depth data in MaskRG.set_reward_generator
    # Preprocess depth image
    depth_image = depth_m_rg / 20 / 255
    depth_image = np.repeat(depth_image.reshape(1024, 1024, 1), 3, axis=2)
    depth_image = depth_image.transpose(2, 0, 1)
    # Get segmentation
    with torch.no_grad():
        depth_tensor = torch.tensor(depth_image).to(mask_rg.model.device).float()
        img_pred = mask_rg.model.eval_single_img([depth_tensor])

    # Get the bounding boxes and segmentation masks
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


