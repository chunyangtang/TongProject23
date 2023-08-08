import sys
sys.path.append('.')
sys.path.append('./push_to_see/src/push_DQN')
sys.path.append('./push_to_see/src/mask_rg')
sys.path.append('./push_to_see/src')
sys.path.append('./push_to_see')
from push_to_see.pushtosee_segmentation import get_segmentation
from push_to_see.src.push_DQN.robot import Robot

# Using Robot with default parameters (in given Task_no_*)
robot = Robot()
# Perform segmentation, push_times normally required to be 30.
rectangles, binary_masks = get_segmentation(robot=robot, push_times=1)