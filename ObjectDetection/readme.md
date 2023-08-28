# Object Detection

**This is the folder that tangcy practice the basic usage of v-rep and a tentative object detection algorithm implementation using deep-mask and Learning Instance Segmentation by Interaction.**


## 1. V-REP
- `beginner.ttt` is the v-rep file that used for basic v-rep usage practicing.
- `UR5_Controller.py` is the python script that used for controlling the UR5 robot in v-rep.
- `csdn_demo.py` & `csdn_demo2.py` are the referenced code found at CSDN, which are used for controlling the UR5 robot in v-rep.

## 2. Object Detection
**Notice that this part of code is NOT WORKING!**
- `deepmask-pytorch` is a folder that contains original deepmask implementation.
- `utils` folder, `DeepMask.py`, `main.py` are the files that I modified from original deepmask, aiming at utilizing them to implement "*Learning Instance Segmentation by Interaction*".
- `robust_set_loss.py` is the loss function provided by "*Learning Instance Segmentation by Interaction*".