# TongProject23

This is the repository for `tangcy03` & `alicebob142857` 's 2023 Tong Class summer project.

## File structures

Folder `vrepScenes` mainly contains scenes of the simulated robot arm environment, `push_to_see` contains the method of object segmentations by pushing.

`ObjectDetection` is the legacy folder of object segmentation and was not finished.

## Usage

At the root directory (i.e. `TongProject23`), open a scene in `vrepScenes` to start the simulation environment.

The segmentation task is performed in the file `./seg_demo.py` for demo.

This function will push the scene for the given `push_times` parameter and then return the segmented objects' corner coordinates and masks.