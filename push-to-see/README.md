# Push-to-See 
## Learning Non-Prehensile Manipulation to Enhance Instance Segmentation via Deep Q-Learning
This is the main repository of our Push-to-See model. For technical details, the paper is accessible via [this link.](https://ieeexplore.ieee.org/document/9811645)

### Model weights
Trained model weigths can be download through the following links:
- Mask-RG (trained on Inria objects for 40 epochs): [link](https://www.dropbox.com/s/mqf7iwmxeti76wx/maskrg_inria_v1_40ep.pth?dl=0)
- Push-DQN (trained for 41k episodes): [link](https://www.dropbox.com/s/96qqmt809gceguj/push_dqn_41k.pth?dl=0)

### Demos

A short description of the model and video demonstrations can be seen on [this video.](https://www.youtube.com/watch?v=CtMaCpACAjU)

###
(Instructions for installation, training and running will be included in this readme file. For now, please contact the corresponding author if you require any help via baris.serhan@manchester.ac.uk)  


## Instructions
All instructions and code implementations are based on the folder `push-to-see` as the working directory.
### Generating dataset
1. Open the simulation scene at `./simulation/data_generation_scene.ttt` using CoppeliaSim. 
2. Run `python ./src/database_generator.py` to generate the dataset. (The dataset is set to be saved at `./Database_vrep_inria/` in configuration which is the same in the model-training config file.)
### Training Mask-RG
