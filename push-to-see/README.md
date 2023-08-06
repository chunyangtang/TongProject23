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

### Installation
1. All instructions and code implementations are based on the folder `push-to-see` as the working directory except `pushtosee_segmetation.py` which should be run in the parent directory.
2. Create a folder named `pretrained` in the working directory and **download the model weights** listed above. `data`, `exps`, `logs` and `Database_vrep_inria` can also be created for their usages in the programs below.
3. Install the required packages listed in `requirements.txt` of the parent directory (`TongProject23`) using `pip install -r ../requirements.txt`.

### Generating dataset
1. Open the simulation scene at `./simulation/data_generation_scene.ttt` using CoppeliaSim. Notice the objects of the scene is used in `Task_no_1.ttt` rather than `Task_no_2_VPG_heap.ttt`
2. Run `python ./src/database_generator.py` to generate the dataset. (The dataset is set to be saved at `./Database_vrep_inria/` in configuration which is the same in the model-training config file.)

**Known issues**: Dataset genrating process fail every two steps for the reason that vrep failed to add new objects to simulation.

### Training Mask-RG
1. Get the dataset ready and set the path in `model_config.yaml` to load the dataset.
2. Run `python ./src/mask_rg/main.py --train` to train the model. (Set model saving plcae in config file.)
3. Run `python ./src/mask_rg/main.py --evaluate` to evaluate the model. (Change config file for the right model to be loaded, it's currently loading the pretrained model weights provided by the article.)

**Known issues**: Dataset does not have train/test indices data and so program won't work.

### Utilizing Mask-RCNN directly
I created a file located at `./maskrcnn_model_test.py` to make segmentations directly using the pretrained model.
1. Make sure the pretrained weight of mask-rg is ready and the program can find it.
2. Place images to be segmented at `./data/` folder.
2. Run `python ./maskrcnn_model_test.py --img {IMG_FILENAME}` to see the result, result is saved at `./exps/` folder.

### Training Push-DQN
1. properly setup pretrained models of Mask-RG and Push-DQN in their config files.
2. Open the simulation scene at `./simulation/Task_no_*.ttt` using CoppeliaSim.
3. Run `python ./src/push_dqn/main.py` to test the model. (Defaultly the `setup.istesting` configuration in config file is set True, making DQN work in EXPLOIT mode rather than EXPLORE.)

Notice that Mask-RG will automatically and implicitly load pretrained weights in `mask_rg/mask_rg.py` so training Mask-RG model in advance is mandatory.

The Push-DQN model can be trained from scratch when `model.file` is set to `new` in `push-DQN_config.yaml`, remeber to change `istesting` to false for training.

**Known issues**: A "list out of range" error often occurs at the end of `./src/mask_rg/rewards.py>RewardGenerator>print_seg_diff`, it is currently ignored by adding a `try...except` block.

### Utilizing the full pretrained Push-DQN
`pushtosee_segmetation.py` is implemented using the adapted version of `push_DQN/main.py` named `push_DQN/main_connector.py`. The models & configs should be ready and set to test mode.

**This file is designed to be working at `TongProject23` folder. When calling it, start your program in that directory or try use os.chdir() or sys.path.append(".") to use it.**

Open the simulation, call the function `get_segmentation` and get the segmentation of each object.

This file could potentially be optimized.