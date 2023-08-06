import sys
sys.path.append('.')
from model import MaskRGNetwork
from dataset import PushAndGraspDataset
import yaml
import numpy as np
import os
from torch.utils import data as td
import argparse


def main():

    parser = argparse.ArgumentParser(description='Train or evaluate the Mask-RCNN model')
    parser.add_argument('--train', action='store_true', help='train the model')
    parser.add_argument('--evaluate', action='store_true', help='evaluate the model')
    args = parser.parse_args()
    
    with open('model_config.yaml') as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)

    # create the model
    model = MaskRGNetwork(configuration)

    # create dataset objects and set the model
    dataset = PushAndGraspDataset(configuration)
    test_indices = os.path.join(configuration['dataset']['path'], configuration['dataset']['test_indices'])
    test_subset = td.Subset(dataset, test_indices)
    train_indices = os.path.join(configuration['dataset']['path'], configuration['dataset']['train_indices'])
    train_subset = td.Subset(dataset, train_indices)

    if args.train:
        # Training:
        model.set_data(train_subset)
        model.train_model()
        print("Training finished!")

    if args.evaluate:
        # Evaluation:
        # this loads the saved weights from the file in the config file
        model.load_weights()
        # load a new dataset for the evaluation
        model.set_data(test_subset, is_test=True, batch_size=20)

        # evaluate
        res = model.evaluate_model()
        np.save('exps/results.npy', res)
        with open('exps/results.txt', 'w') as output:
            output.write(res)
        print("Evaluation finished!")


if __name__ == "__main__":
    main()