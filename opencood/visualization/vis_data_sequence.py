# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

""""
This script is a command-line tool for visualizing sequential LiDAR (and possibly other sensor) data using a dataset and parameters specified in a YAML config file. It:

Loads configuration,
Prepares the dataset and data loader,
Parses visualization options from the command line,
Calls a visualization utility to render the data (with chosen coloring).
"""
import os # For file path manipulations.
import argparse # For parsing command-line arguments.
from torch.utils.data import DataLoader # from PyTorch: For batching and loading data efficiently.

from opencood.hypes_yaml.yaml_utils import load_yaml # For loading YAML configuration files.
from opencood.visualization import vis_utils # Presumably includes visualization functions.
from opencood.data_utils.datasets.early_fusion_vis_dataset import \
    EarlyFusionVisDataset


def vis_parser():
    """
    Defines a function to parse command-line arguments.
    Adds a --color_mode argument (default "intensity") to choose how LiDAR points are colored (e.g. by intensity, z-value, or constant color).
    Returns the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="data visualization")
    parser.add_argument('--color_mode', type=str, default="intensity",
                        help='lidar color rendering mode, e.g. intensity,'
                             'z-value or constant.')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    # Checks if the script is run directly, not imported.
    # Gets the directory where this script is located (current_path).
    current_path = os.path.dirname(os.path.realpath(__file__))
    params = load_yaml(os.path.join(current_path,
                                    '../hypes_yaml/visualization.yaml')) # Loads a YAML configuration file named visualization.yaml from ../hypes_yaml/.

    # Creates an instance of EarlyFusionVisDataset with the loaded parameters:
    # visualize=True: Indicates this is for visualization, possibly changing dataset behavior.
    # train=False: Indicates this is not training data (likely validation/test/visualization).
    opencda_dataset = EarlyFusionVisDataset(params, visualize=True,
                                            train=False)

    # batch_size=1: Loads one data sample at a time.
    # num_workers=8: Loads data using 8 parallel subprocesses.
    # collate_fn: Uses the dataset's batch collation function for combining samples.
    # shuffle=False: No shuffling, so data is processed in order.
    # pin_memory=False: (Relevant for GPU, not pinning memory.)
    data_loader = DataLoader(opencda_dataset, batch_size=1, num_workers=8,
                             collate_fn=opencda_dataset.collate_batch_train,
                             shuffle=False,
                             pin_memory=False)
    
    # Parses command-line arguments to get the desired color_mode.
    # Calls the visualization function:
    # data_loader: The data iterator.
    # params['postprocess']['order']: The post-processing order from the YAML config.
    # color_mode=opt.color_mode: The chosen color mode for rendering LiDAR points.
    opt = vis_parser()
    vis_utils.visualize_sequence_dataloader(data_loader,
                                            params['postprocess']['order'],
                                            color_mode=opt.color_mode)

# python vis_data_sequence.py --color_mode=z-value
# This would visualize the dataset, coloring LiDAR points by their z-value.

