# -*- coding: utf-8 -*-
"""
Class for data augmentation

DataAugmentor for performing data augmentation on 3D point cloud data, typically used for 
tasks like autonomous driving perception.

Purpose:
Provides a configurable, modular way to apply a sequence of data augmentation transformations 
to input data, especially point clouds and ground-truth bounding boxes.

Typical Usage:
Used in training pipelines to increase data diversity and improve model robustness by randomly 
transforming the data.
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from functools import partial

import numpy as np

from opencood.data_utils.augmentor import augment_utils


class DataAugmentor(object):
    """
    Data Augmentor.

    Parameters
    ----------
    augment_config : list
        A list of augmentation configuration.

    Attributes
    ----------
    data_augmentor_queue : list
        The list of data augmented functions.
    """

    def __init__(self, augment_config, train=True):
        self.data_augmentor_queue = []
        self.train = train # enables augmentation if true, false skip

        # For each config, instantiates the corresponding augmentation method (by name), 
        # partially applied with its config, and queues it in self.data_augmentor_queue.
        for cur_cfg in augment_config:
            cur_augmentor = getattr(self, cur_cfg['NAME'])(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)


    """
    Augmentation Methods:
    All three methods follow the same pattern:
    If data_dict is None, returns a partially applied function with the config set.
    Otherwise, performs augmentation on the input data.
    
    Input Data Structure:
    All methods expect a data_dict with these keys:
        'object_bbx_center': Ground-truth box centers (Nx7 or similar)
        'object_bbx_mask': Mask to indicate valid boxes
        'lidar_np': Point cloud data
    """
    
    def random_world_flip(self, data_dict=None, config=None):
        """
        Randomly flips the point cloud and labels along given axes ('x' or 'y').
        """
        if data_dict is None:
            return partial(self.random_world_flip, config=config)

        gt_boxes, gt_mask, points = data_dict['object_bbx_center'], \
                                    data_dict['object_bbx_mask'], \
                                    data_dict['lidar_np']
        gt_boxes_valid = gt_boxes[gt_mask == 1]

        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            gt_boxes_valid, points = getattr(augment_utils,
                                             'random_flip_along_%s' % cur_axis)(
                gt_boxes_valid, points,
            )

        gt_boxes[:gt_boxes_valid.shape[0], :] = gt_boxes_valid

        data_dict['object_bbx_center'] = gt_boxes
        data_dict['object_bbx_mask'] = gt_mask
        data_dict['lidar_np'] = points

        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        """
        Randomly rotates the entire scene (points and labels) within a specified angle range.
        """
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)

        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]

        gt_boxes, gt_mask, points = data_dict['object_bbx_center'], \
                                    data_dict['object_bbx_mask'], \
                                    data_dict['lidar_np']
        gt_boxes_valid = gt_boxes[gt_mask == 1]
        gt_boxes_valid, points = augment_utils.global_rotation(
            gt_boxes_valid, points, rot_range=rot_range
        )
        gt_boxes[:gt_boxes_valid.shape[0], :] = gt_boxes_valid

        data_dict['object_bbx_center'] = gt_boxes
        data_dict['object_bbx_mask'] = gt_mask
        data_dict['lidar_np'] = points

        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        """
        Randomly scales the scene (both points and labels) within a specified scale range.
        """
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)

        gt_boxes, gt_mask, points = data_dict['object_bbx_center'], \
                                    data_dict['object_bbx_mask'], \
                                    data_dict['lidar_np']
        gt_boxes_valid = gt_boxes[gt_mask == 1]

        gt_boxes_valid, points = augment_utils.global_scaling(
            gt_boxes_valid, points, config['WORLD_SCALE_RANGE']
        )
        gt_boxes[:gt_boxes_valid.shape[0], :] = gt_boxes_valid

        data_dict['object_bbx_center'] = gt_boxes
        data_dict['object_bbx_mask'] = gt_mask
        data_dict['lidar_np'] = points

        return data_dict

    def forward(self, data_dict):
        """
        Sequentially applies all augmentations in the queue to data_dict (if in training mode).
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        if self.train:
            for cur_augmentor in self.data_augmentor_queue:
                data_dict = cur_augmentor(data_dict=data_dict)

        return data_dict
