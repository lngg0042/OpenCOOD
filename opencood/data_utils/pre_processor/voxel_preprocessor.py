# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
Convert lidar to voxel. This class was manually designed, and we recommend
to use sp_voxel_preprocessor.

This file defines a VoxelPreprocessor class in Python, used for converting raw LiDAR point cloud data
into a voxel representation. This is commonly used in 3D perception tasks in autonomous driving or robotics.
The implementation inherits from a base class BasePreprocessor, and organizes point cloud data into voxels, 
which are small cubes in 3D space.

PyTorch Collate Function: A function that combines multiple data samples into a batch for deep learning model training.

Notable Details
- The output voxel grid is sparse (only non-empty voxels are stored).
- Each voxel can store up to T points; excess points are truncated.
- Voxel features are augmented with the offset from the mean of the points in the voxel.
- The code is designed to integrate with PyTorch’s DataLoader via custom collate functions.
"""
import sys

import numpy as np
import torch

from opencood.data_utils.pre_processor.base_preprocessor import \
    BasePreprocessor


class VoxelPreprocessor(BasePreprocessor):
    def __init__(self, preprocess_params, train):
        super(VoxelPreprocessor, self).__init__(preprocess_params, train)
        # TODO: add intermediate lidar range later
        # defines the spatial limits for voxelization
        self.lidar_range = self.params['cav_lidar_range']

        # voxel width, height, depth
        self.vw = self.params['args']['vw']
        self.vh = self.params['args']['vh']
        self.vd = self.params['args']['vd']
        # maximum number of points per voxel (for uniformity)
        self.T = self.params['args']['T']

    def preprocess(self, pcd_np):
        """
        Preprocess the lidar points by  voxelization.

        Parameters
        ----------
        pcd_np : np.ndarray 
            The raw lidar.
            a numpy array of raw LiDAR points, shape typically (N, 3) or (N, 4).
            
        Returns
        -------
        data_dict : the structured output dictionary.
        """
        data_dict = {}

        # calculate the voxel coordinates
        voxel_coords = ((pcd_np[:, :3] -
                         np.floor(np.array([self.lidar_range[0],
                                            self.lidar_range[1],
                                            self.lidar_range[2]])) / (
                             self.vw, self.vh, self.vd))).astype(np.int32)

        # convert to  (D, H, W) as the paper 
        # puts coordinates in (D,H,W) order
        voxel_coords = voxel_coords[:, [2, 1, 0]]
        voxel_coords, inv_ind, voxel_counts = np.unique(voxel_coords, axis=0,
                                                        return_inverse=True,
                                                        return_counts=True)

        voxel_features = []

        for i in range(len(voxel_coords)):
            voxel = np.zeros((self.T, 7), dtype=np.float32)
            pts = pcd_np[inv_ind == i]
            if voxel_counts[i] > self.T:
                pts = pts[:self.T, :]
                voxel_counts[i] = self.T

            # augment the points
            voxel[:pts.shape[0], :] = np.concatenate((pts, pts[:, :3] -
                                                      np.mean(pts[:, :3], 0)),
                                                     axis=1)
            voxel_features.append(voxel)

        # Returns a dictionary with voxel_features (shape [n_voxels, T, 7]) and their voxel_coords.
        data_dict['voxel_features'] = np.array(voxel_features)
        data_dict['voxel_coords'] = voxel_coords

        return data_dict

    def collate_batch(self, batch):
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : list or dict
            List or dictionary.

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.

        If batch is a list, calls collate_batch_list() (each item is a dictionary for one frame).
        If batch is a dict, calls collate_batch_dict() (batched by key).
        """

        if isinstance(batch, list):
            return self.collate_batch_list(batch)
        elif isinstance(batch, dict):
            return self.collate_batch_dict(batch)
        else:
            sys.exit('Batch has too be a list or a dictionarn')

    @staticmethod
    def collate_batch_list(batch):
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : list
            List of dictionary. Each dictionary represent a single frame.

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """
        voxel_features = []
        voxel_coords = []
        
        # Combines a list of dictionaries (one per sample) into a single batch.
        # Pads the voxel coordinates with a batch index.
        for i in range(len(batch)):
            voxel_features.append(batch[i]['voxel_features'])
            coords = batch[i]['voxel_coords']
            voxel_coords.append(
                np.pad(coords, ((0, 0), (1, 0)),
                       mode='constant', constant_values=i))

        # Returns a dict with batched voxel_features and voxel_coords as PyTorch tensors.
        voxel_features = torch.from_numpy(np.concatenate(voxel_features))
        voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))

        return {'voxel_features': voxel_features,
                'voxel_coords': voxel_coords}

    @staticmethod
    def collate_batch_dict(batch: dict):
        """
        Collate batch if the batch is a dictionary,
        eg: {'voxel_features': [feature1, feature2...., feature n]}
        
        Similar to the above but for when input is a dictionary of lists (batched by key)
        Combines and pads voxel coordinates with batch indices, returns tensors.
        
        Parameters
        ----------
        batch : dict

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """
        voxel_features = \
            torch.from_numpy(np.concatenate(batch['voxel_features']))
        coords = batch['voxel_coords']
        voxel_coords = []

        for i in range(len(coords)):
            voxel_coords.append(
                np.pad(coords[i], ((0, 0), (1, 0)),
                       mode='constant', constant_values=i))
        voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))

        return {'voxel_features': voxel_features,
                'voxel_coords': voxel_coords}
