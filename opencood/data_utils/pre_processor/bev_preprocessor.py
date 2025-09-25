# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
Convert lidar to bev

bev preprocessor

single BEV (C,H,W)
batched BEV (N,C,H,W)
"""

import numpy as np
import torch
from opencood.data_utils.pre_processor.base_preprocessor import \
    BasePreprocessor


class BevPreprocessor(BasePreprocessor):
    def __init__(self, preprocess_params, train):
        super(BevPreprocessor, self).__init__(preprocess_params, train)
        # spatial coverage of the LiDAR sensor
        self.lidar_range = self.params['cav_lidar_range']
        # Contains settings like BEV grid resolution and shape
        self.geometry_param = preprocess_params["geometry_param"]

    def preprocess(self, pcd_raw):
        """
        Preprocess the lidar points to BEV representations.

        Parameters
        ----------
        pcd_raw : np.ndarray
            The raw lidar.
             an Nx4 numpy array (X, Y, Z, intensity).

        Returns
        -------
        data_dict : the structured output dictionary.
        A dictionary with "bev_input" as a tensor shaped for neural network input.
        """
        # initializes BEV grid and an intensity counter
        bev = np.zeros(self.geometry_param['input_shape'], dtype=np.float32) # shaped defined here, which is tuple (height, width, channels)
        intensity_map_count = np.zeros((bev.shape[0], bev.shape[1]),
                                       dtype=np.int)
        bev_origin = np.array(
            [self.geometry_param["L1"], self.geometry_param["W1"],
             self.geometry_param["H1"]]).reshape(1, -1)

        # Shifts and scales each point from real-world coordinates to grid indices.
        indices = ((pcd_raw[:, :3] - bev_origin) / self.geometry_param[
            "res"]).astype(int)
        # For each point:
            # Sets the occupancy channel to 1 (indicates presence).
            # Accumulates intensity in the last channel.
            # Increments the point counter for that cell.
        for i in range(indices.shape[0]):
            bev[indices[i, 0], indices[i, 1], indices[i, 2]] = 1
            bev[indices[i, 0], indices[i, 1], -1] += pcd_raw[i, 3]
            intensity_map_count[indices[i, 0], indices[i, 1]] += 1
        # Divides the accumulated intensity by the count to get the average per cell.
        divide_mask = intensity_map_count != 0
        bev[divide_mask, -1] = np.divide(bev[divide_mask, -1],
                                         intensity_map_count[divide_mask])
        
        # Transposes the result to channel-first format.
        data_dict = {
            "bev_input": np.transpose(bev, (2, 0, 1)) # BEV tensor shape: channels height width
        }
        return data_dict

    @staticmethod
    def collate_batch_list(batch):
        """
        Customized pytorch data loader collate function.
        Concatenates the "bev_input" arrays along the batch dimension and converts them to a PyTorch tensor.
        
        Parameters
        ----------
        batch : list
            List of dictionary. Each dictionary represent a single frame.

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """
        bev_input_list = [
            x["bev_input"][np.newaxis, ...] for x in batch
        ]
        processed_batch = {
            "bev_input": torch.from_numpy(
                np.concatenate(bev_input_list, axis=0))
        }
        return processed_batch

    @staticmethod
    def collate_batch_dict(batch):
        """
        Customized pytorch data loader collate function.
        Concatenates and converts as above (collate_batch_list)

        Parameters
        ----------
        batch : dict
            Dict of list. Each element represents a CAV.
            A dict where "bev_input" is a list of BEV arrays (e.g., for multiple vehicles/CAVs in one scene).

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """
        bev_input_list = [
            x[np.newaxis, ...] for x in batch["bev_input"]
        ]
        processed_batch = {
            "bev_input": torch.from_numpy(
                np.concatenate(bev_input_list, axis=0))
        }
        return processed_batch

    def collate_batch(self, batch):
        """
        Customized pytorch data loader collate function.
        Selects batch collation method based on input type

        Parameters
        ----------
        batch : list / dict
            Batched data.
        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """
        if isinstance(batch, list):
            return self.collate_batch_list(batch)
        elif isinstance(batch, dict):
            return self.collate_batch_dict(batch)
        else:
            raise NotImplemented
