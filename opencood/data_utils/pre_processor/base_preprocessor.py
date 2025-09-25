# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import numpy as np

from opencood.utils import pcd_utils


class BasePreprocessor(object):
    """
    Basic Lidar pre-processor.

    This file defines the BasePreprocessor class, which provides basic preprocessing utilities for Lidar point 
    cloud data in the OpenCOOD framework. The class includes methods for downsampling Lidar points and projecting 
    them onto a Bird’s Eye View (BEV) occupancy map.
    
    Parameters
    ----------
    preprocess_params : dict
        The dictionary containing all parameters of the preprocessing.

    train : bool
        Train or test mode.
    """

    def __init__(self, preprocess_params, train):
        self.params = preprocess_params
        self.train = train

    def preprocess(self, pcd_np):
        """
        Preprocess the lidar points by simple sampling.

        Parameters
        ----------
        pcd_np : np.ndarray
            The raw lidar.
            The raw Lidar point cloud as a NumPy array.

        Returns
        -------
        data_dict : the output dictionary.

        The class provides basic preprocessing for Lidar point clouds:
           -  Downsampling to a fixed number of points.
           -  Projecting 3D points into a 2D BEV occupancy map.

        It is designed to be extended or used as a component for more complex preprocessing pipelines 
        in autonomous driving or sensor fusion tasks.
        """
        data_dict = {}
        # It retrieves the number of points to sample from self.params.
        sample_num = self.params['args']['sample_num']

        # Calls downsample_lidar from pcd_utils to downsample the input point cloud to the specified number.
        pcd_np = pcd_utils.downsample_lidar(pcd_np, sample_num)
        # Returns a dictionary with the downsampled points under the key 'downsample_lidar'.
        data_dict['downsample_lidar'] = pcd_np

        return data_dict

    def project_points_to_bev_map(self, points, ratio=0.1):
        """
        Project points to BEV occupancy map with default ratio=0.1.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) / (N, 4)

        ratio : float
            Discretization parameters. Default is 0.1.

        Returns
        -------
        bev_map : np.ndarray
            BEV occupancy map including projected points with shape
            (img_row, img_col).

        points: The input point cloud (NumPy array, shape (N, 3) or (N, 4)).
        ratio: Discretization parameter for the BEV map resolution (default 0.1).
        cav_lidar_range: 6D vector (L1, W1, H1, L2, W2, H2) defining the region of interest for Lidar data.
        Calculates the BEV map size based on the range and ratio.
        Initializes a blank BEV map.
        Transforms 3D point coordinates into BEV map indices using the origin and ratio.
        Applies a mask so only points within the valid BEV area are considered.
        Sets the corresponding cells in the BEV map to 1 where points exist.
        Returns the BEV occupancy map (binary image of Lidar point locations).

        """
        L1, W1, H1, L2, W2, H2 = self.params["cav_lidar_range"]
        img_row = int((L2 - L1) / ratio)
        img_col = int((W2 - W1) / ratio)
        bev_map = np.zeros((img_row, img_col))
        bev_origin = np.array([L1, W1, H1]).reshape(1, -1)
        # (N, 3)
        indices = ((points[:, :3] - bev_origin) / ratio).astype(int)
        mask = np.logical_and(indices[:, 0] > 0, indices[:, 0] < img_row)
        mask = np.logical_and(mask, np.logical_and(indices[:, 1] > 0,
                                                   indices[:, 1] < img_col))
        indices = indices[mask, :]
        bev_map[indices[:, 0], indices[:, 1]] = 1
        return bev_map
