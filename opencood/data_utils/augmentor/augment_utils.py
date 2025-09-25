# -*- coding: utf-8 -*-
# Author: OpenPCDet

import numpy as np

from opencood.utils import common_utils

"""
gt_boxes: Shape (N, 7 + C). Each box is [x, y, z, dx, dy, dz, heading, [vx], [vy]]
x, y, z: Center of the box (in 3D space)
dx, dy, dz: Dimensions of the box (length, width, height)
heading: Orientation of the box (rotation around the vertical axis, often called yaw)
[vx], [vy]: (Optional) Velocity components in x and y direction

points: Shape (M, 3 + C). Each point is [x, y, z, ...]

"""


def random_flip_along_x(gt_boxes, points):
    """
    Purpose: Randomly flips the scene along the X-axis (left-right flip).
    (x,y,z)-> (x,-y,z)
    How it works:
        With 50% probability, it negates the Y-coordinates for both gt_boxes and points.
        It also negates the heading (rotation angle) of the boxes and the Y velocity if present.
    Use case: Simulates mirrored scenes, increasing data diversity.
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]

    return gt_boxes, points


def random_flip_along_y(gt_boxes, points):
    """
    Purpose: Randomly flips the scene along the Y-axis (front-back flip).
    How it works:
        With 50% probability, it negates the X-coordinates for both gt_boxes and points.
        It also negates the heading and adds π (to maintain orientation).
        Negates the X velocity if present.
    Use case: Simulates driving in the opposite direction, further diversifying the dataset.
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        points[:, 0] = -points[:, 0]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7] = -gt_boxes[:, 7]

    return gt_boxes, points


def global_rotation(gt_boxes, points, rot_range):
    """
    Purpose: Applies a random rotation to the entire scene around the Z-axis (vertical axis).
    How it works:
        Samples a noise_rotation angle from the given rot_range.
        Rotates all points and the positions of the bounding boxes using a utility function (rotate_points_along_z).
        Adjusts the heading of each box by adding the rotation.
        If velocity vectors are present, they are also rotated accordingly.
    Use case: Makes the model invariant to the orientation of objects and scenes.

    model invariant: model’s predictions do not change even if the input data is transformed in certain ways.
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0],
                                       rot_range[1])
    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :],
                                                np.array([noise_rotation]))[0]

    gt_boxes[:, 0:3] = \
        common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3],
                                           np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation

    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[
            np.newaxis, :, :],
            np.array([noise_rotation]))[0][:, 0:2]

    return gt_boxes, points


def global_scaling(gt_boxes, points, scale_range):
    """
    Purpose: Scales the entire scene by a random factor.
    How it works:
        Samples a scaling factor within scale_range.
        Multiplies the first 3 coordinates of points and the first 6 coordinates of boxes by this scale (scaling position and size).
    Use case: Simulates objects/scenes appearing at different sizes/distances, improving generalization.
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale

    return gt_boxes, points
