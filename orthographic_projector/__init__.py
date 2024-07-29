import cv2
import numpy as np

from .orthographic_projector import (
    generate_projections as _internal_generate_projections,
)


def __preprocess_point_cloud(points, colors, precision):
    if type(points) is not np.ndarray or points.dtype is not np.double:
        points = np.array(points, dtype=np.double)
    if type(colors) is not np.ndarray or colors.dtype is not np.double:
        colors = np.array(colors, dtype=np.double)
    if points.shape != colors.shape:
        raise Exception("Points and colors must have the same shape.")
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    if np.any(min_bound < 0):
        points -= min_bound
    if np.any(max_bound < 1) or (points.max() <= 1):
        points = (
            (1 << precision) * (points - points.min()) / (points.max() - points.min())
        )
    if colors.max() <= 1 and colors.min() >= 0:
        colors = colors * 255
    colors = colors.astype(np.uint8)
    return points, colors


def apply_cropping(images, ocp_maps):
    if images.dtype != np.uint8 or ocp_maps.dtype != np.uint8:
        images = images.astype(np.uint8)
        ocp_maps = ocp_maps.astype(np.uint8)
    images_result = []
    ocp_maps_result = []
    for i in range(len(images)):
        image, ocp_map = images[i], ocp_maps[i]
        x, y, w, h = cv2.boundingRect(ocp_map)
        cropped_image = image[y : y + h, x : x + w]
        cropped_ocp_map = ocp_map[y : y + h, x : x + w]
        images_result.append(cropped_image)
        ocp_maps_result.append(cropped_ocp_map)
    return images_result, ocp_maps_result


def generate_projections(
    points, colors, precision, filtering=2, crop=False, verbose=True
):
    """
    Generate projections from a given point cloud.

    Parameters
    ----------
    points : (M, 3) array_like
        Points from the point cloud.
    colors : (M, 3) array_like
        Colors from the point cloud.
    filtering : int, optional
        Filtering factor.
        Default is 2.
    crop : bool, optional
        If True, the generated projections will be cropped.
        Default is False.
    verbose: bool, optional
        Whether to display verbose information or not.
        Default is False.

    Returns
    -------
    projections : (P, N, M, C) np.ndarray
        A set of six RGB images corresponding to the projections generated
        from the point cloud.
    occupancy_maps : (P, N, M) np.ndarray
        A set of binary images corresponding to the occupancy maps
        from the generated projections.

    Notes
    -----
    For the points arguments, any kind of dtype is accepted, but
    the array will eventually be converted to np.double.

    For the colors arguments, it is expected that the colors are
    on the [0, 1] range, or [0, 255]. Other ranges are not supported.
    Any dtype is accepted, but the array will eventually be converted
    to np.uint8.

    It is recommended to simply read the point cloud using open3d,
    and pass the points and colors parameters as np.ndarrays.

    Point clouds without colors currently are not supported.
    """
    points, colors = __preprocess_point_cloud(points, colors, precision)
    images, ocp_maps = _internal_generate_projections(
        points, colors, precision, filtering, verbose
    )
    images, ocp_maps = np.asarray(images), np.asarray(ocp_maps)
    if crop is True:
        images, ocp_maps = apply_cropping(images, ocp_maps)
    return images, ocp_maps
