import cv2
import numpy as np

from .orthographic_projector import (
    generate_projections as _internal_generate_projections,
)


def __find_scaling_factor(points):
    columns = np.sort(points, axis=0)
    min_distance = float("inf")
    for i in range(1, points.shape[0]):
        prev_elements = [columns[i - 1][0], columns[i - 1][1], columns[i - 1][2]]
        curr_elements = [columns[i][0], columns[i][1], columns[i][2]]
        differences = [
            abs(prev_elements[0] - curr_elements[0]),
            abs(prev_elements[1] - curr_elements[1]),
            abs(prev_elements[2] - curr_elements[2]),
        ]
        differences = [i for i in differences if i != 0]
        if differences:
            min_distance = min(min_distance, min(differences))
    return np.rint(1 / (min_distance + np.finfo(np.double).eps))


def __preprocess_point_cloud(points, colors, precision, verbose):
    if type(points) is not np.ndarray or points.dtype is not np.double:
        points = np.array(points, dtype=np.double)
    if type(colors) is not np.ndarray or colors.dtype is not np.double:
        colors = np.array(colors, dtype=np.double)
    if points.shape != colors.shape:
        raise Exception("Points and colors must have the same shape.")
    min_coord = points.min()
    if min_coord < 0:
        points -= min_coord
        if verbose:
            print("Found negative points on PC. Displacement applied.")
    max_coord = points.max()
    if max_coord <= 1:
        scaling_factor = __find_scaling_factor(points)
        points *= scaling_factor
        max_coord = points.max()
        if verbose:
            print(f"PC denormalized using a scaling factor of {scaling_factor}.")
    scale = 2**precision
    if scale < max_coord:
        points /= max_coord
        points *= scale
        if verbose:
            print(f"PC reescaled to fit projection size of {scale}x{scale}.")
    if colors.max() <= 1 and colors.min() >= 0:
        colors = colors * 255
        if verbose:
            print("PC colors normalized to the [0, 255] interval.")
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


def apply_padding(images, ocp_maps, precision):
    images_result = []
    for i in range(len(images)):
        image, ocp_map = images[i], ocp_maps[i]
        image = image.astype(np.uint8)
        ocp_map = ocp_map.astype(np.uint8)
        # Create a border to prevent the closing operation touching the borders of the image
        border_size = 3 * precision
        border_sizes = (border_size, border_size, border_size, border_size)
        color = (255, 255, 255)
        border_type = cv2.BORDER_CONSTANT
        image = cv2.copyMakeBorder(image, *border_sizes, border_type, value=color)
        ocp_map = cv2.copyMakeBorder(ocp_map, *border_sizes, border_type, value=0)
        # Apply a closing operation and setup the inpainting mask
        kernel = np.ones((precision, precision), np.uint8)
        closed_ocp_map = cv2.morphologyEx(ocp_map, cv2.MORPH_CLOSE, kernel)
        not_mask = ((~(closed_ocp_map * 255)) / 255).astype(np.uint8)
        selected_object = cv2.bitwise_or(not_mask, ocp_map)
        mask = (selected_object != 1).astype(np.uint8)
        # [TEMPORARY] Save the inpainting masks
        image_bgr = cv2.cvtColor(selected_object * 255, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"map_{i}.png", image_bgr)
        # Apply the inpainting
        padded_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
        # Crop the created borders of the image
        top, bottom, left, right = border_sizes
        image_cropped = padded_image[top:-bottom, left:-right]
        # Store the final result
        images_result.append(image_cropped)
    return images_result


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
    points, colors = __preprocess_point_cloud(points, colors, precision, verbose)
    images, ocp_maps = _internal_generate_projections(
        points, colors, precision, filtering, verbose
    )
    images, ocp_maps = np.asarray(images), np.asarray(ocp_maps)
    if crop is True:
        images, ocp_maps = apply_cropping(images, ocp_maps)
    return images, ocp_maps
