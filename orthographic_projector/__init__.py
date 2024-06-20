import cv2
import numpy as np
from .orthographic_projector import generate_projections as _internal_generate_projections


def __preprocess_point_cloud(points, colors, precision):
    if type(points) != np.ndarray:
        points = np.array(points)
    if type(colors) != np.ndarray:
        colors = np.array(colors)
    if points.shape != colors.shape:
        raise Exception('Points and colors must have the same shape.')
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    if np.any(min_bound < 0):
        points -= min_bound
    if np.any(max_bound < 1) or (points.max() <= 1):
        points = (1 << precision) * (points - points.min()) / (points.max() - points.min())
    if colors.max() <= 1 and colors.min() >= 0:
        colors = (colors * 255).astype(np.uint8)
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
        cropped_image = image[y:y+h, x:x+w]
        cropped_ocp_map = ocp_map[y:y+h, x:x+w]
        images_result.append(cropped_image)
        ocp_maps_result.append(cropped_ocp_map)
    return images_result, ocp_maps_result


def generate_projections(points, colors, precision, filtering, crop=False, verbose=True):
    points, colors = __preprocess_point_cloud(points, colors, precision)
    images, ocp_maps = _internal_generate_projections(points, colors, precision, filtering, verbose)
    images, ocp_maps = np.asarray(images), np.asarray(ocp_maps)
    if crop is True:
        images, ocp_maps = apply_cropping(images, ocp_maps)
    return images, ocp_maps