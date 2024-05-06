import cv2
import numpy as np


from .orthographic_projector import *

__doc__ = orthographic_projector.__doc__
if hasattr(orthographic_projector, "__all__"):
    __all__ = orthographic_projector.__all__


def preprocess_cloud(cloud, precision):
    geometry = np.asarray(cloud.points, dtype=np.float64)
    color = np.asarray(cloud.colors, dtype=np.float64)
    min_bound = cloud.get_min_bound()
    max_bound = cloud.get_max_bound()
    if np.any(min_bound < 0):
        geometry -= min_bound
        if np.any(max_bound < 1) or (geometry.max() <= 1):
            g = geometry
            geometry = (1 << precision) * (g - g.min()) / (g.max() - g.min())
    return geometry, color


def crop_img(image, ocp_map):
    if image.dtype != np.uint8 or ocp_map.dtype != np.uint8:
        image = image.astype(np.uint8)
        ocp_map = ocp_map.astype(np.uint8)
    x, y, w, h = cv2.boundingRect(ocp_map)
    cropped_image = image[y:y+h, x:x+w]
    cropped_ocp_map = ocp_map[y:y+h, x:x+w]
    return cropped_image, cropped_ocp_map


def pad_img(image, ocp_map):
    if image.dtype != np.uint8 or ocp_map.dtype != np.uint8:
        image = image.astype(np.uint8)
        ocp_map = ocp_map.astype(np.uint8)
    mask = (ocp_map != 1).astype(np.uint8)
    padded_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
    return padded_image