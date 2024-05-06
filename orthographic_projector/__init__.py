import cv2
import numpy as np
from .orthographic_projector import generate_projections


def __preprocess_cloud(points, colors, precision):
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    if np.any(min_bound < 0):
        points -= min_bound
    if np.any(max_bound < 1) or (points.max() <= 1):
        points = (1 << precision) * (points - points.min()) / (points.max() - points.min())
    if colors.max() <= 1 and colors.min() >= 0:
        colors = (colors * 255).astype(np.uint8)
    return points, colors


def __crop_img(image, ocp_map):
    if image.dtype != np.uint8 or ocp_map.dtype != np.uint8:
        image = image.astype(np.uint8)
        ocp_map = ocp_map.astype(np.uint8)
    x, y, w, h = cv2.boundingRect(ocp_map)
    cropped_image = image[y:y+h, x:x+w]
    cropped_ocp_map = ocp_map[y:y+h, x:x+w]
    return cropped_image, cropped_ocp_map


def generate_projections(points, colors, precision, filtering, crop):
    if type(points) != np.ndarray:
        points = np.array(points)
    if type(colors) != np.ndarray:
        colors = np.array(colors)
    points, colors = __preprocess_cloud(points, colors, precision)
    img, ocp_map = orthographic_projector.generate_projections(points, colors, precision, filtering)
    img, ocp_map = np.asarray(img), np.asarray(ocp_map)
    if crop is True:
        img_tmp = []
        ocp_map_tmp = []
        for i in range(6):
            im, ocp = __crop_img(img[i], ocp_map[i])
            img_tmp.append(im)
            ocp_map_tmp.append(ocp)
        img, ocp_map = img_tmp, ocp_map_tmp
    return img, ocp_map