import numpy as np
import open3d as o3d
import time
import projectors
import cv2
from reference import orthographic_projection


def preprocess_cloud(cloud):
    geometry = np.asarray(cloud.points, dtype=np.float64)
    color = np.asarray(cloud.colors, dtype=np.float64)
    min_bound = cloud.get_min_bound()
    max_bound = cloud.get_max_bound()
    if np.any(min_bound < 0):
        geometry -= min_bound
        if np.any(max_bound < 1) or (geometry.max() <= 1):
            g = geometry
            geometry = 1000.0 * (g - g.min()) / (g.max() - g.min())
    return geometry, color


def crop_img(image, ocp_map):
    x, y, w, h = cv2.boundingRect(ocp_map)
    cropped_image = image[y:y+h, x:x+w]
    cropped_ocp_map = ocp_map[y:y+h, x:x+w]
    return (cropped_image, cropped_ocp_map)


def pad_img(image, ocp_map):
    mask = (ocp_map != 1).astype(np.uint8)
    padded_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
    return padded_image


def save_projections(projections, ocp_maps, crop, pad):
    for i in range(len(projections)):
        img = projections[i].astype(np.uint8)
        ocp_map = ocp_maps[i].astype(np.uint8)
        if crop is True:
            cropped_img, cropped_ocp_map = crop_img(img, ocp_map)
            img = cropped_img
        if pad is True:
            padded_img = pad_img(cropped_img, cropped_ocp_map)
            img = padded_img
        image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'projection_{i}.png', image_bgr)


pc = o3d.io.read_point_cloud('/home/arthurc/redandblack_vox10_1550.ply')

points, colors = preprocess_cloud(pc)

precision = 10
filtering = 2

print('Computing test...')
colors2 = colors * 255
s = time.time()
img, ocp_map = projectors.orthographic_projection(points, colors2, precision, filtering)
f = time.time()
img, ocp_map = np.asarray(img), np.asarray(ocp_map)
print(f'Done. Time taken: {(f-s):.2f} s')

save_projections(img, ocp_map, crop=True, pad=False)

print('Computing ref...')
s = time.time()
img_ref, ocp_map_ref = orthographic_projection(points, colors, precision, filtering)
f = time.time()
img_ref, ocp_map_ref = np.asarray(img_ref), np.asarray(ocp_map_ref)
print(f'Done. Time taken: {(f-s):.2f} s')

print('img_ref == img, ocp_map_ref == ocp_map:')
print(np.all(img_ref == img), np.all(ocp_map_ref == ocp_map))