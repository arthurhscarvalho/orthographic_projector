import numpy as np
import open3d as o3d
import time
# import projectors
# import cv2
# from reference import orthographic_projection
import orthographic_projector


# def save_projections(projections, ocp_maps, crop, pad):
#     for i in range(len(projections)):
#         img = projections[i].astype(np.uint8)
#         ocp_map = ocp_maps[i].astype(np.uint8)
#         if crop is True:
#             cropped_img, cropped_ocp_map = crop_img(img, ocp_map)
#             img = cropped_img
#         if pad is True:
#             padded_img = pad_img(cropped_img, cropped_ocp_map)
#             img = padded_img
#         image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         cv2.imwrite(f'projection_{i}.png', image_bgr)


precision = 10
filtering = 2

pc = o3d.io.read_point_cloud('/home/arthurc/redandblack_vox10_1550.ply')

points, colors = orthographic_projector.preprocess_cloud(pc, precision)

print('Computing test...')
colors2 = colors * 255
s = time.time()
img, ocp_map = orthographic_projector.generate_projections(points, colors2, precision, filtering)
f = time.time()
img, ocp_map = np.asarray(img), np.asarray(ocp_map)
print(f'Done. Time taken: {(f-s):.2f} s')

print('Cropping...')
im, ocp = orthographic_projector.crop_img(img[0], ocp_map[0])
print('Padding...')
im = orthographic_projector.pad_img(im, ocp)

# save_projections(img, ocp_map, crop=True, pad=False)

# print('Computing ref...')
# s = time.time()
# img_ref, ocp_map_ref = orthographic_projection(points, colors, precision, filtering)
# f = time.time()
# img_ref, ocp_map_ref = np.asarray(img_ref), np.asarray(ocp_map_ref)
# print(f'Done. Time taken: {(f-s):.2f} s')

# print('img_ref == img, ocp_map_ref == ocp_map:')
# print(np.all(img_ref == img), np.all(ocp_map_ref == ocp_map))