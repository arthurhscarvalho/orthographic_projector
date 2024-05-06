import numpy as np
import open3d as o3d
import time
import cv2
import orthographic_projector


def save_projections(projections):
    for i in range(len(projections)):
        image = projections[i].astype(np.uint8)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'projection_{i}.png', image_bgr)


PC_PATH = '/home/arthurc/longdress_vox10_1300.ply'
pc = o3d.io.read_point_cloud(PC_PATH)
points, colors = np.asarray(pc.points), np.asarray(pc.colors)

precision = 10
filtering = 2
crop = True

print('Computing test...')
t0 = time.time()
img, ocp_map = orthographic_projector.generate_projections(points, colors, precision, filtering, crop)
t1 = time.time()
print(f'Done. Time taken: {(t1-t0):.2f} s')

save_projections(img)