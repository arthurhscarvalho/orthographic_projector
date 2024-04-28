import numpy as np
import open3d as o3d
import time
import projectors
from reference import orthographic_projection


pc = o3d.io.read_point_cloud('/home/arthurc/romanoillamp_vox10.ply')

points = np.asarray(pc.points)
colors = np.asarray(pc.colors)

if np.any(pc.get_min_bound() < 0):
    print(f'Point Cloud has negative points')
    points = points - pc.get_min_bound()
else:
    print(f'Point Cloud does not have negative points')

precision = 10
filtering = 2

print('Computing test...')
colors2 = colors * 255
s = time.time()
a, b = projectors.orthographic_projection(points, colors2, precision, filtering)
f = time.time()
a, b = np.asarray(a), np.asarray(b)
print(f'Done. Time taken: {(f-s):.2f} s')

print('Computing ref...')
s = time.time()
a_ref, b_ref = orthographic_projection(points, colors, precision, filtering)
f = time.time()
a_ref, b_ref = np.asarray(a_ref), np.asarray(b_ref)
print(f'Done. Time taken: {(f-s):.2f} s')

print('a_ref == a, b_ref == b:')
print(np.all(a_ref == a), np.all(b_ref == b))