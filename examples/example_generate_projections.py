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


# Load the point cloud through open3d
PC_PATH = './examples/redandblack_vox10_1550.ply'
pc = o3d.io.read_point_cloud(PC_PATH)
points, colors = np.asarray(pc.points), np.asarray(pc.colors)

# orthographic_projector parameters
precision = 10
filtering = 2
crop = True
save = True

t0 = time.time()
# The generate_projections function can be used for generating the projections
images, ocp_maps = orthographic_projector.generate_projections(points, colors, precision, filtering)
# The crop parameter could optionally be passed to the generate_projections function,
# but it can also be called after the generation process
if crop:
    images, ocp_maps = orthographic_projector.apply_cropping(images, ocp_maps)
t1 = time.time()
print(f'Done. Time taken: {(t1-t0):.2f} s')

# The save_projections function is just an example intended for visualization of the results
if save:
    save_projections(images)