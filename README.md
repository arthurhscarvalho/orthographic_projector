# Orthographic Projector
#### A Rust-based Python library for generating cubic orthographic projections from point clouds.

## Prerequisites
- Python 3.10 or later
- rustc 1.77.2 or later
- pip

## Setup
### From PyPI
```bash
pip install orthographic-projector
# or
python -m pip install orthographic-projector
```

### From source
```bash
# Clone this repository
https://github.com/akaTsunemori/orthographic_projector.git

# cd into the project folder
cd orthographic_projector

# Setup and activate the conda environment
conda env create -f environment.yml
conda activate orthographic-projector

# Compile the project into a python module using maturin
maturin develop -r
```

## Usage
```python
import cv2
import numpy as np
import open3d as o3d
import orthographic_projector
import time


def save_projections(projections):
    for i in range(len(projections)):
        image = projections[i].astype(np.uint8)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"projection_{i}.png", image_bgr)


# Load the point cloud through open3d
PC_PATH = "./examples/redandblack_vox10_1550.ply"
pc = o3d.io.read_point_cloud(PC_PATH)
points, colors = np.asarray(pc.points), np.asarray(pc.colors)

# orthographic_projector parameters
precision = 10
filtering = 2
crop = True
save = True

t0 = time.time()
# The generate_projections function can be used for generating the projections
images, ocp_maps = orthographic_projector.generate_projections(
    points, colors, precision, filtering
)
# The crop parameter could optionally be passed to the generate_projections function,
# but it can also be called after the generation process
if crop:
    images, ocp_maps = orthographic_projector.apply_cropping(images, ocp_maps)
t1 = time.time()
print(f"Done. Time taken: {(t1-t0):.2f} s")

# The save_projections function is just an example intended for visualization of the results
if save:
    save_projections(images)
```

This example is also available in the *[examples/example_generate_projections.py](https://github.com/akaTsunemori/orthographic_projector/blob/main/examples/example_generate_projections.py)* folder.

## Results
These are the generated projections to be expected from the provided example.

<details>
    <summary>Spoiler</summary>
    <img src="https://i.imgur.com/cKTmA5s.png" alt="Projection 1">
    <img src="https://i.imgur.com/KbkAOOw.png" alt="Projection 2">
    <img src="https://i.imgur.com/79DYoLQ.png" alt="Projection 3">
    <img src="https://i.imgur.com/mNveRev.png" alt="Projection 4">
    <img src="https://i.imgur.com/lszqcn2.png" alt="Projection 5">
    <img src="https://i.imgur.com/LaEhUNb.png" alt="Projection 6">
</details>

## References
- Orthographic projector

    [1] A. Javaheri, C. Brites, F. Pereira, and J. Ascenso, “Joint geometry and color projection-based point cloud quality metric,” IEEE Access, vol. 10, pp. 90 481–90 497, 2022.

- Source for the example point cloud

    [3] E. Alexiou, I. Viola, T. M. Borges, T. A. Fonseca, R. L. De Queiroz, and T. Ebrahimi, “A comprehensive study of the rate-distortion performance in mpeg point cloud compression,” APSIPA Transactions on Signal and Information Processing, vol. 8, 2019

## Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated, you can simply open an issue.

## License
MIT License

---

> GitHub [@akaTsunemori](https://github.com/akaTsunemori)
