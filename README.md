# orthographic_projector

## Purpose

A Python package with a Rust extension for generating six orthographic RGB projections and occupancy maps from a colored point cloud. It supports depth-based filtering and occupancy-based cropping. These projections form the input to the point cloud quality assessment method described in the associated publication.

The PyPI distribution is named `orthographic-projector`; the Python import is `orthographic_projector`.

## Related projects

[OpenCV](https://github.com/opencv/opencv-python) provides the cropping and image export operations. [Open3D](https://github.com/isl-org/Open3D) is an optional point cloud reader used by the bundled example.

The [Python interface](orthographic_projector/__init__.py) and [Rust projection implementation](src/projector.rs) are maintained in this repository and built together through Maturin.

## Requirements

- Python 3.10, 3.11, or 3.12 for the documented setup and wheel matrix; Python 3.11 or 3.12 is recommended.
- NumPy, OpenCV, and the other runtime dependencies declared in [pyproject.toml](pyproject.toml).
- For source builds: [uv](https://docs.astral.sh/uv/), a current stable Rust toolchain, Cargo, a native linker/C compiler, and Git.

Projection runs on the CPU. No GPU is required.

## Installation

### From PyPI

```bash
pip install orthographic-projector
```

Compatible published wheels include the Rust extension.

### From source

```bash
git clone https://github.com/arthurhscarvalho/orthographic_projector.git
cd orthographic_projector
uv sync --locked --python 3.12
```

uv creates `.venv/`, builds an editable installation through Maturin, and installs the development tools. [uv.lock](uv.lock) records the resolved dependencies. Use `uv run` without activating the environment.

After changing the Rust extension, rebuild it:

```bash
uv run --locked maturin develop --release
```

To build a release wheel:

```bash
uv run --locked maturin build --release --out dist
```

## Usage

```python
import numpy as np
import orthographic_projector as op

points = np.load("points.npy")  # XYZ coordinates, shape (N, 3).
colors = np.load("colors.npy")  # RGB values, shape (N, 3).

images, occupancy_maps = op.generate_projections(
    points,
    colors,
    precision=8,
    filtering=2,
    crop=False,
    verbose=False,
)
print(images.shape)          # (6, 256, 256, 3)
print(occupancy_maps.shape)  # (6, 256, 256)

cropped_images, cropped_maps = op.apply_cropping(images, occupancy_maps)
print(len(cropped_images))  # 6; each view can have a different height and width.
```

| Argument | Description |
| --- | --- |
| `points` | A nonempty, nondegenerate XYZ array of shape `(N, 3)`, converted to `float64`. Supply finite coordinates with spatial variation. |
| `colors` | An RGB array matching the coordinate shape. Supply values in `[0, 1]` or `[0, 255]`; the backend receives `uint8`. |
| `precision` | Required resolution exponent. Each uncropped image has height and width `2**precision`. Larger values increase memory and processing costs. |
| `filtering` | Nonnegative neighborhood radius for depth-based filtering; defaults to `2`. Set `0` to disable filtering. Keep the radius smaller than half the image dimension. |
| `crop` | Crop each view to its occupied bounding rectangle; defaults to `False`. |
| `verbose` | Print preprocessing and projection information; defaults to `True`. |

Preprocessing shifts negative coordinates and rescales the point cloud to the projection grid.

| Output | Shape and dtype |
| --- | --- |
| Uncropped RGB images | `(6, 2**precision, 2**precision, 3)`, `uint64`, with values in `[0, 255]`. |
| Uncropped occupancy maps | `(6, 2**precision, 2**precision)`, `float64`, with `1` for occupied pixels and `0` for background. |
| Cropped RGB images | A list of six `uint8` arrays of shape `(H_i, W_i, 3)`. |
| Cropped occupancy maps | A list of six `uint8` arrays of shape `(H_i, W_i)`, with values `0` or `1`. |

The six views correspond to the minimum- and maximum-depth projections along the X, Y, and Z axes, in that order. Unoccupied RGB pixels are white. Passing `crop=True` to `generate_projections` applies the same cropping operation as `apply_cropping`.

### Bundled PLY example

The [example script](examples/example_generate_projections.py) loads [redandblack_vox10_1550.ply](examples/redandblack_vox10_1550.ply), generates six projections at resolution `1024 × 1024`, crops them, and saves `projection_0.png` through `projection_5.png` in the repository root.

Install the optional example dependencies and run from the repository root:

```bash
uv sync --locked --python 3.12 --group examples
uv run --locked --group examples python examples/example_generate_projections.py
```

Change `PC_PATH`, `precision`, `filtering`, `crop`, and `save` in the script to use another cloud or configuration. OpenCV expects BGR images when saving; the script converts the RGB projections before writing them.

## Publications

The first paper describes the associated quality assessment method. The second describes the projection approach on which the implementation is based. The third is the source reference for the example point cloud.

1. Arthur H. S. Carvalho, Pedro G. Freitas, Mateus Gonçalves, Johann Homonnai, and Mylène C. Q. Farias. “Perception-Driven Point Cloud Quality Assessment Through Projections and Deep Structure Similarity.” *2024 IEEE 26th International Workshop on Multimedia Signal Processing (MMSP)*, pp. 1–6, 2024. [DOI: 10.1109/MMSP61759.2024.10743953](https://doi.org/10.1109/MMSP61759.2024.10743953).
2. Alireza Javaheri, Catarina Brites, Fernando Pereira, and João Ascenso. “Joint Geometry and Color Projection-Based Point Cloud Quality Metric.” *IEEE Access*, vol. 10, pp. 90481–90497, 2022. [DOI: 10.1109/ACCESS.2022.3198995](https://doi.org/10.1109/ACCESS.2022.3198995).
3. Evangelos Alexiou, Irene Viola, Tomás M. Borges, Tiago A. Fonseca, Ricardo L. de Queiroz, and Touradj Ebrahimi. “A comprehensive study of the rate-distortion performance in MPEG point cloud compression.” *APSIPA Transactions on Signal and Information Processing*, vol. 8, article e27, 2019. [DOI: 10.1017/ATSIP.2019.20](https://doi.org/10.1017/ATSIP.2019.20).

```bibtex
@inproceedings{carvalho2024perception,
  author    = {Carvalho, Arthur H. S. and Freitas, Pedro G. and Gonçalves, Mateus and Homonnai, Johann and Farias, Mylène C. Q.},
  title     = {Perception-Driven Point Cloud Quality Assessment Through Projections and Deep Structure Similarity},
  booktitle = {2024 IEEE 26th International Workshop on Multimedia Signal Processing (MMSP)},
  pages     = {1--6},
  year      = {2024},
  doi       = {10.1109/MMSP61759.2024.10743953}
}

@article{javaheri2022projection,
  author  = {Javaheri, Alireza and Brites, Catarina and Pereira, Fernando and Ascenso, João},
  title   = {Joint Geometry and Color Projection-Based Point Cloud Quality Metric},
  journal = {IEEE Access},
  volume  = {10},
  pages   = {90481--90497},
  year    = {2022},
  doi     = {10.1109/ACCESS.2022.3198995}
}

@article{alexiou2019compression,
  author  = {Alexiou, Evangelos and Viola, Irene and Borges, Tomás M. and Fonseca, Tiago A. and de Queiroz, Ricardo L. and Ebrahimi, Touradj},
  title   = {A comprehensive study of the rate-distortion performance in {MPEG} point cloud compression},
  journal = {APSIPA Transactions on Signal and Information Processing},
  volume  = {8},
  pages   = {e27},
  year    = {2019},
  doi     = {10.1017/ATSIP.2019.20}
}
```

## License

[MIT License](LICENSE).
