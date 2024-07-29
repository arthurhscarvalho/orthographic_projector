extern crate ndarray as nd;
extern crate num;
extern crate numpy;
extern crate pyo3;

use numpy::{PyArray3, PyArray4};
use numpy::{PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod projector;
mod utils;

#[pyfunction]
fn generate_projections<'py>(
    _py: Python<'py>,
    points: PyReadonlyArray2<f64>,
    colors: PyReadonlyArray2<u8>,
    precision: u64,
    filtering: u64,
    verbose: bool,
) -> (&'py PyArray4<u64>, &'py PyArray3<f64>) {
    if verbose {
        println!("Generating projections");
    }
    let (points, colors) = (utils::to_ndarray(&points), utils::to_ndarray(&colors));
    let (images, ocp_maps, freqs) =
        projector::compute_projections(points, colors, precision, filtering);
    let (images, ocp_maps) = (images.to_pyarray(_py), ocp_maps.to_pyarray(_py));
    if verbose {
        for i in 0..6 {
            println!("{} points removed from projection {}", &freqs[i], &i);
        }
    }
    (images, ocp_maps)
}

#[pymodule]
fn orthographic_projector(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_projections, m)?)?;
    Ok(())
}
