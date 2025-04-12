extern crate ndarray as nd;
extern crate num;
extern crate numpy;
extern crate pyo3;

use numpy::PyReadonlyArray2;
use numpy::{PyArray3, PyArray4};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod projector;
mod utils;

#[pyfunction]
fn generate_projections<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f64>,
    colors: PyReadonlyArray2<u8>,
    precision: u64,
    filtering: u64,
    verbose: bool,
) -> (Bound<'py, PyArray4<u64>>, Bound<'py, PyArray3<f64>>) {
    if verbose {
        println!("Generating projections");
    }
    let (points, colors) = (utils::to_ndarray(&points), utils::to_ndarray(&colors));
    let (images, ocp_maps, freqs) =
        projector::compute_projections(points, colors, precision, filtering);
    let images = PyArray4::from_owned_array(py, images);
    let ocp_maps = PyArray3::from_owned_array(py, ocp_maps);
    if verbose {
        for i in 0..6 {
            println!("{} points removed from projection {}", &freqs[i], &i);
        }
    }
    (images, ocp_maps)
}

#[pymodule]
fn orthographic_projector(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_projections, m)?)?;
    Ok(())
}
