use ndarray::prelude::*;
use numpy::ToPyArray;
use numpy::{PyArray3, PyArray4};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

fn vec_to_2d_with_floor(vec: &Vec<Vec<f64>>) -> Array2<u64> {
    let nrows = vec.len();
    let ncols = vec[0].len();
    let flattened: Vec<u64> = vec
        .iter()
        .flat_map(|row| row.iter().map(|&val| val.floor() as u64))
        .collect();
    let array = Array2::from_shape_vec((nrows, ncols), flattened).unwrap();
    return array;
}

#[pyfunction]
fn generate_projections(
    _py: Python,
    points: Vec<Vec<f64>>,
    colors: Vec<Vec<f64>>,
    precision: u64,
    filtering: u64,
) -> (&PyArray4<u64>, &PyArray3<f64>) {
    let max_bound: u64 = 1 << precision;
    let max_bound_f64: f64 = max_bound as f64;
    let max_bound_u = max_bound as usize;
    let rows = max_bound_u;
    let columns = max_bound_u;
    let channels: usize = 3;
    let images: usize = 6;
    let initial_colors: u64 = 255;
    let mut img = Array::from_elem((images, rows, columns, channels), initial_colors);
    let mut ocp_map = Array::zeros((images, rows, columns));
    let mut min_depth = Array::zeros((channels, rows, columns));
    let mut max_depth = Array::from_elem((channels, rows, columns), max_bound_f64);
    let points_f = vec_to_2d_with_floor(&points);
    let colors_f = vec_to_2d_with_floor(&colors);
    let plane: [(usize, usize); 3] = [(1, 2), (0, 2), (0, 1)];
    let total_rows = points.len() as usize;
    for i in 0..total_rows {
        if points[i][0] >= max_bound_f64
            || points[i][1] >= max_bound_f64
            || points[i][2] >= max_bound_f64
        {
            continue;
        }
        for j in 0usize..3usize {
            let k1 = points_f[[i, plane[j].0]] as usize;
            let k2 = points_f[[i, plane[j].1]] as usize;
            if points[i][j] <= max_depth[[j, k1, k2]] {
                img.slice_mut(s![2 * j, k1, k2, ..])
                    .assign(&colors_f.slice(s![i, ..]));
                ocp_map[[2 * j, k1, k2]] = 1.0;
                max_depth[[j, k1, k2]] = points[i][j];
            }
            if points[i][j] >= min_depth[[j, k1, k2]] {
                img.slice_mut(s![2 * j + 1, k1, k2, ..])
                    .assign(&colors_f.slice(s![i, ..]));
                ocp_map[[2 * j + 1, k1, k2]] = 1.0;
                min_depth[[j, k1, k2]] = points[i][j];
            }
        }
    }
    if filtering == 0 {
        return (img.to_pyarray(_py), ocp_map.to_pyarray(_py));
    }
    let w = filtering as usize;
    let mut freqs: [u64; 6] = [0, 0, 0, 0, 0, 0];
    let mut bias: f64;
    for i in w..(max_bound_u - w) {
        for j in w..(max_bound_u - w) {
            bias = 1.0;
            for k in 0usize..6usize {
                let depth_channel: usize = (k / 2) as usize;
                let curr_depth = if bias == 1.0 {
                    &mut max_depth
                } else {
                    &mut min_depth
                };
                let curr_depth_slice = &curr_depth.slice(s![
                    depth_channel,
                    (i - w)..(i + w + 1),
                    (j - w)..(j + w + 1)
                ]);
                let ocp_map_slice = &ocp_map.slice(s![
                    k,
                    (i - w)..(i + w + 1),
                    (j - w)..(j + w + 1)
                ]);
                let curr_depth_filtered = curr_depth_slice * ocp_map_slice;
                let weighted_local_average =
                    (curr_depth_filtered.sum() / (ocp_map_slice.sum())) + bias * 20.0;
                if ocp_map[[k, i, j]] == 1.0
                    && curr_depth[[depth_channel, i, j]] * bias > weighted_local_average * bias
                {
                    ocp_map[[k, i, j]] = 0.0;
                    img.slice_mut(s![k, i, j, ..]).fill(255);
                    freqs[k] += 1
                }
                bias *= -1.0;
            }
        }
    }
    for i in 0..6 {
        println!("{} points removed from projection {}", &freqs[i], &i);
    }
    return (img.to_pyarray(_py), ocp_map.to_pyarray(_py));
}

#[pymodule]
fn orthographic_projector(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_projections, m)?)?;
    Ok(())
}
