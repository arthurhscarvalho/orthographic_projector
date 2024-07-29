use nd::{s, Array2, Array3, Array4};

fn floor_array<'a>(array: &'a Array2<f64>) -> Array2<u64> {
    let v: Vec<u64> = array.iter().cloned().map(|x| x.floor() as u64).collect();
    let shape = (array.shape()[0], array.shape()[1]);
    Array2::from_shape_vec(shape, v).unwrap()
}

pub fn compute_projections(
    points: Array2<f64>,
    colors: Array2<f64>,
    precision: u64,
    filtering: u64,
) -> (Array4<u64>, Array3<f64>, [u64; 6]) {
    let max_bound: u64 = 1 << precision;
    let max_bound_f64: f64 = max_bound as f64;
    let max_bound_u = max_bound as usize;
    let rows = max_bound_u;
    let columns = max_bound_u;
    let channels: usize = 3;
    let num_images: usize = 6;
    let initial_colors: u64 = 255;
    let mut images = Array4::from_elem((num_images, rows, columns, channels), initial_colors);
    let mut ocp_maps = Array3::zeros((num_images, rows, columns));
    let mut min_depth = Array3::zeros((channels, rows, columns));
    let mut max_depth = Array3::from_elem((channels, rows, columns), max_bound_f64);
    let (points_f, colors_f) = (floor_array(&points), floor_array(&colors));
    let plane: [(usize, usize); 3] = [(1, 2), (0, 2), (0, 1)];
    let total_rows = points.nrows() as usize;
    for i in 0..total_rows {
        if points[[i, 0]] >= max_bound_f64
            || points[[i, 1]] >= max_bound_f64
            || points[[i, 2]] >= max_bound_f64
        {
            continue;
        }
        for j in 0usize..3usize {
            let k1 = points_f[[i, plane[j].0]] as usize;
            let k2 = points_f[[i, plane[j].1]] as usize;
            if points[[i, j]] <= max_depth[[j, k1, k2]] {
                images
                    .slice_mut(s![2 * j, k1, k2, ..])
                    .assign(&colors_f.slice(s![i, ..]));
                ocp_maps[[2 * j, k1, k2]] = 1.0;
                max_depth[[j, k1, k2]] = points[[i, j]];
            }
            if points[[i, j]] >= min_depth[[j, k1, k2]] {
                images
                    .slice_mut(s![2 * j + 1, k1, k2, ..])
                    .assign(&colors_f.slice(s![i, ..]));
                ocp_maps[[2 * j + 1, k1, k2]] = 1.0;
                min_depth[[j, k1, k2]] = points[[i, j]];
            }
        }
    }
    let mut freqs: [u64; 6] = [0, 0, 0, 0, 0, 0];
    if filtering == 0 {
        return (images, ocp_maps, freqs);
    }
    let w = filtering as usize;
    let mut bias: f64;
    for i in w..(max_bound_u - w) {
        for j in w..(max_bound_u - w) {
            bias = 1.0;
            for k in 0usize..6usize {
                let depth_channel = (k / 2) as usize;
                let swap_depth = if bias == 1.0 {
                    &mut max_depth
                } else {
                    &mut min_depth
                };
                let curr_depth_slice = &swap_depth.slice(s![
                    depth_channel,
                    (i - w)..(i + w + 1),
                    (j - w)..(j + w + 1)
                ]);
                let ocp_map_slice =
                    &ocp_maps.slice(s![k, (i - w)..(i + w + 1), (j - w)..(j + w + 1)]);
                let curr_depth_filtered = curr_depth_slice * ocp_map_slice;
                let weighted_local_average =
                    (curr_depth_filtered.sum() / ocp_map_slice.sum()) + bias * 20.0;
                if ocp_maps[[k, i, j]] == 1.0
                    && swap_depth[[depth_channel, i, j]] * bias > weighted_local_average * bias
                {
                    ocp_maps[[k, i, j]] = 0.0;
                    images.slice_mut(s![k, i, j, ..]).fill(255);
                    freqs[k] += 1
                }
                bias *= -1.0;
            }
        }
    }
    (images, ocp_maps, freqs)
}
