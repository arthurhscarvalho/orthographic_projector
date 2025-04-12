use nd::Array2;
use num::ToPrimitive;
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};

pub fn to_ndarray<T>(pyarray: &PyReadonlyArray2<T>) -> Array2<f64>
where
    T: numpy::Element + ToPrimitive,
{
    let shape = pyarray.shape();
    let v: Vec<f64> = pyarray
        .as_array()
        .iter()
        .filter_map(|x| x.to_f64())
        .collect();
    Array2::from_shape_vec((shape[0], shape[1]), v).unwrap()
}
