use nd::Array2;
use num::ToPrimitive;
use numpy::PyReadonlyArray2;

pub fn to_ndarray<'a, T>(pyarray: &'a PyReadonlyArray2<T>) -> Array2<f64>
where
    T: numpy::Element + ToPrimitive,
{
    let v: Vec<f64> = pyarray
        .as_array()
        .iter()
        .cloned()
        .map(|x| x.to_f64().unwrap())
        .collect();
    let shape = (pyarray.shape()[0], pyarray.shape()[1]);
    Array2::from_shape_vec(shape, v).unwrap()
}
