use ndarray::Array2;

pub fn relu(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| if v > 0.0 { v } else { 0.0 })
}

pub fn relu_derivative(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
}

pub fn softmax(x: &Array2<f64>) -> Array2<f64> {
    let max_x = x.fold_axis(ndarray::Axis(1), f64::NEG_INFINITY, |&a, &b| a.max(b));
    let exp_x = (x - &max_x.insert_axis(ndarray::Axis(1))).mapv(f64::exp);
    let sum_exp_x = exp_x.sum_axis(ndarray::Axis(1)).insert_axis(ndarray::Axis(1));
    exp_x / sum_exp_x
}