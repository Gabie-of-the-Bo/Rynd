use ndarray::Slice;

use crate::array::NDArray;

#[macro_export]
macro_rules! rynd_error {
    ($pat: expr $( , $more: expr)*) => {
        {
            use colored::Colorize;

            eprintln!(
                "[{}] {}",
                "Error".red(),
                format!($pat, $($more,)*)
            );
    
            std::process::exit(1);
        }
    };
}

pub fn rynd_matmul_check(a: &NDArray, b: &NDArray) {
    let shape_a = a.shape();
    let shape_b = b.shape();
    
    if shape_a.len() != 2 {
        rynd_error!("Expected left operand of matrix multiplication to be of dimension 2 (shape is {:?})", shape_a);
    }
    
    if shape_b.len() != 2 {
        rynd_error!("Expected right operand of matrix multiplication to be of dimension 2 (shape is {:?})", shape_b);
    }

    if shape_a[1] != shape_b[0] {
        rynd_error!("Incompatible array shapes for matrix multiplication ({:?} x {:?})", shape_a, shape_b);
    }
}

pub fn rynd_normalize_dim(arr: &NDArray, dim: &mut i64) {
    let shape = arr.shape();
    let orig = *dim;

    if *dim < 0 {
        *dim += shape.len() as i64;
    }

    if *dim < 0 || *dim as usize >= shape.len() {
        rynd_error!("Dimension {} is invalid (shape is {:?})", orig, shape);
    }
}

pub fn rynd_dims_check(arr: &NDArray, min_dims: Option<usize>, max_dims: Option<usize>) {
    let shape = arr.shape();

    if let Some(i) = min_dims {
        if shape.len() < i {
            rynd_error!("Expected array to have at least {} dimensions (it has {})", i, shape.len());
        }
    }

    if let Some(i) = max_dims {
        if shape.len() > i {
            rynd_error!("Expected array to have at most {} dimensions (it has {})", i, shape.len());
        }
    }
}

pub fn rynd_slice_check(arr: &NDArray, slice: &Slice, dim_idx: usize) {
    let shape = arr.shape();
    let dim = shape[dim_idx];

    if slice.start.abs() as usize >= dim {
        rynd_error!("Slice start out of bounds for dimension {} ({} >= {})", dim_idx, slice.start, dim);
    }

    if let Some(end) = slice.end {
        if end.abs() as usize > dim {
            rynd_error!("Slice end out of bounds for dimension {} ({} > {})", dim_idx, end, dim);
        }
    }
}