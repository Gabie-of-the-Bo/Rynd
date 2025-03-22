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