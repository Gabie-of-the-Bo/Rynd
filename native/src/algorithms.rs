use ndarray::{Array, ArrayView, ArrayViewMut, Axis, Dimension, RemoveAxis};

use crate::rynd_error;

pub fn sort_view_axis<T, D>(mut view: ArrayViewMut<T, D>, axis: Axis)
where
    T: PartialOrd + Clone,
    D: Dimension,
{
    for mut lane in view.lanes_mut(axis) {
        if let Some(slice) = lane.as_slice_mut() {
            slice.sort_by(|a, b| a.partial_cmp(b).unwrap());

        } else {
            let mut vec: Vec<T> = lane.iter().cloned().collect();
            vec.sort_by(|a, b| a.partial_cmp(b).unwrap());

            for (elem, sorted_val) in lane.iter_mut().zip(vec) {
                *elem = sorted_val;
            }
        }
    }
}

pub fn argsort_axis<T, D>(view: &ArrayView<T, D>, axis: Axis) -> Array<i64, D>
where
    T: PartialOrd,
    D: Dimension,
{
    let mut result = Array::<i64, D>::zeros(view.raw_dim());

    for (lane, mut indices_lane) in view.lanes(axis).into_iter().zip(result.lanes_mut(axis)) {
        let mut idx: Vec<i64> = (0..lane.len()).map(|i| i as i64).collect();
        idx.sort_by(|&i, &j| lane[i as usize].partial_cmp(&lane[j as usize]).unwrap());

        for (dest, &i) in indices_lane.iter_mut().zip(idx.iter()) {
            *dest = i;
        }
    }
    result
}

pub fn stack_axis<'a, T, D>(a: &ArrayView<'a, T, D>, b: &ArrayView<'a, T, D>, axis: Axis) -> Array<T, D::Larger>
where
    T: PartialOrd + Clone,
    D: Dimension,
{
    match ndarray::stack(axis, &[a.clone(), b.clone()]) {
        Ok(r) => r,
        Err(_) => rynd_error!("Unable to stack arrays of shape {:?} and {:?} over axis {}", a.shape(), b.shape(), axis.0),
    }
}

pub fn concat_axis<'a, T, D>(a: &ArrayView<'a, T, D>, b: &ArrayView<'a, T, D>, axis: Axis) -> Array<T, D>
where
    T: PartialOrd + Clone,
    D: Dimension + RemoveAxis,
{
    match ndarray::concatenate(axis, &[a.clone(), b.clone()]) {
        Ok(r) => r,
        Err(_) => rynd_error!("Unable to concatenate arrays of shape {:?} and {:?} over axis {}", a.shape(), b.shape(), axis.0),
    }
}