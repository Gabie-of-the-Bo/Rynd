use ndarray::{Array, ArrayView, ArrayViewMut, Axis, Dimension, IntoDimension, Ix2, RemoveAxis, Slice};
use rand_distr::num_traits::Zero;

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

pub fn reverse_axis<'a, T, D>(view: &'a mut ArrayViewMut<T, D>, axis: Axis) -> ArrayViewMut<'a, T, D>
where
    D: Dimension,
{
    view.slice_axis_mut(axis, Slice::from(0..).step_by(-1))
}

pub fn diff_axis<T, D>(view: &ArrayView<T, D>, axis: Axis) -> Array<T, D>
where
    T: Copy + std::ops::Sub<Output = T>,
    D: Dimension,
{
    let first = view.slice_axis(axis, Slice::from(..-1));
    let second = view.slice_axis(axis, Slice::from(1..));

    &second - &first
}

pub fn cumsum_axis<T, D>(view: &ArrayView<T, D>, axis: Axis) -> Array<T, D>
where
    T: Copy + std::ops::Add<Output = T>,
    D: Dimension,
{
    let mut result = view.to_owned();
    
    for mut lane in result.lanes_mut(axis) {
        if !lane.is_empty() {
            for i in 1..lane.len() {
                lane[i] = lane[i - 1] + lane[i];
            }
        }
    }

    result
}

pub fn min_axis<T, D>(view: &ArrayView<T, D>, axis: Axis) -> Array<T, D::Smaller>
where
    T: Copy + PartialOrd,
    D: RemoveAxis,
{
    view.map_axis(axis, |lane| {
        lane.iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    })
}

pub fn max_axis<T, D>(view: &ArrayView<T, D>, axis: Axis) -> Array<T, D::Smaller>
where
    T: Copy + PartialOrd,
    D: RemoveAxis,
{
    view.map_axis(axis, |lane| {
        lane.iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    })
}

pub fn argmin_axis<T, D>(view: &ArrayView<T, D>, axis: Axis) -> Array<i64, D::Smaller>
where
    T: Copy + PartialOrd,
    D: RemoveAxis,
{
    view.map_axis(axis, |lane| {
        lane.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as i64)
            .unwrap()
    })
}

pub fn argmax_axis<T, D>(view: &ArrayView<T, D>, axis: Axis) -> Array<i64, D::Smaller>
where
    T: Copy + PartialOrd,
    D: RemoveAxis,
{
    view.map_axis(axis, |lane| {
        lane.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as i64)
            .unwrap()
    })
}

pub fn nonzero<T, D>(view: &ArrayView<T, D>) -> Array<i64, Ix2>
where
    T: PartialEq + Zero,
    D: Dimension,
{
    let ndim = view.ndim();

    let indices: Vec<i64> = view
        .indexed_iter()
        .filter(|(_, value)| **value != T::zero())
        .flat_map(|(index, _)| {
            index.into_dimension().slice().iter().map(|&ix| ix as i64).collect::<Vec<_>>()
        })
        .collect();
    
    let num_nonzero = indices.len() / ndim;
    
    Array::from_shape_vec((num_nonzero, ndim), indices).unwrap()
}