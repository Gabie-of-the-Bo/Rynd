use ndarray::{ArrayViewMut, Axis, Dimension};

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