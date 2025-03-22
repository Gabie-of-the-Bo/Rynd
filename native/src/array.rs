use std::{fmt::Display, os::raw::c_void};

use ndarray::Slice;

use crate::{owned::{NDArrayOwned, NDArrayType}, rynd_error, view::NDArrayView};

#[derive(Clone)]
pub enum NDArray {
    Owned(NDArrayOwned),
    View(NDArrayView)
}

impl From<NDArrayOwned> for NDArray {
    fn from(value: NDArrayOwned) -> Self {
        NDArray::Owned(value)
    }
}

impl From<NDArrayView> for NDArray {
    fn from(value: NDArrayView) -> Self {
        NDArray::View(value)
    }
}

macro_rules! view_binop {
    ($name: ident) => {
        pub fn $name(&mut self, other: &mut NDArray) -> NDArray {
            NDArray::from(self.view().$name(&other.view()))
        }
    };
}

macro_rules! view_binop_scalar {
    ($name: ident, $t: ty) => {
        pub fn $name(&mut self, scalar: $t, reverse: bool) -> NDArray {
            NDArray::from(self.view().$name(scalar, reverse))
        }
    };
}

impl NDArray {
    pub fn new(tp: NDArrayType, shape: Vec<usize>) -> Self {
        NDArray::from(NDArrayOwned::new(tp, shape))
    }

    pub fn from_ptr(tp: NDArrayType, shape: Vec<usize>, ptr: *const c_void) -> Self {
        let mut arr = NDArrayOwned::new(tp, shape.clone());

        match &mut arr {
            NDArrayOwned::Int(a) => {
                let slice = unsafe { std::slice::from_raw_parts(ptr as *const i64, shape.iter().product()) };
                a.iter_mut().zip(slice).for_each(|(i, v)| { *i = *v; });
            },

            NDArrayOwned::Float(a) => {
                let slice = unsafe { std::slice::from_raw_parts(ptr as *const f64, shape.iter().product()) };
                a.iter_mut().zip(slice).for_each(|(i, v)| { *i = *v; });
            },

            NDArrayOwned::Bool(a) => {
                let slice = unsafe { std::slice::from_raw_parts(ptr as *const i64, shape.iter().product()) };
                a.iter_mut().zip(slice).for_each(|(i, v)| { *i = *v != 0; });
            }
        }

        arr.into()
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            NDArray::Owned(a) => a.shape(),
            NDArray::View(v) => v.shape(),
        }
    }

    pub fn view(&mut self) -> NDArrayView {
        match self {
            NDArray::Owned(a) => a.view(),
            NDArray::View(v) => v.clone(),
        }
    }

    pub fn cast(&mut self, tp: NDArrayType) -> Self {
        match self {
            NDArray::Owned(a) => a.cast(tp).into(),
            NDArray::View(v) => v.owned().cast(tp).into(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            NDArray::Owned(a) => a.len(),
            NDArray::View(v) => v.len(),
        }
    }

    pub fn get_i64(&mut self, idx: usize) -> i64 {
        match self {
            NDArray::Owned(a) => a.view().get_i64(idx),
            NDArray::View(v) => v.get_i64(idx),
        }
    }

    pub fn get_f64(&mut self, idx: usize) -> f64 {
        match self {
            NDArray::Owned(a) => a.view().get_f64(idx),
            NDArray::View(v) => v.get_f64(idx),
        }
    }

    pub fn get_bool(&mut self, idx: usize) -> bool {
        match self {
            NDArray::Owned(a) => a.view().get_bool(idx),
            NDArray::View(v) => v.get_bool(idx),
        }
    }

    pub fn index(&mut self, idx: &mut NDArray) -> Self {
        let obj = self.view();
        let idx_view = idx.view();

        obj.index(&idx_view).into()
    }

    fn compatible_shapes(a: &[usize], b: &[usize]) -> bool {
        a.iter().product::<usize>() == b.iter().product::<usize>()
    }

    pub fn reshape(&mut self, shape: Vec<usize>) -> Self {
        if !Self::compatible_shapes(self.shape(), &shape) {
            rynd_error!("Unable to reshape array with shape {:?} to shape {:?}", self.shape(), shape);
        }

        match self {
            NDArray::Owned(a) => a.reshape(shape).into(),
            NDArray::View(v) => v.reshape(shape).into(),
        }
    }

    pub fn slice(&mut self, slices: Vec<Slice>) -> Self {
        match self {
            NDArray::Owned(a) => a.view().slice(slices).into(),
            NDArray::View(v) => v.slice(slices).into(),
        }
    }

    pub fn assign(&mut self, other: &mut NDArray) {
        match self {
            NDArray::Owned(a) => a.view().assign(&other.view()),
            NDArray::View(v) => v.assign(&other.view()),
        }
    }
    
    view_binop!(sum);
    view_binop!(sub);
    view_binop!(mul);
    view_binop!(div);
    view_binop!(modulo);
    view_binop!(pow);
    view_binop!(eq);
    view_binop!(neq);
    view_binop!(lt);
    view_binop!(gt);
    view_binop!(leq);
    view_binop!(geq);

    pub fn assign_scalar_i64(&mut self, other: i64) {
        match self {
            NDArray::Owned(a) => a.view().assign_scalar_i64(other),
            NDArray::View(v) => v.assign_scalar_i64(other),
        }
    }

    pub fn assign_scalar_f64(&mut self, other: f64) {
        match self {
            NDArray::Owned(a) => a.view().assign_scalar_f64(other),
            NDArray::View(v) => v.assign_scalar_f64(other),
        }
    }

    view_binop_scalar!(sum_scalar_i64, i64);
    view_binop_scalar!(sum_scalar_f64, f64);

    view_binop_scalar!(sub_scalar_i64, i64);
    view_binop_scalar!(sub_scalar_f64, f64);

    view_binop_scalar!(mul_scalar_i64, i64);
    view_binop_scalar!(mul_scalar_f64, f64);

    view_binop_scalar!(div_scalar_i64, i64);
    view_binop_scalar!(div_scalar_f64, f64);

    view_binop_scalar!(mod_scalar_i64, i64);
    view_binop_scalar!(mod_scalar_f64, f64);

    view_binop_scalar!(pow_scalar_i64, i64);
    view_binop_scalar!(pow_scalar_f64, f64);

    view_binop_scalar!(eq_scalar_i64, i64);
    view_binop_scalar!(eq_scalar_f64, f64);

    view_binop_scalar!(neq_scalar_i64, i64);
    view_binop_scalar!(neq_scalar_f64, f64);

    view_binop_scalar!(lt_scalar_i64, i64);
    view_binop_scalar!(lt_scalar_f64, f64);

    view_binop_scalar!(gt_scalar_i64, i64);
    view_binop_scalar!(gt_scalar_f64, f64);

    view_binop_scalar!(leq_scalar_i64, i64);
    view_binop_scalar!(leq_scalar_f64, f64);

    view_binop_scalar!(geq_scalar_i64, i64);
    view_binop_scalar!(geq_scalar_f64, f64);

    pub fn axis_sum(&mut self, axis: Option<usize>) -> NDArray {
        match self {
            NDArray::Owned(a) => a.view().axis_sum(axis).into(),
            NDArray::View(a) => a.axis_sum(axis).into(),
        }
    }

    pub fn axis_sort(&mut self, axis: usize) {
        match self {
            NDArray::Owned(a) => a.view().axis_sort(axis).into(),
            NDArray::View(a) => a.axis_sort(axis).into(),
        }
    }
}

impl Display for NDArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NDArray::Owned(a) => write!(f, "{}", a),
            NDArray::View(v) => write!(f, "{}", v),
        }
    }
}