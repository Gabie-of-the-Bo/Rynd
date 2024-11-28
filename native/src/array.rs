use std::fmt::Display;

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

impl NDArray {
    pub fn new(tp: NDArrayType, shape: Vec<usize>) -> Self {
        NDArray::from(NDArrayOwned::new(tp, shape))
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
    
    view_binop!(sum);
    view_binop!(sub);
    view_binop!(mul);
    view_binop!(div);
    view_binop!(pow);
    view_binop!(eq);
    view_binop!(neq);
}

impl Display for NDArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NDArray::Owned(a) => write!(f, "{}", a),
            NDArray::View(v) => write!(f, "{}", v),
        }
    }
}