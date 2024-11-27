use std::fmt::Display;

use crate::{owned::{NDArrayOwned, NDArrayType}, view::NDArrayView};

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
        pub fn $name(&self, other: &NDArray) -> NDArray {
            NDArray::from(self.view().$name(&other.view()))
        }
    };
}

impl NDArray {
    pub fn new(tp: NDArrayType, shape: Vec<usize>) -> Self {
        NDArray::from(NDArrayOwned::new(tp, shape))
    }

    pub fn view(&self) -> NDArrayView {
        match self {
            NDArray::Owned(a) => a.view(),
            NDArray::View(v) => v.clone(),
        }
    }

    pub fn cast(&self, tp: NDArrayType) -> Self {
        match self {
            NDArray::Owned(a) => a.cast(tp).into(),
            NDArray::View(v) => v.owned().cast(tp).into(),
        }
    }

    pub fn index(&self, idx: &NDArray) -> Self {
        let obj = self.view();
        let idx_view = idx.view();

        obj.index(&idx_view).into()
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
        write!(f, "{}", self.view())
    }
}