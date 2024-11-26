use ndarray::{Dim, IxDynImpl, RawArrayView, Zip};

use crate::{owned::NDArrayOwned, rynd_error};

type DynRawArrayView<T> = RawArrayView<T, Dim<IxDynImpl>>;
#[derive(Clone)]
pub enum NDArrayView {
    Int(DynRawArrayView<i64>),
    Float(DynRawArrayView<f64>),
    Bool(DynRawArrayView<bool>),
}

impl From<DynRawArrayView<i64>> for NDArrayView {
    fn from(value: DynRawArrayView<i64>) -> Self {
        Self::Int(value)
    }
}

impl From<DynRawArrayView<f64>> for NDArrayView {
    fn from(value: DynRawArrayView<f64>) -> Self {
        Self::Float(value)
    }
}

impl From<DynRawArrayView<bool>> for NDArrayView {
    fn from(value: DynRawArrayView<bool>) -> Self {
        Self::Bool(value)
    }
}

macro_rules! view {
    ($raw: expr) => {
        &unsafe { $raw.clone().deref_into_view() }  
    };
}

macro_rules! arr_zip {
    ($a: ident, $b: ident, $op: expr) => {
        Zip::from(view!($a)).and(view!($b)).map_collect(|$a, $b| $op)
    };
}

macro_rules! broadcast_op_general {
    ($a: ident, $b: ident, $aa: ident, $bb: ident, $op: tt, $l_op: expr) => {
        match ($a, $b) {
            (NDArrayView::Int($aa), NDArrayView::Int($bb)) => NDArrayOwned::from(view!($aa) $op view!($bb)),
            (NDArrayView::Int($aa), NDArrayView::Float($bb)) => NDArrayOwned::from(view!($aa).mapv(|i| i as f64) $op view!($bb)),
            (NDArrayView::Int($aa), NDArrayView::Bool($bb)) => NDArrayOwned::from(view!($aa) $op view!($bb).mapv(|i| i as i64)),
            (NDArrayView::Float($aa), NDArrayView::Int($bb)) => NDArrayOwned::from(view!($aa) $op view!($bb).mapv(|i| i as f64)),
            (NDArrayView::Float($aa), NDArrayView::Float($bb)) => NDArrayOwned::from(view!($aa) $op view!($bb)),
            (NDArrayView::Float($aa), NDArrayView::Bool($bb)) => NDArrayOwned::from(view!($aa) $op view!($bb).mapv(|i| i as i64 as f64)),
            (NDArrayView::Bool($aa), NDArrayView::Int($bb)) => NDArrayOwned::from(view!($aa).mapv(|i| i as i64) $op view!($bb)),
            (NDArrayView::Bool($aa), NDArrayView::Float($bb)) => NDArrayOwned::from(view!($aa).mapv(|i| i as i64 as f64) $op view!($bb)),
            (NDArrayView::Bool($aa), NDArrayView::Bool($bb)) => $l_op,
        }
    };
}

macro_rules! broadcast_op {
    ($a: ident, $b: ident, $aa: ident, $bb: ident, $op: tt, $l_op: tt) => {
        broadcast_op_general!($a, $b, $aa, $bb, $op, NDArrayOwned::from(view!($aa) $l_op view!($bb)))
    };
}

macro_rules! zip_op {
    ($a: ident, $b: ident, $aa: ident, $bb: ident, $op: tt) => {
        match ($a, $b) {
            (NDArrayView::Int($aa), NDArrayView::Int($bb)) => NDArrayOwned::from(arr_zip!($aa, $bb, $aa $op $bb)),
            (NDArrayView::Int($aa), NDArrayView::Float($bb)) => NDArrayOwned::from(arr_zip!($aa, $bb, *$aa as f64 $op *$bb)),
            (NDArrayView::Int($aa), NDArrayView::Bool($bb)) => NDArrayOwned::from(arr_zip!($aa, $bb, *$aa $op *$bb as i64)),
            (NDArrayView::Float($aa), NDArrayView::Int($bb)) => NDArrayOwned::from(arr_zip!($aa, $bb, *$aa $op *$bb as f64)),
            (NDArrayView::Float($aa), NDArrayView::Float($bb)) => NDArrayOwned::from(arr_zip!($aa, $bb, $aa $op $bb)),
            (NDArrayView::Float($aa), NDArrayView::Bool($bb)) => NDArrayOwned::from(arr_zip!($aa, $bb, *$aa $op *$bb as i64 as f64)),
            (NDArrayView::Bool($aa), NDArrayView::Int($bb)) => NDArrayOwned::from(arr_zip!($aa, $bb, *$aa as i64 $op *$bb)),
            (NDArrayView::Bool($aa), NDArrayView::Float($bb)) => NDArrayOwned::from(arr_zip!($aa, $bb, *$aa as i64 as f64 $op *$bb)),
            (NDArrayView::Bool($aa), NDArrayView::Bool($bb)) => NDArrayOwned::from(arr_zip!($aa, $bb, $aa $op $bb)),
        }
    };
}

impl NDArrayView {
    pub fn owned(&self) -> NDArrayOwned {
        match self {
            NDArrayView::Int(v) => view!(v).to_owned().into(),
            NDArrayView::Float(v) => view!(v).to_owned().into(),
            NDArrayView::Bool(v) => view!(v).to_owned().into(),
        }
    }

    pub fn sum(&self, other: &NDArrayView) -> NDArrayOwned {
        broadcast_op!(self, other, a, b, +, ^)
    }

    pub fn sub(&self, other: &NDArrayView) -> NDArrayOwned {
        broadcast_op!(self, other, a, b, -, ^)
    }

    pub fn mul(&self, other: &NDArrayView) -> NDArrayOwned {
        broadcast_op!(self, other, a, b, *, &)
    }

    pub fn div(&self, other: &NDArrayView) -> NDArrayOwned {
        broadcast_op_general!(self, other, _a, _b, *, rynd_error!("Unable to divide two boolean arrays"))
    }

    pub fn eq(&self, other: &NDArrayView) -> NDArrayOwned {
        zip_op!(self, other, a, b, ==)
    }

    pub fn neq(&self, other: &NDArrayView) -> NDArrayOwned {
        zip_op!(self, other, a, b, !=)
    }

    pub fn pow(&self, other: &NDArrayView) -> NDArrayOwned {
        match (self, other) {
            (NDArrayView::Int(a), NDArrayView::Int(b)) => NDArrayOwned::from(arr_zip!(a, b, (*a as f64).powf(*b as f64))),
            (NDArrayView::Int(a), NDArrayView::Float(b)) => NDArrayOwned::from(arr_zip!(a, b, (*a as f64).powf(*b))),
            (NDArrayView::Int(a), NDArrayView::Bool(b)) => NDArrayOwned::from(arr_zip!(a, b, (*a as f64).powf(*b as i64 as f64))),
            (NDArrayView::Float(a), NDArrayView::Int(b)) => NDArrayOwned::from(arr_zip!(a, b, a.powf(*b as f64))),
            (NDArrayView::Float(a), NDArrayView::Float(b)) => NDArrayOwned::from(arr_zip!(a, b, a.powf(*b))),
            (NDArrayView::Float(a), NDArrayView::Bool(b)) => NDArrayOwned::from(arr_zip!(a, b, a.powf(*b as i64 as f64))),
            (NDArrayView::Bool(a), NDArrayView::Int(b)) => NDArrayOwned::from(arr_zip!(a, b, (*a as i64 as f64).powf(*b as f64))),
            (NDArrayView::Bool(a), NDArrayView::Float(b)) => NDArrayOwned::from(arr_zip!(a, b, (*a as i64 as f64).powf(*b as f64))),
            (NDArrayView::Bool(a), NDArrayView::Bool(b)) => NDArrayOwned::from(view!(a) & view!(b)),
        }
    }
}

impl std::fmt::Display for NDArrayView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NDArrayView::Int(a) => write!(f, "{}", view!(a)),
            NDArrayView::Float(a) => write!(f, "{}", view!(a)),
            NDArrayView::Bool(a) => write!(f, "{}", view!(a)),
        }
    }
}