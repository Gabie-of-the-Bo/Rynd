use ndarray::{ArrayD, Zip};

use crate::rynd_error;

#[derive(Clone)]
pub enum NDArrayType {
    Int, Float, Bool
}

impl TryFrom<usize> for NDArrayType {
    type Error = ();

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(NDArrayType::Int),
            2 => Ok(NDArrayType::Float),
            3 => Ok(NDArrayType::Bool),
            _ => Err(())
        }
    }
}

#[derive(Clone)]
pub enum NDArray {
    Int(ArrayD<i64>),
    Float(ArrayD<f64>),
    Bool(ArrayD<bool>)
}

impl From<ArrayD<i64>> for NDArray {
    fn from(value: ArrayD<i64>) -> Self {
        Self::Int(value)
    }
}

impl From<ArrayD<f64>> for NDArray {
    fn from(value: ArrayD<f64>) -> Self {
        Self::Float(value)
    }
}

impl From<ArrayD<bool>> for NDArray {
    fn from(value: ArrayD<bool>) -> Self {
        Self::Bool(value)
    }
}

macro_rules! arr_zip {
    ($a: ident, $b: ident, $op: expr) => {
        Zip::from($a).and($b).map_collect(|$a, $b| $op)
    };
}

macro_rules! broadcast_op_general {
    ($a: ident, $b: ident, $aa: ident, $bb: ident, $op: tt, $l_op: expr) => {
        match ($a, $b) {
            (NDArray::Int($aa), NDArray::Int($bb)) => NDArray::from($aa $op $bb),
            (NDArray::Int($aa), NDArray::Float($bb)) => NDArray::from($aa.mapv(|i| i as f64) $op $bb),
            (NDArray::Int($aa), NDArray::Bool($bb)) => NDArray::from($aa $op $bb.mapv(|i| i as i64)),
            (NDArray::Float($aa), NDArray::Int($bb)) => NDArray::from($aa $op $bb.mapv(|i| i as f64)),
            (NDArray::Float($aa), NDArray::Float($bb)) => NDArray::from($aa $op $bb),
            (NDArray::Float($aa), NDArray::Bool($bb)) => NDArray::from($aa $op $bb.mapv(|i| i as i64 as f64)),
            (NDArray::Bool($aa), NDArray::Int($bb)) => NDArray::from($aa.mapv(|i| i as i64) $op $bb),
            (NDArray::Bool($aa), NDArray::Float($bb)) => NDArray::from($aa.mapv(|i| i as i64 as f64) $op $bb),
            (NDArray::Bool($aa), NDArray::Bool($bb)) => $l_op,
        }
    };
}

macro_rules! broadcast_op {
    ($a: ident, $b: ident, $aa: ident, $bb: ident, $op: tt, $l_op: tt) => {
        broadcast_op_general!($a, $b, $aa, $bb, $op, NDArray::from($aa $l_op $bb))
    };
}

macro_rules! zip_op {
    ($a: ident, $b: ident, $aa: ident, $bb: ident, $op: tt) => {
        match ($a, $b) {
            (NDArray::Int($aa), NDArray::Int($bb)) => NDArray::from(arr_zip!($aa, $bb, $aa $op $bb)),
            (NDArray::Int($aa), NDArray::Float($bb)) => NDArray::from(arr_zip!($aa, $bb, *$aa as f64 $op *$bb)),
            (NDArray::Int($aa), NDArray::Bool($bb)) => NDArray::from(arr_zip!($aa, $bb, *$aa $op *$bb as i64)),
            (NDArray::Float($aa), NDArray::Int($bb)) => NDArray::from(arr_zip!($aa, $bb, *$aa $op *$bb as f64)),
            (NDArray::Float($aa), NDArray::Float($bb)) => NDArray::from(arr_zip!($aa, $bb, $aa $op $bb)),
            (NDArray::Float($aa), NDArray::Bool($bb)) => NDArray::from(arr_zip!($aa, $bb, *$aa $op *$bb as i64 as f64)),
            (NDArray::Bool($aa), NDArray::Int($bb)) => NDArray::from(arr_zip!($aa, $bb, *$aa as i64 $op *$bb)),
            (NDArray::Bool($aa), NDArray::Float($bb)) => NDArray::from(arr_zip!($aa, $bb, *$aa as i64 as f64 $op *$bb)),
            (NDArray::Bool($aa), NDArray::Bool($bb)) => NDArray::from(arr_zip!($aa, $bb, $aa $op $bb)),
        }
    };
}

impl NDArray {
    pub fn new(tp: NDArrayType, shape: Vec<usize>) -> Self {
        match tp {
            NDArrayType::Int => NDArray::Int(ArrayD::default(shape)),
            NDArrayType::Float => NDArray::Float(ArrayD::default(shape)),
            NDArrayType::Bool => NDArray::Bool(ArrayD::default(shape)),
        }
    }

    pub fn cast(&self, tp: NDArrayType) -> Self {
        match (tp, self) {
            (NDArrayType::Int, NDArray::Int(array)) => NDArray::from(array.clone()),
            (NDArrayType::Int, NDArray::Float(array)) => NDArray::from(array.mapv(|i| i as i64)),
            (NDArrayType::Int, NDArray::Bool(array)) => NDArray::from(array.mapv(|i| i as i64)),
            (NDArrayType::Float, NDArray::Int(array)) => NDArray::from(array.mapv(|i| i as f64)),
            (NDArrayType::Float, NDArray::Float(array)) => NDArray::from(array.clone()),
            (NDArrayType::Float, NDArray::Bool(array)) => NDArray::from(array.mapv(|i| i as i64 as f64)),
            (NDArrayType::Bool, NDArray::Int(array)) => NDArray::from(array.mapv(|i| i != 0)),
            (NDArrayType::Bool, NDArray::Float(array)) => NDArray::from(array.mapv(|i| i != 0.0)),
            (NDArrayType::Bool, NDArray::Bool(array)) => NDArray::from(array.clone()),
        }
    }

    pub fn sum(&self, other: &NDArray) -> Self {
        broadcast_op!(self, other, a, b, +, ^)
    }

    pub fn sub(&self, other: &NDArray) -> Self {
        broadcast_op!(self, other, a, b, -, ^)
    }

    pub fn mul(&self, other: &NDArray) -> Self {
        broadcast_op!(self, other, a, b, *, &)
    }

    pub fn div(&self, other: &NDArray) -> Self {
        broadcast_op_general!(self, other, _a, _b, *, rynd_error!("Unable to divide two boolean arrays"))
    }

    pub fn eq(&self, other: &NDArray) -> Self {
        zip_op!(self, other, a, b, ==)
    }

    pub fn neq(&self, other: &NDArray) -> Self {
        zip_op!(self, other, a, b, !=)
    }

    pub fn pow(&self, other: &NDArray) -> Self {
        match (self, other) {
            (NDArray::Int(a), NDArray::Int(b)) => NDArray::from(arr_zip!(a, b, (*a as f64).powf(*b as f64))),
            (NDArray::Int(a), NDArray::Float(b)) => NDArray::from(arr_zip!(a, b, (*a as f64).powf(*b))),
            (NDArray::Int(a), NDArray::Bool(b)) => NDArray::from(arr_zip!(a, b, (*a as f64).powf(*b as i64 as f64))),
            (NDArray::Float(a), NDArray::Int(b)) => NDArray::from(arr_zip!(a, b, a.powf(*b as f64))),
            (NDArray::Float(a), NDArray::Float(b)) => NDArray::from(arr_zip!(a, b, a.powf(*b))),
            (NDArray::Float(a), NDArray::Bool(b)) => NDArray::from(arr_zip!(a, b, a.powf(*b as i64 as f64))),
            (NDArray::Bool(a), NDArray::Int(b)) => NDArray::from(arr_zip!(a, b, (*a as i64 as f64).powf(*b as f64))),
            (NDArray::Bool(a), NDArray::Float(b)) => NDArray::from(arr_zip!(a, b, (*a as i64 as f64).powf(*b as f64))),
            (NDArray::Bool(a), NDArray::Bool(b)) => NDArray::from(a & b),
        }
    }
}

impl std::fmt::Display for NDArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NDArray::Int(array) => write!(f, "{array}"),
            NDArray::Float(array) => write!(f, "{array}"),
            NDArray::Bool(array) => write!(f, "{array}"),
        }
    }
}