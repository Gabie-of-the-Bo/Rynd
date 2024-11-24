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

macro_rules! arr_zip {
    ($a: ident, $b: ident, $op: expr) => {
        Zip::from($a).and($b).map_collect(|$a, $b| $op)
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
            (NDArrayType::Int, NDArray::Int(array)) => NDArray::Int(array.clone()),
            (NDArrayType::Int, NDArray::Float(array)) => NDArray::Int(array.mapv(|i| i as i64)),
            (NDArrayType::Int, NDArray::Bool(array)) => NDArray::Int(array.mapv(|i| i as i64)),
            (NDArrayType::Float, NDArray::Int(array)) => NDArray::Float(array.mapv(|i| i as f64)),
            (NDArrayType::Float, NDArray::Float(array)) => NDArray::Float(array.clone()),
            (NDArrayType::Float, NDArray::Bool(array)) => NDArray::Float(array.mapv(|i| i as i64 as f64)),
            (NDArrayType::Bool, NDArray::Int(array)) => NDArray::Bool(array.mapv(|i| i != 0)),
            (NDArrayType::Bool, NDArray::Float(array)) => NDArray::Bool(array.mapv(|i| i != 0.0)),
            (NDArrayType::Bool, NDArray::Bool(array)) => NDArray::Bool(array.clone()),
        }
    }

    pub fn sum(&self, other: &NDArray) -> Self {
        match (self, other) {
            (NDArray::Int(a), NDArray::Int(b)) => NDArray::Int(a + b),
            (NDArray::Int(a), NDArray::Float(b)) => NDArray::Float(a.mapv(|i| i as f64) + b),
            (NDArray::Int(a), NDArray::Bool(b)) => NDArray::Int(a + b.mapv(|i| i as i64)),
            (NDArray::Float(a), NDArray::Int(b)) => NDArray::Float(a + b.mapv(|i| i as f64)),
            (NDArray::Float(a), NDArray::Float(b)) => NDArray::Float(a + b),
            (NDArray::Float(a), NDArray::Bool(b)) => NDArray::Float(a + b.mapv(|i| i as i64 as f64)),
            (NDArray::Bool(a), NDArray::Int(b)) => NDArray::Int(a.mapv(|i| i as i64) + b),
            (NDArray::Bool(a), NDArray::Float(b)) => NDArray::Float(a.mapv(|i| i as i64 as f64) + b),
            (NDArray::Bool(a), NDArray::Bool(b)) => NDArray::Bool(a ^ b),
        }
    }

    pub fn sub(&self, other: &NDArray) -> Self {
        match (self, other) {
            (NDArray::Int(a), NDArray::Int(b)) => NDArray::Int(a - b),
            (NDArray::Int(a), NDArray::Float(b)) => NDArray::Float(a.mapv(|i| i as f64) - b),
            (NDArray::Int(a), NDArray::Bool(b)) => NDArray::Int(a - b.mapv(|i| i as i64)),
            (NDArray::Float(a), NDArray::Int(b)) => NDArray::Float(a - b.mapv(|i| i as f64)),
            (NDArray::Float(a), NDArray::Float(b)) => NDArray::Float(a - b),
            (NDArray::Float(a), NDArray::Bool(b)) => NDArray::Float(a - b.mapv(|i| i as i64 as f64)),
            (NDArray::Bool(a), NDArray::Int(b)) => NDArray::Int(a.mapv(|i| i as i64) - b),
            (NDArray::Bool(a), NDArray::Float(b)) => NDArray::Float(a.mapv(|i| i as i64 as f64) - b),
            (NDArray::Bool(a), NDArray::Bool(b)) => NDArray::Bool(a ^ b),
        }
    }

    pub fn mul(&self, other: &NDArray) -> Self {
        match (self, other) {
            (NDArray::Int(a), NDArray::Int(b)) => NDArray::Int(a * b),
            (NDArray::Int(a), NDArray::Float(b)) => NDArray::Float(a.mapv(|i| i as f64) * b),
            (NDArray::Int(a), NDArray::Bool(b)) => NDArray::Int(a * b.mapv(|i| i as i64)),
            (NDArray::Float(a), NDArray::Int(b)) => NDArray::Float(a * b.mapv(|i| i as f64)),
            (NDArray::Float(a), NDArray::Float(b)) => NDArray::Float(a * b),
            (NDArray::Float(a), NDArray::Bool(b)) => NDArray::Float(a * b.mapv(|i| i as i64 as f64)),
            (NDArray::Bool(a), NDArray::Int(b)) => NDArray::Int(a.mapv(|i| i as i64) * b),
            (NDArray::Bool(a), NDArray::Float(b)) => NDArray::Float(a.mapv(|i| i as i64 as f64) * b),
            (NDArray::Bool(a), NDArray::Bool(b)) => NDArray::Bool(a & b),
        }
    }

    pub fn div(&self, other: &NDArray) -> Self {
        match (self, other) {
            (NDArray::Int(a), NDArray::Int(b)) => NDArray::Int(a / b),
            (NDArray::Int(a), NDArray::Float(b)) => NDArray::Float(a.mapv(|i| i as f64) / b),
            (NDArray::Int(a), NDArray::Bool(b)) => NDArray::Int(a / b.mapv(|i| i as i64)),
            (NDArray::Float(a), NDArray::Int(b)) => NDArray::Float(a / b.mapv(|i| i as f64)),
            (NDArray::Float(a), NDArray::Float(b)) => NDArray::Float(a / b),
            (NDArray::Float(a), NDArray::Bool(b)) => NDArray::Float(a / b.mapv(|i| i as i64 as f64)),
            (NDArray::Bool(a), NDArray::Int(b)) => NDArray::Int(a.mapv(|i| i as i64) / b),
            (NDArray::Bool(a), NDArray::Float(b)) => NDArray::Float(a.mapv(|i| i as i64 as f64) / b),
            (NDArray::Bool(_), NDArray::Bool(_)) => rynd_error!("Unable to divide two boolean arrays"),
        }
    }

    pub fn eq(&self, other: &NDArray) -> Self {
        match (self, other) {
            (NDArray::Int(a), NDArray::Int(b)) => NDArray::Bool(arr_zip!(a, b, a == b)),
            (NDArray::Int(a), NDArray::Float(b)) => NDArray::Bool(arr_zip!(a, b, *a as f64 == *b)),
            (NDArray::Int(a), NDArray::Bool(b)) => NDArray::Bool(arr_zip!(a, b, *a == *b as i64)),
            (NDArray::Float(a), NDArray::Int(b)) => NDArray::Bool(arr_zip!(a, b, *a == *b as f64)),
            (NDArray::Float(a), NDArray::Float(b)) => NDArray::Bool(arr_zip!(a, b, a == b)),
            (NDArray::Float(a), NDArray::Bool(b)) => NDArray::Bool(arr_zip!(a, b, *a == *b as i64 as f64)),
            (NDArray::Bool(a), NDArray::Int(b)) => NDArray::Bool(arr_zip!(a, b, *a as i64 == *b)),
            (NDArray::Bool(a), NDArray::Float(b)) => NDArray::Bool(arr_zip!(a, b, *a as i64 as f64 == *b)),
            (NDArray::Bool(a), NDArray::Bool(b)) => NDArray::Bool(arr_zip!(a, b, a == b)),
        }
    }

    pub fn neq(&self, other: &NDArray) -> Self {
        match (self, other) {
            (NDArray::Int(a), NDArray::Int(b)) => NDArray::Bool(arr_zip!(a, b, a != b)),
            (NDArray::Int(a), NDArray::Float(b)) => NDArray::Bool(arr_zip!(a, b, *a as f64 != *b)),
            (NDArray::Int(a), NDArray::Bool(b)) => NDArray::Bool(arr_zip!(a, b, *a != *b as i64)),
            (NDArray::Float(a), NDArray::Int(b)) => NDArray::Bool(arr_zip!(a, b, *a != *b as f64)),
            (NDArray::Float(a), NDArray::Float(b)) => NDArray::Bool(arr_zip!(a, b, a != b)),
            (NDArray::Float(a), NDArray::Bool(b)) => NDArray::Bool(arr_zip!(a, b, *a != *b as i64 as f64)),
            (NDArray::Bool(a), NDArray::Int(b)) => NDArray::Bool(arr_zip!(a, b, *a as i64 != *b)),
            (NDArray::Bool(a), NDArray::Float(b)) => NDArray::Bool(arr_zip!(a, b, *a as i64 as f64 != *b)),
            (NDArray::Bool(a), NDArray::Bool(b)) => NDArray::Bool(arr_zip!(a, b, a != b)),
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