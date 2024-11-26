use ndarray::{Array1, ArrayD};

use crate::view::NDArrayView;

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
pub enum NDArrayOwned {
    Int(ArrayD<i64>),
    Float(ArrayD<f64>),
    Bool(ArrayD<bool>),
}

impl From<ArrayD<i64>> for NDArrayOwned {
    fn from(value: ArrayD<i64>) -> Self {
        Self::Int(value)
    }
}

impl From<ArrayD<f64>> for NDArrayOwned {
    fn from(value: ArrayD<f64>) -> Self {
        Self::Float(value)
    }
}

impl From<ArrayD<bool>> for NDArrayOwned {
    fn from(value: ArrayD<bool>) -> Self {
        Self::Bool(value)
    }
}

impl NDArrayOwned {
    pub fn new(tp: NDArrayType, shape: Vec<usize>) -> Self {
        match tp {
            NDArrayType::Int => NDArrayOwned::Int(ArrayD::default(shape)),
            NDArrayType::Float => NDArrayOwned::Float(ArrayD::default(shape)),
            NDArrayType::Bool => NDArrayOwned::Bool(ArrayD::default(shape)),
        }
    }

    pub fn iota(l: i64) -> Self {
        NDArrayOwned::from(Array1::<i64>::from_iter(0..l).into_dyn())
    }

    pub fn linspace(f: i64, t: i64, s: usize) -> Self {
        NDArrayOwned::from(Array1::<i64>::from_iter((f..t).step_by(s)).into_dyn())
    }

    pub fn view(&self) -> NDArrayView {
        match self {
            NDArrayOwned::Int(a) => NDArrayView::Int(a.raw_view()),
            NDArrayOwned::Float(a) => NDArrayView::Float(a.raw_view()),
            NDArrayOwned::Bool(a) => NDArrayView::Bool(a.raw_view()),
        }
    }

    pub fn cast(&self, tp: NDArrayType) -> Self {
        match (tp, self) {
            (NDArrayType::Int, NDArrayOwned::Int(array)) => NDArrayOwned::from(array.clone()),
            (NDArrayType::Int, NDArrayOwned::Float(array)) => NDArrayOwned::from(array.mapv(|i| i as i64)),
            (NDArrayType::Int, NDArrayOwned::Bool(array)) => NDArrayOwned::from(array.mapv(|i| i as i64)),
            (NDArrayType::Float, NDArrayOwned::Int(array)) => NDArrayOwned::from(array.mapv(|i| i as f64)),
            (NDArrayType::Float, NDArrayOwned::Float(array)) => NDArrayOwned::from(array.clone()),
            (NDArrayType::Float, NDArrayOwned::Bool(array)) => NDArrayOwned::from(array.mapv(|i| i as i64 as f64)),
            (NDArrayType::Bool, NDArrayOwned::Int(array)) => NDArrayOwned::from(array.mapv(|i| i != 0)),
            (NDArrayType::Bool, NDArrayOwned::Float(array)) => NDArrayOwned::from(array.mapv(|i| i != 0.0)),
            (NDArrayType::Bool, NDArrayOwned::Bool(array)) => NDArrayOwned::from(array.clone()),
        }
    }
}

impl std::fmt::Display for NDArrayOwned {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NDArrayOwned::Int(array) => write!(f, "{array}"),
            NDArrayOwned::Float(array) => write!(f, "{array}"),
            NDArrayOwned::Bool(array) => write!(f, "{array}"),
        }
    }
}