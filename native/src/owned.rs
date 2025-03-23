use ndarray::{Array1, ArrayD};
use rand::Rng;
use rand_distr::{Distribution, Normal};

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

macro_rules! match_op {
    ($obj: expr, $n: ident, $op: expr) => {
        match $obj {
            NDArrayOwned::Int($n) => $op,
            NDArrayOwned::Float($n) => $op,
            NDArrayOwned::Bool($n) => $op,
        }
    };
}

impl NDArrayOwned {
    pub fn new(tp: NDArrayType, shape: Vec<usize>) -> Self {
        match tp {
            NDArrayType::Int => NDArrayOwned::Int(ArrayD::default(shape)),
            NDArrayType::Float => NDArrayOwned::Float(ArrayD::default(shape)),
            NDArrayType::Bool => NDArrayOwned::Bool(ArrayD::default(shape)),
        }
    }

    pub fn len(&self) -> usize {
        match_op!(self, a, a.len())
    }

    pub fn shape(&self) -> &[usize] {
        match_op!(self, a, a.shape())
    }

    pub fn iota(l: i64) -> Self {
        NDArrayOwned::from(Array1::<i64>::from_iter(0..l).into_dyn())
    }

    pub fn rand(shape: Vec<usize>) -> Self {
        let mut rng = rand::rng();
        let mut result = ArrayD::<f64>::zeros(shape);

        result.mapv_inplace(|_| rng.random());

        result.into()
    }

    pub fn normal(mean: f64, std: f64, shape: Vec<usize>) -> Self {
        let mut rng = rand::rng();
        let normal = Normal::new(mean, std).unwrap();
        let mut result = ArrayD::<f64>::zeros(shape);

        result.mapv_inplace(|_| normal.sample(&mut rng));

        result.into()
    }

    pub fn linspace(f: i64, t: i64, s: usize) -> Self {
        NDArrayOwned::from(Array1::<i64>::from_iter((f..t).step_by(s)).into_dyn())
    }

    pub fn view(&mut self) -> NDArrayView {
        match_op!(self, a, a.raw_view_mut().into())
    }

    pub fn cast(&mut self, tp: NDArrayType) -> Self {
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

    pub fn reshape(&mut self, shape: Vec<usize>) -> NDArrayView {
        match_op!(self, a, a.view_mut().into_shape_with_order(shape).unwrap().raw_view_mut().into())
    }
}

impl std::fmt::Display for NDArrayOwned {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match_op!(self, a, write!(f, "{a}"))
    }
}