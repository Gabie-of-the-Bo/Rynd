use ndarray::{Array1, ArrayBase, ArrayViewD, Axis, Dim, Ix2, IxDynImpl, OwnedRepr, RawArrayViewMut, Slice, Zip};

use crate::{algorithms::{argmax_axis, argmin_axis, argsort_axis, concat_axis, cumsum_axis, diff_axis, max_axis, min_axis, nonzero, reverse_axis, sort_view_axis, stack_axis}, owned::NDArrayOwned, rynd_error};

type DynRawArrayView<T> = RawArrayViewMut<T, Dim<IxDynImpl>>;
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

macro_rules! scalar_op {
    ($obj: expr, $n: ident, $reverse: ident, $scalar: expr, $op: tt, $l_op: tt) => {
        match ($obj, $reverse) {
            (NDArrayView::Int($n), false) => view!($n).mapv(|i| i $op $scalar as i64).into(),
            (NDArrayView::Float($n), false) => view!($n).mapv(|i| i $op $scalar as f64).into(),
            (NDArrayView::Bool($n), false) => view!($n).mapv(|i| i $l_op ($scalar as i64 != 0)).into(),
            (NDArrayView::Int($n), true) => view!($n).mapv(|i| ($scalar as i64) $op i).into(),
            (NDArrayView::Float($n), true) => view!($n).mapv(|i| ($scalar as f64) $op i).into(),
            (NDArrayView::Bool($n), true) => view!($n).mapv(|i| ($scalar as i64 != 0) $l_op i).into(),
        }
    };
}

macro_rules! scalar_fn {
    ($obj: expr, $n: ident, $reverse: ident, $scalar: expr, $func_f: tt, $func_i: tt, $l_op: tt) => {
        match ($obj, $reverse) {
            (NDArrayView::Int($n), false) => view!($n).mapv(|i| i.$func_i($scalar as u32)).into(),
            (NDArrayView::Float($n), false) => view!($n).mapv(|i| i.$func_f($scalar as f64)).into(),
            (NDArrayView::Bool($n), false) => view!($n).mapv(|i| i $l_op ($scalar as i64 != 0)).into(),
            (NDArrayView::Int($n), true) => view!($n).mapv(|i| ($scalar as i64).$func_i(i as u32)).into(),
            (NDArrayView::Float($n), true) => view!($n).mapv(|i| ($scalar as f64).$func_f(i)).into(),
            (NDArrayView::Bool($n), true) => view!($n).mapv(|i| ($scalar as i64 != 0) $l_op i).into(),
        }
    };
}

macro_rules! scalar_op_def {
    ($name1: ident, $name2: ident, $op: tt, $l_op: tt) => {
        pub fn $name1(&self, scalar: i64, reverse: bool) -> NDArrayOwned {
            scalar_op!(self, a, reverse, scalar, $op, $l_op)
        }
        
        pub fn $name2(&self, scalar: f64, reverse: bool) -> NDArrayOwned {
            scalar_op!(self, a, reverse, scalar, $op, $l_op)
        }
    };
}

macro_rules! match_op {
    ($obj: expr, $n: ident, $op: expr) => {
        match $obj {
            NDArrayView::Int($n) => $op,
            NDArrayView::Float($n) => $op,
            NDArrayView::Bool($n) => $op,
        }
    };
}

macro_rules! view {
    ($raw: expr) => {
        &unsafe { $raw.clone().deref_into_view() }  
    };
}

macro_rules! view_mut {
    ($raw: expr) => {
        unsafe { $raw.clone().deref_into_view_mut() }  
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
            (NDArrayView::Int($aa), NDArrayView::Float($bb)) => NDArrayOwned::from(arr_zip!($aa, $bb, (*$aa as f64) $op *$bb)),
            (NDArrayView::Int($aa), NDArrayView::Bool($bb)) => NDArrayOwned::from(arr_zip!($aa, $bb, *$aa $op *$bb as i64)),
            (NDArrayView::Float($aa), NDArrayView::Int($bb)) => NDArrayOwned::from(arr_zip!($aa, $bb, *$aa $op *$bb as f64)),
            (NDArrayView::Float($aa), NDArrayView::Float($bb)) => NDArrayOwned::from(arr_zip!($aa, $bb, $aa $op $bb)),
            (NDArrayView::Float($aa), NDArrayView::Bool($bb)) => NDArrayOwned::from(arr_zip!($aa, $bb, *$aa $op *$bb as i64 as f64)),
            (NDArrayView::Bool($aa), NDArrayView::Int($bb)) => NDArrayOwned::from(arr_zip!($aa, $bb, (*$aa as i64) $op *$bb)),
            (NDArrayView::Bool($aa), NDArrayView::Float($bb)) => NDArrayOwned::from(arr_zip!($aa, $bb, (*$aa as i64 as f64) $op *$bb)),
            (NDArrayView::Bool($aa), NDArrayView::Bool($bb)) => NDArrayOwned::from(arr_zip!($aa, $bb, $aa $op $bb)),
        }
    };
}

macro_rules! unary_float_fn {
    ($name: ident) => {
        pub fn $name(&self) -> NDArrayOwned {
            match self {
                NDArrayView::Int(a) => view!(a).mapv(|i| i as f64).$name().into(),
                NDArrayView::Float(a) => view!(a).$name().into(),
                NDArrayView::Bool(a) => view!(a).mapv(|i| i as i64 as f64).$name().into(),
            }
        }
    };
}

impl NDArrayView {
    pub fn owned(&self) -> NDArrayOwned {
        match_op!(self, v, view!(v).to_owned().into())
    }

    pub fn assign(&self, other: &NDArrayView) {
        match (self, other) {
            (NDArrayView::Int(a), NDArrayView::Int(b)) => view_mut!(a).zip_mut_with(view!(b), |i, v| *i = *v),
            (NDArrayView::Int(a), NDArrayView::Float(b)) => view_mut!(a).zip_mut_with(view!(b), |i, v| *i = *v as i64),
            (NDArrayView::Int(a), NDArrayView::Bool(b)) => view_mut!(a).zip_mut_with(view!(b), |i, v| *i = *v as i64),
            (NDArrayView::Float(a), NDArrayView::Int(b)) => view_mut!(a).zip_mut_with(view!(b), |i, v| *i = *v as f64),
            (NDArrayView::Float(a), NDArrayView::Float(b)) => view_mut!(a).zip_mut_with(view!(b), |i, v| *i = *v),
            (NDArrayView::Float(a), NDArrayView::Bool(b)) => view_mut!(a).zip_mut_with(view!(b), |i, v| *i = *v as i64 as f64),
            (NDArrayView::Bool(a), NDArrayView::Int(b)) => view_mut!(a).zip_mut_with(view!(b), |i, v| *i = *v != 0),
            (NDArrayView::Bool(a), NDArrayView::Float(b)) => view_mut!(a).zip_mut_with(view!(b), |i, v| *i = *v != 0.0),
            (NDArrayView::Bool(a), NDArrayView::Bool(b)) => view_mut!(a).zip_mut_with(view!(b), |i, v| *i = *v),
        }
    }

    pub fn assign_mask(&self, other: &NDArrayView, mask: &NDArrayView) {
        let m = match mask {
            NDArrayView::Bool(m) => m,
            _ => unreachable!()
        };

        match (self, other) {
            (NDArrayView::Int(a), NDArrayView::Int(b)) => Zip::from(view_mut!(a)).and(view!(b)).and(view!(m)).for_each(|a, v, m| if *m { *a = *v; }),
            (NDArrayView::Int(a), NDArrayView::Float(b)) => Zip::from(view_mut!(a)).and(view!(b)).and(view!(m)).for_each(|a, v, m| if *m { *a = *v as i64; }),
            (NDArrayView::Int(a), NDArrayView::Bool(b)) => Zip::from(view_mut!(a)).and(view!(b)).and(view!(m)).for_each(|a, v, m| if *m { *a = *v as i64; }),
            (NDArrayView::Float(a), NDArrayView::Int(b)) => Zip::from(view_mut!(a)).and(view!(b)).and(view!(m)).for_each(|a, v, m| if *m { *a = *v as f64; }),
            (NDArrayView::Float(a), NDArrayView::Float(b)) => Zip::from(view_mut!(a)).and(view!(b)).and(view!(m)).for_each(|a, v, m| if *m { *a = *v; }),
            (NDArrayView::Float(a), NDArrayView::Bool(b)) => Zip::from(view_mut!(a)).and(view!(b)).and(view!(m)).for_each(|a, v, m| if *m { *a = *v as i64 as f64; }),
            (NDArrayView::Bool(a), NDArrayView::Int(b)) => Zip::from(view_mut!(a)).and(view!(b)).and(view!(m)).for_each(|a, v, m| if *m { *a = *v != 0; }),
            (NDArrayView::Bool(a), NDArrayView::Float(b)) => Zip::from(view_mut!(a)).and(view!(b)).and(view!(m)).for_each(|a, v, m| if *m { *a = *v != 0.0; }),
            (NDArrayView::Bool(a), NDArrayView::Bool(b)) => Zip::from(view_mut!(a)).and(view!(b)).and(view!(m)).for_each(|a, v, m| if *m { *a = *v; }),
        }
    }

    pub fn len(&self) -> usize {
        match_op!(self, a, a.len())
    }

    pub fn shape(&self) -> &[usize] {
        match_op!(self, a, a.shape())
    }

    pub fn get_i64(&self, idx: usize) -> i64 {
        match self {
            NDArrayView::Int(a) => view!(a)[idx],
            NDArrayView::Float(a) => view!(a)[idx] as i64,
            NDArrayView::Bool(a) => view!(a)[idx] as i64,
        }
    }

    pub fn get_f64(&self, idx: usize) -> f64 {
        match self {
            NDArrayView::Int(a) => view!(a)[idx] as f64,
            NDArrayView::Float(a) => view!(a)[idx],
            NDArrayView::Bool(a) => view!(a)[idx] as i64 as f64,
        }
    }

    pub fn get_bool(&self, idx: usize) -> bool {
        match self {
            NDArrayView::Int(a) => view!(a)[idx] != 0,
            NDArrayView::Float(a) => view!(a)[idx] != 0.0,
            NDArrayView::Bool(a) => view!(a)[idx],
        }
    }

    pub fn assign_scalar_i64(&self, other: i64) {
        match self {
            NDArrayView::Int(a) => view_mut!(a).iter_mut().for_each(|i| *i = other),
            NDArrayView::Float(a) => view_mut!(a).iter_mut().for_each(|i| *i = other as f64),
            NDArrayView::Bool(a) => view_mut!(a).iter_mut().for_each(|i| *i = other != 0),
        }
    }

    pub fn assign_scalar_f64(&self, other: f64) {
        match self {
            NDArrayView::Int(a) => view_mut!(a).iter_mut().for_each(|i| *i = other as i64),
            NDArrayView::Float(a) => view_mut!(a).iter_mut().for_each(|i| *i = other),
            NDArrayView::Bool(a) => view_mut!(a).iter_mut().for_each(|i| *i = other != 0.0),
        }
    }

    pub fn assign_scalar_i64_mask(&self, other: i64, mask: &NDArrayView) {
        let m = match mask {
            NDArrayView::Bool(m) => m,
            _ => unreachable!()
        };
        
        match self {
            NDArrayView::Int(a) => view_mut!(a).zip_mut_with(view!(m), |i, v| if *v { *i = other }),
            NDArrayView::Float(a) => view_mut!(a).zip_mut_with(view!(m), |i, v| if *v { *i = other as f64 }),
            NDArrayView::Bool(a) => view_mut!(a).zip_mut_with(view!(m), |i, v| if *v { *i = other != 0 }),
        }
    }

    pub fn assign_scalar_f64_mask(&self, other: f64, mask: &NDArrayView) {
        let m = match mask {
            NDArrayView::Bool(m) => m,
            _ => unreachable!()
        };
        
        match self {
            NDArrayView::Int(a) => view_mut!(a).zip_mut_with(view!(m), |i, v| if *v { *i = other as i64 }),
            NDArrayView::Float(a) => view_mut!(a).zip_mut_with(view!(m), |i, v| if *v { *i = other }),
            NDArrayView::Bool(a) => view_mut!(a).zip_mut_with(view!(m), |i, v| if *v { *i = other != 0.0 }),
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

    pub fn modulo(&self, other: &NDArrayView) -> NDArrayOwned {
        broadcast_op_general!(self, other, _a, _b, %, rynd_error!("Unable to divide two boolean arrays"))
    }

    pub fn eq(&self, other: &NDArrayView) -> NDArrayOwned {
        zip_op!(self, other, a, b, ==)
    }

    pub fn neq(&self, other: &NDArrayView) -> NDArrayOwned {
        zip_op!(self, other, a, b, !=)
    }

    pub fn lt(&self, other: &NDArrayView) -> NDArrayOwned {
        zip_op!(self, other, a, b, <)
    }

    pub fn gt(&self, other: &NDArrayView) -> NDArrayOwned {
        zip_op!(self, other, a, b, >)
    }

    pub fn leq(&self, other: &NDArrayView) -> NDArrayOwned {
        zip_op!(self, other, a, b, <=)
    }

    pub fn geq(&self, other: &NDArrayView) -> NDArrayOwned {
        zip_op!(self, other, a, b, >=)
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

    scalar_op_def!(sum_scalar_i64, sum_scalar_f64, +, ^);
    scalar_op_def!(sub_scalar_i64, sub_scalar_f64, -, ^);
    scalar_op_def!(mul_scalar_i64, mul_scalar_f64, *, &);
    scalar_op_def!(eq_scalar_i64, eq_scalar_f64, ==, ==);
    scalar_op_def!(neq_scalar_i64, neq_scalar_f64, !=, !=);
    scalar_op_def!(lt_scalar_i64, lt_scalar_f64, <, <);
    scalar_op_def!(gt_scalar_i64, gt_scalar_f64, >, >);
    scalar_op_def!(leq_scalar_i64, leq_scalar_f64, <=, <=);
    scalar_op_def!(geq_scalar_i64, geq_scalar_f64, >=, >=);

    pub fn div_scalar_i64(&self, scalar: i64, reverse: bool) -> NDArrayOwned {
        if matches!(self, NDArrayView::Bool(_)) {
            rynd_error!("Unable to divide boolean array");
        }

        scalar_op!(self, a, reverse, scalar, /, &)
    }

    pub fn div_scalar_f64(&self, scalar: f64, reverse: bool) -> NDArrayOwned {
        if matches!(self, NDArrayView::Bool(_)) {
            rynd_error!("Unable to divide boolean array");
        }

        scalar_op!(self, a, reverse, scalar, /, &)
    }

    pub fn mod_scalar_i64(&self, scalar: i64, reverse: bool) -> NDArrayOwned {
        if matches!(self, NDArrayView::Bool(_)) {
            rynd_error!("Unable to divide boolean array");
        }

        scalar_op!(self, a, reverse, scalar, %, &)
    }

    pub fn mod_scalar_f64(&self, scalar: f64, reverse: bool) -> NDArrayOwned {
        if matches!(self, NDArrayView::Bool(_)) {
            rynd_error!("Unable to divide boolean array");
        }

        scalar_op!(self, a, reverse, scalar, %, &)
    }

    pub fn pow_scalar_i64(&self, scalar: i64, reverse: bool) -> NDArrayOwned {
        if matches!(self, NDArrayView::Bool(_)) {
            rynd_error!("Unable to calculate the power of a boolean array");
        }

        scalar_fn!(self, a, reverse, scalar, powf, pow, &)
    }

    pub fn pow_scalar_f64(&self, scalar: f64, reverse: bool) -> NDArrayOwned {
        if matches!(self, NDArrayView::Bool(_)) {
            rynd_error!("Unable to calculate the power of a boolean array");
        }

        scalar_fn!(self, a, reverse, scalar, powf, pow, &)
    }

    fn mask<T>(v: &DynRawArrayView<T>, mask: &DynRawArrayView<bool>) -> NDArrayOwned 
        where NDArrayOwned: From<ArrayBase<OwnedRepr<T>, Dim<IxDynImpl>>>,
              T: Clone { 
        if v.shape() != mask.shape() {
            rynd_error!("Array with shape {:?} cannot mask array with shape {:?}", mask.shape(), v.shape())
        }

        Array1::<T>::from_iter(
            view!(v).iter()
                    .zip(view!(mask))
                    .filter_map(|(v, m)| {
                        if *m {
                            Some(v.clone())
                        } else {
                            None
                        }
                    })
        ).into_dyn().into()
    }

    fn index_view_i64(dim: usize, mut index: i64) -> usize {
        while index < 0 {
            index += dim as i64;
        }

        if index as u64 > usize::MAX as u64 {
            rynd_error!("Invalid array index: {}", index);
        }

        index as usize
    }

    fn index_view<T: Clone>(v: &ArrayViewD<T>, index: &[i64]) -> T {
        // Prepare indexes
        let mapped_idx = index.iter()
                              .zip(v.shape())
                              .map(|(i, d)| Self::index_view_i64(*d, *i))
                              .collect::<Vec<_>>();

        v[mapped_idx.as_slice()].clone()
    }

    fn array_index<T>(v: &DynRawArrayView<T>, index: &DynRawArrayView<i64>) -> NDArrayOwned 
        where NDArrayOwned: From<ArrayBase<OwnedRepr<T>, Dim<IxDynImpl>>>,
              T: Clone { 
        if !Self::valid_index(v, index) {
            rynd_error!("Array with shape {:?} cannot index array with shape {:?}", index.shape(), v.shape())
        }
        
        let fv = view!(v);

        if index.shape().len() == 2 {
            Array1::<T>::from_iter(
                view!(index).lanes(Axis(index.shape().len() - 1)).into_iter()
                            .map(|i| Self::index_view(fv, i.as_slice().unwrap()))
            ).into_dyn().into()

        } else {
            Array1::<T>::from_iter(
                view!(index).iter().map(|i| Self::index_view(fv, &[*i]))
            ).into_dyn().into()
        }
    }

    fn valid_index<T>(v: &DynRawArrayView<T>, index: &DynRawArrayView<i64>) -> bool {
        (
            v.shape().len() == *index.shape().last().unwrap() ||
            (v.shape().len() == 1 && index.shape().len() == 1)
        ) && index.shape().len() <= 2
    }

    pub fn index(&self, other: &NDArrayView) -> NDArrayOwned {
        match (self, other) {
            (NDArrayView::Int(a), NDArrayView::Int(b)) => Self::array_index(a, b),
            (NDArrayView::Float(a), NDArrayView::Int(b)) => Self::array_index(a, b),
            (NDArrayView::Bool(a), NDArrayView::Int(b)) => Self::array_index(a, b),
            (NDArrayView::Int(a), NDArrayView::Bool(b)) => Self::mask(a, b),
            (NDArrayView::Float(a), NDArrayView::Bool(b)) => Self::mask(a, b),
            (NDArrayView::Bool(a), NDArrayView::Bool(b)) => Self::mask(a, b),

            (_, NDArrayView::Float(_)) => rynd_error!("Unable to use float as an index"),
        }
    }

    pub fn reshape(&mut self, shape: Vec<usize>) -> NDArrayView {
        match_op!(self, a, view_mut!(a).into_shape_with_order(shape).unwrap().raw_view_mut().into())
    }

    pub fn slice(&mut self, slices: Vec<Slice>) -> NDArrayView {
        match_op!(self, a, view_mut!(a).slice_each_axis_mut(|ax| {
            slices.get(ax.axis.index()).cloned().unwrap_or_else(|| Slice::new(0, None, 1))
        }).raw_view_mut().into())
    }

    pub fn permute(&self, permutation: &[usize]) -> NDArrayView {
        match self {
            NDArrayView::Int(a) => view_mut!(a).permuted_axes(permutation).raw_view_mut().into(),
            NDArrayView::Float(a) => view_mut!(a).permuted_axes(permutation).raw_view_mut().into(),
            NDArrayView::Bool(a) => view_mut!(a).permuted_axes(permutation).raw_view_mut().into(),
        }
    }

    pub fn matmul(&self, other: &NDArrayView) -> NDArrayOwned {
        match self {
            NDArrayView::Int(a) => {
                let a_d2 = a.clone().into_dimensionality::<Ix2>().ok().unwrap();

                match other {
                    NDArrayView::Int(b) => {
                        let b_d2 = b.clone().into_dimensionality::<Ix2>().ok().unwrap();
                        view!(a_d2).dot(view!(b_d2)).into_dyn().into()
                    },

                    NDArrayView::Float(b) => {
                        let b_d2 = b.clone().into_dimensionality::<Ix2>().ok().unwrap();
                        let a_float = view!(a_d2).mapv(|i| i as f64);
                        a_float.dot(view!(b_d2)).into_dyn().into()
                    },

                    NDArrayView::Bool(b) => {
                        let b_d2 = b.clone().into_dimensionality::<Ix2>().ok().unwrap();
                        let b_int = view!(b_d2).mapv(|i| i as i64);
                        view!(a_d2).dot(&b_int).into_dyn().into()
                    },
                }
            },

            NDArrayView::Float(a) => {
                let a_d2 = a.clone().into_dimensionality::<Ix2>().ok().unwrap();

                match other {
                    NDArrayView::Int(b) => {
                        let b_d2 = b.clone().into_dimensionality::<Ix2>().ok().unwrap();
                        let b_float = view!(b_d2).mapv(|i| i as f64);
                        view!(a_d2).dot(&b_float).into_dyn().into()
                    },

                    NDArrayView::Float(b) => {
                        let b_d2 = b.clone().into_dimensionality::<Ix2>().ok().unwrap();
                        view!(a_d2).dot(view!(b_d2)).into_dyn().into()
                    },

                    NDArrayView::Bool(b) => {
                        let b_d2 = b.clone().into_dimensionality::<Ix2>().ok().unwrap();
                        let b_float = view!(b_d2).mapv(|i| i as i64 as f64);
                        view!(a_d2).dot(&b_float).into_dyn().into()
                    },
                }
            },
            NDArrayView::Bool(a) => {
                let a_d2 = a.clone().into_dimensionality::<Ix2>().ok().unwrap();

                match other {
                    NDArrayView::Int(b) => {
                        let b_d2 = b.clone().into_dimensionality::<Ix2>().ok().unwrap();
                        let a_int = view!(a_d2).mapv(|i| i as i64);
                        a_int.dot(view!(b_d2)).into_dyn().into()
                    },

                    NDArrayView::Float(b) => {
                        let b_d2 = b.clone().into_dimensionality::<Ix2>().ok().unwrap();
                        let a_float = view!(a_d2).mapv(|i| i as i64 as f64);
                        a_float.dot(view!(b_d2)).into_dyn().into()
                    },

                    NDArrayView::Bool(b) => {
                        let b_d2 = b.clone().into_dimensionality::<Ix2>().ok().unwrap();
                        let a_int = view!(a_d2).mapv(|i| i as i64);
                        let b_int = view!(b_d2).mapv(|i| i as i64);
                        a_int.dot(&b_int).into_dyn().into()
                    },
                }
            },
        }
    }

    pub fn axis_reverse(&mut self, axis: usize) -> NDArrayView {
        match self {
            NDArrayView::Int(a) => reverse_axis(&mut view_mut!(a), Axis(axis)).raw_view_mut().into(),
            NDArrayView::Float(a) => reverse_axis(&mut view_mut!(a), Axis(axis)).raw_view_mut().into(),
            NDArrayView::Bool(a) => reverse_axis(&mut view_mut!(a), Axis(axis)).raw_view_mut().into()
        }    
    }

    pub fn axis_sum(&self, axis: usize) -> NDArrayOwned {
        match self {
            NDArrayView::Int(a) => view!(a).sum_axis(Axis(axis)).into(),
            NDArrayView::Float(a) => view!(a).sum_axis(Axis(axis)).into(),
            NDArrayView::Bool(a) => view!(a).mapv(|i| i as i64).sum_axis(Axis(axis)).into(),
        }    
    }

    pub fn axis_sort(&self, axis: usize) {
        match self {
            NDArrayView::Int(a) => sort_view_axis(view_mut!(a), Axis(axis)),
            NDArrayView::Float(a) => sort_view_axis(view_mut!(a), Axis(axis)),
            NDArrayView::Bool(a) => sort_view_axis(view_mut!(a), Axis(axis))
        }
    }

    pub fn axis_argsort(&self, axis: usize) -> NDArrayOwned {
        match self {
            NDArrayView::Int(a) => argsort_axis(view!(a), Axis(axis)).into(),
            NDArrayView::Float(a) => argsort_axis(view!(a), Axis(axis)).into(),
            NDArrayView::Bool(a) => argsort_axis(&view!(a).mapv(|i| i as i64).view(), Axis(axis)).into(),
        }
    }

    pub fn axis_mean(&self, axis: usize) -> NDArrayOwned {
        match self {
            NDArrayView::Int(a) => view!(a).mean_axis(Axis(axis)).unwrap().into(),
            NDArrayView::Float(a) => view!(a).mean_axis(Axis(axis)).unwrap().into(),
            NDArrayView::Bool(a) => view!(a).mapv(|i| i as i64).mean_axis(Axis(axis)).unwrap().into(),
        }
    }

    pub fn axis_var(&self, axis: usize) -> NDArrayOwned {
        match self {
            NDArrayView::Int(a) => view!(a).mapv(|i| i as f64).var_axis(Axis(axis), 0.0).into(),
            NDArrayView::Float(a) => view!(a).var_axis(Axis(axis), 0.0).into(),
            NDArrayView::Bool(a) => view!(a).mapv(|i| i as i64 as f64).var_axis(Axis(axis), 0.0).into(),
        }
    }

    pub fn axis_std(&self, axis: usize) -> NDArrayOwned {
        match self {
            NDArrayView::Int(a) => view!(a).mapv(|i| i as f64).std_axis(Axis(axis), 0.0).into(),
            NDArrayView::Float(a) => view!(a).std_axis(Axis(axis), 0.0).into(),
            NDArrayView::Bool(a) => view!(a).mapv(|i| i as i64 as f64).std_axis(Axis(axis), 0.0).into(),
        }
    }

    pub fn axis_diff(&self, axis: usize) -> NDArrayOwned {
        match self {
            NDArrayView::Int(a) => diff_axis(view!(a), Axis(axis)).into(),
            NDArrayView::Float(a) => diff_axis(view!(a), Axis(axis)).into(),
            NDArrayView::Bool(a) => diff_axis(&view!(a).mapv(|i| i as i64).view(), Axis(axis)).into(),
        }
    }

    pub fn axis_cumsum(&self, axis: usize) -> NDArrayOwned {
        match self {
            NDArrayView::Int(a) => cumsum_axis(view!(a), Axis(axis)).into(),
            NDArrayView::Float(a) => cumsum_axis(view!(a), Axis(axis)).into(),
            NDArrayView::Bool(a) => cumsum_axis(&view!(a).mapv(|i| i as i64).view(), Axis(axis)).into(),
        }
    }

    pub fn axis_min(&self, axis: usize) -> NDArrayOwned {
        match self {
            NDArrayView::Int(a) => min_axis(view!(a), Axis(axis)).into(),
            NDArrayView::Float(a) => min_axis(view!(a), Axis(axis)).into(),
            NDArrayView::Bool(a) => min_axis(&view!(a).mapv(|i| i as i64).view(), Axis(axis)).into(),
        }
    }

    pub fn axis_max(&self, axis: usize) -> NDArrayOwned {
        match self {
            NDArrayView::Int(a) => max_axis(view!(a), Axis(axis)).into(),
            NDArrayView::Float(a) => max_axis(view!(a), Axis(axis)).into(),
            NDArrayView::Bool(a) => max_axis(&view!(a).mapv(|i| i as i64).view(), Axis(axis)).into(),
        }
    }

    pub fn axis_argmin(&self, axis: usize) -> NDArrayOwned {
        match self {
            NDArrayView::Int(a) => argmin_axis(view!(a), Axis(axis)).into(),
            NDArrayView::Float(a) => argmin_axis(view!(a), Axis(axis)).into(),
            NDArrayView::Bool(a) => argmin_axis(&view!(a).mapv(|i| i as i64).view(), Axis(axis)).into(),
        }
    }

    pub fn axis_argmax(&self, axis: usize) -> NDArrayOwned {
        match self {
            NDArrayView::Int(a) => argmax_axis(view!(a), Axis(axis)).into(),
            NDArrayView::Float(a) => argmax_axis(view!(a), Axis(axis)).into(),
            NDArrayView::Bool(a) => argmax_axis(&view!(a).mapv(|i| i as i64).view(), Axis(axis)).into(),
        }
    }

    pub fn stack(&self, other: &NDArrayView, axis: usize) -> NDArrayOwned {
        match (self, other) {
            (NDArrayView::Int(a), NDArrayView::Int(b)) => stack_axis(view!(a), view!(b), Axis(axis)).into(),
            (NDArrayView::Float(a), NDArrayView::Float(b)) => stack_axis(view!(a), view!(b), Axis(axis)).into(),
            (NDArrayView::Bool(a), NDArrayView::Bool(b)) => stack_axis(view!(a), view!(b), Axis(axis)).into(),
            _ => rynd_error!("Unable to stack: incompatible array types")
        }
    }

    pub fn concat(&self, other: &NDArrayView, axis: usize) -> NDArrayOwned {
        match (self, other) {
            (NDArrayView::Int(a), NDArrayView::Int(b)) => concat_axis(view!(a), view!(b), Axis(axis)).into(),
            (NDArrayView::Float(a), NDArrayView::Float(b)) => concat_axis(view!(a), view!(b), Axis(axis)).into(),
            (NDArrayView::Bool(a), NDArrayView::Bool(b)) => concat_axis(view!(a), view!(b), Axis(axis)).into(),
            _ => rynd_error!("Unable to concat: incompatible array types")
        }
    }

    pub fn floor(&self) -> NDArrayOwned {
        match self {
            NDArrayView::Int(a) => view!(a).clone().into_owned().into(),
            NDArrayView::Float(a) => view!(a).floor().into(),
            NDArrayView::Bool(a) => view!(a).clone().into_owned().into(),
        }
    }

    pub fn ceil(&self) -> NDArrayOwned {
        match self {
            NDArrayView::Int(a) => view!(a).clone().into_owned().into(),
            NDArrayView::Float(a) => view!(a).ceil().into(),
            NDArrayView::Bool(a) => view!(a).clone().into_owned().into(),
        }
    }

    pub fn round(&self) -> NDArrayOwned {
        match self {
            NDArrayView::Int(a) => view!(a).clone().into_owned().into(),
            NDArrayView::Float(a) => view!(a).round().into(),
            NDArrayView::Bool(a) => view!(a).clone().into_owned().into(),
        }
    }

    pub fn nonzero(&self) -> NDArrayOwned {
        match self {
            NDArrayView::Int(a) => nonzero(view!(a)).into_dyn().into(),
            NDArrayView::Float(a) => nonzero(view!(a)).into_dyn().into(),
            NDArrayView::Bool(a) => nonzero(&view!(a).mapv(|i| i as i64).view()).into_dyn().into(),
        }
    }

    pub fn clip(&self, low: f64, high: f64) -> NDArrayOwned {
        match self {
            NDArrayView::Int(a) => view!(a).clamp(low as i64, high as i64).into(),
            NDArrayView::Float(a) => view!(a).clamp(low, high).into(),
            NDArrayView::Bool(_) => rynd_error!("Unable to clip a boolean array"),
        }
    }

    unary_float_fn!(cos);
    unary_float_fn!(sin);
    unary_float_fn!(tan);
    unary_float_fn!(sqrt);
    unary_float_fn!(log2);
    unary_float_fn!(log10);
}

impl std::fmt::Display for NDArrayView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match_op!(self, a, write!(f, "{}", view!(a)))
    }
}