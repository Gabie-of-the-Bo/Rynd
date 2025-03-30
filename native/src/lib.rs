use std::{io::Write, os::raw::c_void};

use array::NDArray;
use error::{rynd_dims_check, rynd_matmul_check, rynd_normalize_dim, rynd_permute_check, rynd_slice_check};
use memory::{free_array_ptr, ptr_to_ref, register_and_leak, register_view};
use ndarray::{Array1, Slice};
use owned::{NDArrayOwned, NDArrayType};
use rynaffi::{ryna_ffi_function, FFIArgs, FFIReturn};

mod owned;
mod view;
mod array;
mod error;
mod memory;
mod algorithms;

// Memory management
ryna_ffi_function!(malloc(args, out) {
    let size = args[0].as_i64() as usize;
    let layout = std::alloc::Layout::array::<i64>(size).expect("Invalid layout");
    let ptr = unsafe { std::alloc::alloc(layout) };

    unsafe { *out = (ptr as *const c_void).into(); }
});

ryna_ffi_function!(free(args, _out) {
    let ptr = args[0].as_ptr() as *mut u8;
    let size = args[1].as_i64() as usize;
    let layout = std::alloc::Layout::array::<i64>(size).expect("Invalid layout");

    unsafe { std::alloc::dealloc(ptr, layout) };
});

ryna_ffi_function!(array_from_ptr(args, out) {
    let ptr = args[0].as_ptr();
    let tp = args[1].as_i64() as usize;
    let num_dims = args[2].as_i64() as usize;
    let shape = args[3..3 + num_dims].into_iter().map(|i| i.as_i64() as usize).collect::<Vec<_>>();

    let array = match tp.try_into() {
        Ok(t) => Box::new(NDArray::from_ptr(t, shape, ptr)),
        Err(_) => rynd_error!("Invalid array type {tp}"),
    };

    unsafe { *out = register_and_leak(array).into(); }
});

ryna_ffi_function!(create_array(args, out) {
    let tp = args[0].as_i64() as usize;
    let num_dims = args[1].as_i64() as usize;
    let shape = args[2..2 + num_dims].into_iter().map(|i| i.as_i64() as usize).collect::<Vec<_>>();

    let array = match tp.try_into() {
        Ok(t) => Box::new(NDArray::new(t, shape)),
        Err(_) => rynd_error!("Invalid array type {tp}"),
    };

    unsafe { *out = register_and_leak(array).into(); }
});

ryna_ffi_function!(copy_array(args, out) {
    let a = ptr_to_ref(args[0].as_ptr());

    let res = Box::new(a.clone());

    unsafe { *out = register_and_leak(res).into(); }
});

ryna_ffi_function!(free_array(args, _out) {
    let ptr = args[0].as_ptr();
    
    free_array_ptr(ptr);
});

// Operators
macro_rules! binop_rynd_ffi {
    ($function: ident, $name: ident) => {
        ryna_ffi_function!($function(args, out) {
            let a = ptr_to_ref(args[0].as_ptr());
            let b = ptr_to_ref(args[1].as_ptr());
        
            let res = Box::new(a.$name(b));
        
            unsafe { *out = register_and_leak(res).into(); }
        });                
    };
}

binop_rynd_ffi!(sum_arrays, sum);
binop_rynd_ffi!(sub_arrays, sub);
binop_rynd_ffi!(mul_arrays, mul);
binop_rynd_ffi!(div_arrays, div);
binop_rynd_ffi!(mod_arrays, modulo);
binop_rynd_ffi!(pow_arrays, pow);
binop_rynd_ffi!(eq_arrays, eq);
binop_rynd_ffi!(neq_arrays, neq);
binop_rynd_ffi!(lt_arrays, lt);
binop_rynd_ffi!(gt_arrays, gt);
binop_rynd_ffi!(leq_arrays, leq);
binop_rynd_ffi!(geq_arrays, geq);
binop_rynd_ffi!(index_arrays, index);

ryna_ffi_function!(assign_arrays(args, _out) {
    let a = ptr_to_ref(args[0].as_ptr());
    let b = ptr_to_ref(args[1].as_ptr());

    if a.shape() != b.shape() {
        rynd_error!("Unable to assign array of shape {:?} to array of shape {:?}", b.shape(), a.shape());
    }

    a.assign(b);
});   

ryna_ffi_function!(assign_arrays_mask(args, _out) {
    let a = ptr_to_ref(args[0].as_ptr());
    let b = ptr_to_ref(args[1].as_ptr());
    let m = ptr_to_ref(args[2].as_ptr());

    if a.shape() != b.shape() {
        rynd_error!("Unable to assign array of shape {:?} to array of shape {:?}", b.shape(), a.shape());
    }

    if a.shape() != m.shape() {
        rynd_error!("Mask shape ({:?}) does not match target array shape ({:?})", m.shape(), a.shape());
    }

    if !m.is_mask() {
        rynd_error!("Given mask array is not a boolean mask");
    }

    a.assign_mask(b, m);
});   

macro_rules! binop_rynd_scalar_ffi {
    ($function: ident, $name_int: ident, $name_float: ident) => {
        ryna_ffi_function!($function(args, out) {
            use rynaffi::FFIValue;

            let a = ptr_to_ref(args[0].as_ptr());
            
            let res = match args[1] {
                FFIValue::Int(v) => a.$name_int(v, args[2].as_i64() != 0),
                FFIValue::Float(v) => a.$name_float(v, args[2].as_i64() != 0),
                _ => unreachable!()
            };
        
            unsafe { *out = register_and_leak(Box::new(res)).into(); }
        });                
    };
}

binop_rynd_scalar_ffi!(sum_array_scalar, sum_scalar_i64, sum_scalar_f64);
binop_rynd_scalar_ffi!(sub_array_scalar, sub_scalar_i64, sub_scalar_f64);
binop_rynd_scalar_ffi!(mul_array_scalar, mul_scalar_i64, mul_scalar_f64);
binop_rynd_scalar_ffi!(div_array_scalar, div_scalar_i64, div_scalar_f64);
binop_rynd_scalar_ffi!(mod_array_scalar, mod_scalar_i64, mod_scalar_f64);
binop_rynd_scalar_ffi!(pow_array_scalar, pow_scalar_i64, pow_scalar_f64);
binop_rynd_scalar_ffi!(eq_array_scalar, eq_scalar_i64, eq_scalar_f64);
binop_rynd_scalar_ffi!(neq_array_scalar, neq_scalar_i64, neq_scalar_f64);
binop_rynd_scalar_ffi!(lt_array_scalar, lt_scalar_i64, lt_scalar_f64);
binop_rynd_scalar_ffi!(gt_array_scalar, gt_scalar_i64, gt_scalar_f64);
binop_rynd_scalar_ffi!(leq_array_scalar, leq_scalar_i64, leq_scalar_f64);
binop_rynd_scalar_ffi!(geq_array_scalar, geq_scalar_i64, geq_scalar_f64);

ryna_ffi_function!(len(args, out) {
    let a = ptr_to_ref(args[0].as_ptr());

    unsafe { *out = (a.len() as i64).into(); }
});

ryna_ffi_function!(shape(args, out) {
    let a = ptr_to_ref(args[0].as_ptr());

    let res = NDArrayOwned::from(Array1::from_iter(a.shape().iter().map(|i| *i as i64)).into_dyn()).into();

    unsafe { *out = register_and_leak(Box::new(res)).into(); }
});

ryna_ffi_function!(assign_array_scalar(args, _out) {
    use rynaffi::FFIValue;

    let a = ptr_to_ref(args[0].as_ptr());
    
    match args[1] {
        FFIValue::Int(v) => a.assign_scalar_i64(v),
        FFIValue::Float(v) => a.assign_scalar_f64(v),
        _ => unreachable!()
    };
});    

ryna_ffi_function!(assign_array_scalar_mask(args, _out) {
    use rynaffi::FFIValue;

    let a = ptr_to_ref(args[0].as_ptr());
    let m = ptr_to_ref(args[2].as_ptr());

    if a.shape() != m.shape() {
        rynd_error!("Mask shape ({:?}) does not match target array shape ({:?})", m.shape(), a.shape());
    }

    if !m.is_mask() {
        rynd_error!("Given mask array is not a boolean mask");
    }

    match args[1] {
        FFIValue::Int(v) => a.assign_scalar_i64_mask(v, m),
        FFIValue::Float(v) => a.assign_scalar_f64_mask(v, m),
        _ => unreachable!()
    };
});     

ryna_ffi_function!(get_elem(args, out) {
    let a = ptr_to_ref(args[0].as_ptr());
    let idx = args[1].as_i64() as usize;
    let tp = args[2].as_i64() as usize;
    
    match tp.try_into() {
        Ok(t) => {
            match t {
                NDArrayType::Int => unsafe { *out = a.get_i64(idx).into()},
                NDArrayType::Float => unsafe { *out = a.get_f64(idx).into()},
                NDArrayType::Bool => unsafe { *out = (a.get_bool(idx) as i64).into()}
            }
        },
        Err(_) => rynd_error!("Invalid array type {tp}"),
    };
});     

// Common array operations
ryna_ffi_function!(iota(args, out) {
    let l = args[0].as_i64();

    let array = Box::new(NDArrayOwned::iota(l).into());

    unsafe { *out = register_and_leak(array).into(); }
});

ryna_ffi_function!(linspace(args, out) {
    let f = args[0].as_i64();
    let t = args[1].as_i64();
    let s = args[2].as_i64() as usize;

    let array = Box::new(NDArrayOwned::linspace(f, t, s).into());

    unsafe { *out = register_and_leak(array).into(); }
});

ryna_ffi_function!(reshape_array(args, out) {
    let arr_ptr = args[0].as_ptr();
    let arr = ptr_to_ref(arr_ptr);
    let num_dims = args[1].as_i64() as usize;
    let shape = args[2..2 + num_dims].into_iter().map(|i| i.as_i64() as usize).collect::<Vec<_>>();

    let res = Box::new(arr.reshape(shape));
    let view_ptr = register_and_leak(res);

    register_view(arr_ptr, view_ptr);

    unsafe { *out = view_ptr.into(); }
});

ryna_ffi_function!(slice_array(args, out) {
    let arr_ptr = args[0].as_ptr();
    let arr = ptr_to_ref(arr_ptr);
    let num_dims = args[1].as_i64() as usize;

    rynd_dims_check(arr, Some(num_dims), None);

    let slices = args[2..2 + num_dims * 3].into_iter()
        .map(|i| i.as_i64() as isize)
        .collect::<Vec<_>>()
        .chunks_exact(3)
        .zip(arr.shape().iter().enumerate())
        .map(|(s, (d_idx, d))| {
            rynd_slice_check(arr, s[0], s[1], s[2], d_idx);

            let start = if s[0] >= 0 { s[0] } else { 1 + *d as isize + s[0] };
            let end = if s[1] >= 0 { s[1] } else { 1 + *d as isize + s[1] };
            let step = s[2];

            Slice::new(start, Some(end), step)
        })
        .collect::<Vec<_>>();

    let res = Box::new(arr.slice(slices));
    let view_ptr = register_and_leak(res);

    register_view(arr_ptr, view_ptr);

    unsafe { *out = view_ptr.into(); }
});

ryna_ffi_function!(permute_axes(args, out) {
    let arr_ptr = args[0].as_ptr();
    let num_dims = args[1].as_i64() as usize;
    let perm = args[2..2 + num_dims].into_iter()
                                    .map(|i| i.as_i64() as usize)
                                    .collect::<Vec<_>>();
    
    let a = ptr_to_ref(arr_ptr);

    rynd_permute_check(a, &perm);

    let res = Box::new(a.permute(&perm));
    let view_ptr = register_and_leak(res);

    register_view(arr_ptr, view_ptr);

    unsafe { *out = view_ptr.into(); }
});

ryna_ffi_function!(matmul(args, out) {
    let a_ptr = args[0].as_ptr();
    let b_ptr = args[1].as_ptr();
    let a = ptr_to_ref(a_ptr);
    let b = ptr_to_ref(b_ptr);

    rynd_matmul_check(a, b);

    let array = Box::new(a.matmul(b));

    unsafe { *out = register_and_leak(array).into(); }
});

ryna_ffi_function!(rand_array(args, out) {
    let num_dims = args[0].as_i64() as usize;
    let shape = args[1..1 + num_dims].into_iter().map(|i| i.as_i64() as usize).collect::<Vec<_>>();

    let array = Box::new(NDArrayOwned::rand(shape).into());

    unsafe { *out = register_and_leak(array).into(); }
});

ryna_ffi_function!(normal_array(args, out) {
    let mean = args[0].as_f64();
    let std = args[1].as_f64();
    let num_dims = args[2].as_i64() as usize;
    let shape = args[3..3 + num_dims].into_iter().map(|i| i.as_i64() as usize).collect::<Vec<_>>();

    let array = Box::new(NDArrayOwned::normal(mean, std, shape).into());

    unsafe { *out = register_and_leak(array).into(); }
});

ryna_ffi_function!(stack_arrays(args, out) {
    let a_ptr = args[0].as_ptr();
    let b_ptr = args[1].as_ptr();
    let dim = args[2].as_i64();
    let a = ptr_to_ref(a_ptr);
    let b = ptr_to_ref(b_ptr);

    if dim < 0 {
        rynd_error!("Negative dimensions are not allowed in stack function ({} given)", dim);
    }

    if dim as usize > a.shape().len() || dim as usize > b.shape().len() {
        rynd_error!("Dimension {} out of range for stack function", dim);
    }

    let array = Box::new(a.stack(b, dim as usize));

    unsafe { *out = register_and_leak(array).into(); }
});

ryna_ffi_function!(concat_arrays(args, out) {
    let a_ptr = args[0].as_ptr();
    let b_ptr = args[1].as_ptr();
    let mut dim = args[2].as_i64();
    let a = ptr_to_ref(a_ptr);
    let b = ptr_to_ref(b_ptr);

    rynd_normalize_dim(a, &mut dim);
    rynd_normalize_dim(b, &mut dim);

    let array = Box::new(a.concat(b, dim as usize));

    unsafe { *out = register_and_leak(array).into(); }
});

// Unary functions
macro_rules! unary_rynd_fn {
    ($public_name: ident, $name: ident) => {
        ryna_ffi_function!($public_name(args, out) {
            let arr = ptr_to_ref(args[0].as_ptr());
            
            let array = Box::new(arr.$name());
        
            unsafe { *out = register_and_leak(array).into(); }
        });        
    };
}

unary_rynd_fn!(floor_array, floor);
unary_rynd_fn!(ceil_array, ceil);
unary_rynd_fn!(round_array, round);
unary_rynd_fn!(nonzero_array, nonzero);
unary_rynd_fn!(cos_array, cos);
unary_rynd_fn!(sin_array, sin);
unary_rynd_fn!(tan_array, tan);
unary_rynd_fn!(sqrt_array, sqrt);
unary_rynd_fn!(log2_array, log2);
unary_rynd_fn!(log10_array, log10);

ryna_ffi_function!(clip_array(args, out) {
    let arr = ptr_to_ref(args[0].as_ptr());
    let low = args[1].as_f64();
    let high = args[2].as_f64();
    
    let array = Box::new(arr.clip(low, high));

    unsafe { *out = register_and_leak(array).into(); }
});

// Axis functions
macro_rules! axis_rynd_fn {
    ($public_name: ident, $name: ident) => {
        ryna_ffi_function!($public_name(args, out) {
            let arr = ptr_to_ref(args[0].as_ptr());
            let mut dim = args[1].as_i64();
            
            rynd_normalize_dim(arr, &mut dim);
        
            let array = Box::new(arr.$name(dim as usize));
        
            unsafe { *out = register_and_leak(array).into(); }
        });
    };
}

axis_rynd_fn!(axis_sum_array, axis_sum);
axis_rynd_fn!(axis_mean_array, axis_mean);
axis_rynd_fn!(axis_var_array, axis_var);
axis_rynd_fn!(axis_std_array, axis_std);
axis_rynd_fn!(axis_argsort_array, axis_argsort);
axis_rynd_fn!(axis_diff_array, axis_diff);
axis_rynd_fn!(axis_cumsum_array, axis_cumsum);
axis_rynd_fn!(axis_min_array, axis_min);
axis_rynd_fn!(axis_max_array, axis_max);
axis_rynd_fn!(axis_argmin_array, axis_argmin);
axis_rynd_fn!(axis_argmax_array, axis_argmax);

ryna_ffi_function!(axis_sort_array(args, _out) {
    let arr = ptr_to_ref(args[0].as_ptr());
    let mut dim = args[1].as_i64();
    
    rynd_normalize_dim(arr, &mut dim);

    arr.axis_sort(dim as usize);
});

ryna_ffi_function!(axis_reverse_array(args, out) {
    let arr_ptr = args[0].as_ptr();
    let arr = ptr_to_ref(arr_ptr);
    let mut dim = args[1].as_i64();
    
    rynd_normalize_dim(arr, &mut dim);

    let res = Box::new(arr.axis_reverse(dim as usize));
    let view_ptr = register_and_leak(res);

    register_view(arr_ptr, view_ptr);

    unsafe { *out = view_ptr.into(); }
});

// Utility
ryna_ffi_function!(cast_array(args, out) {
    let arr = ptr_to_ref(args[0].as_ptr());
    let tp = args[1].as_i64() as usize;

    let array = Box::new(arr.cast(tp.try_into().unwrap()));

    unsafe { *out = register_and_leak(array).into(); }
});

ryna_ffi_function!(print_array(args, _out) {
    let arr = ptr_to_ref(args[0].as_ptr());

    print!("{}", arr);
    std::io::stdout().flush().unwrap();
});