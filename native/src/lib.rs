use std::{io::Write, os::raw::c_void};

use array::NDArray;
use memory::{free_array_ptr, ptr_to_ref, register_and_leak, register_view};
use ndarray::Slice;
use owned::NDArrayOwned;
use rynaffi::{ryna_ffi_function, FFIArgs, FFIReturn};

mod owned;
mod view;
mod array;
mod error;
mod memory;

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
binop_rynd_ffi!(pow_arrays, pow);
binop_rynd_ffi!(eq_arrays, eq);
binop_rynd_ffi!(neq_arrays, neq);
binop_rynd_ffi!(index_arrays, index);

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
binop_rynd_scalar_ffi!(pow_array_scalar, pow_scalar_i64, pow_scalar_f64);
binop_rynd_scalar_ffi!(eq_array_scalar, eq_scalar_i64, eq_scalar_f64);
binop_rynd_scalar_ffi!(neq_array_scalar, neq_scalar_i64, neq_scalar_f64);

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
    let slices = args[2..2 + num_dims * 3].into_iter()
        .map(|i| i.as_i64() as isize)
        .collect::<Vec<_>>()
        .chunks_exact(3)
        .map(|s| Slice::new(s[0], if s[1] == -1 { None } else { Some(s[1] + 1) }, s[2]))
        .collect::<Vec<_>>();

    let res = Box::new(arr.slice(slices));
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