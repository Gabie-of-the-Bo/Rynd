use std::{io::Write, os::raw::c_void};

use array::NDArray;
use ndarray::Array1;
use rynaffi::{ryna_ffi_function, FFIArgs, FFIReturn};

mod array;
mod error;

// Types
fn ptr_to_ref<'a>(ptr: *const c_void) -> &'a NDArray {
    unsafe { &*(ptr as *const NDArray) }
}

// Memory management
ryna_ffi_function!(create_array(args, out) {
    let tp = args[0].as_i64() as usize;
    let num_dims = args[1].as_i64() as usize;
    let shape = args[2..2 + num_dims].into_iter().map(|i| i.as_i64() as usize).collect::<Vec<_>>();

    let array = match tp.try_into() {
        Ok(t) => Box::new(NDArray::new(t, shape)),
        Err(_) => rynd_error!("Invalid array type {tp}"),
    };

    unsafe { *out = (Box::leak(array) as *const NDArray as *const c_void).into(); }
});

ryna_ffi_function!(copy_array(args, out) {
    let a = ptr_to_ref(args[0].as_ptr());

    let res = Box::new(a.clone());

    unsafe { *out = (Box::leak(res) as *const NDArray as *const c_void).into(); }
});

ryna_ffi_function!(free_array(args, _out) {
    let ptr = args[0].as_ptr() as *mut NDArray;
    
    unsafe { std::ptr::drop_in_place(ptr); }
});

// Operators
ryna_ffi_function!(sum_arrays(args, out) {
    let a = ptr_to_ref(args[0].as_ptr());
    let b = ptr_to_ref(args[1].as_ptr());

    let res = Box::new(a.sum(b));

    unsafe { *out = (Box::leak(res) as *const NDArray as *const c_void).into(); }
});

ryna_ffi_function!(sub_arrays(args, out) {
    let a = ptr_to_ref(args[0].as_ptr());
    let b = ptr_to_ref(args[1].as_ptr());

    let res = Box::new(a.sub(b));

    unsafe { *out = (Box::leak(res) as *const NDArray as *const c_void).into(); }
});

ryna_ffi_function!(mul_arrays(args, out) {
    let a = ptr_to_ref(args[0].as_ptr());
    let b = ptr_to_ref(args[1].as_ptr());

    let res = Box::new(a.mul(b));

    unsafe { *out = (Box::leak(res) as *const NDArray as *const c_void).into(); }
});

ryna_ffi_function!(div_arrays(args, out) {
    let a = ptr_to_ref(args[0].as_ptr());
    let b = ptr_to_ref(args[1].as_ptr());

    let res = Box::new(a.div(b));

    unsafe { *out = (Box::leak(res) as *const NDArray as *const c_void).into(); }
});

ryna_ffi_function!(eq_arrays(args, out) {
    let a = ptr_to_ref(args[0].as_ptr());
    let b = ptr_to_ref(args[1].as_ptr());

    let res = Box::new(a.eq(b));

    unsafe { *out = (Box::leak(res) as *const NDArray as *const c_void).into(); }
});

ryna_ffi_function!(neq_arrays(args, out) {
    let a = ptr_to_ref(args[0].as_ptr());
    let b = ptr_to_ref(args[1].as_ptr());

    let res = Box::new(a.neq(b));

    unsafe { *out = (Box::leak(res) as *const NDArray as *const c_void).into(); }
});

// Common array operations
ryna_ffi_function!(iota(args, out) {
    let l = args[0].as_i64();

    let array = Box::new(NDArray::from(Array1::<i64>::from_iter(0..l).into_dyn()));

    unsafe { *out = (Box::leak(array) as *const NDArray as *const c_void).into(); }
});

ryna_ffi_function!(linspace(args, out) {
    let f = args[0].as_i64();
    let t = args[1].as_i64();
    let s = args[2].as_i64() as usize;

    let array = Box::new(NDArray::from(Array1::<i64>::from_iter((f..t).step_by(s)).into_dyn()));

    unsafe { *out = (Box::leak(array) as *const NDArray as *const c_void).into(); }
});

// Utility
ryna_ffi_function!(cast_array(args, out) {
    let arr = ptr_to_ref(args[0].as_ptr());
    let tp = args[1].as_i64() as usize;

    let array = Box::new(arr.cast(tp.try_into().unwrap()));

    unsafe { *out = (Box::leak(array) as *const NDArray as *const c_void).into(); }
});

ryna_ffi_function!(print_array(args, _out) {
    let arr = ptr_to_ref(args[0].as_ptr());

    print!("{}", arr);
    std::io::stdout().flush().unwrap();
});