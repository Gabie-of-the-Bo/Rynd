use std::{io::Write, os::raw::c_void};

use array::NDArray;
use rynaffi::{ryna_ffi_function, FFIArgs, FFIReturn};

mod array;

// Types
fn ptr_to_ref<'a>(ptr: *const c_void) -> &'a NDArray {
    unsafe { &*(ptr as *const NDArray) }
}

// Memory management
ryna_ffi_function!(create_array(args, out) {
    let tp = args[0].as_i64() as usize;
    let num_dims = args[1].as_i64() as usize;
    let shape = args[2..2 + num_dims].into_iter().map(|i| i.as_i64() as usize).collect::<Vec<_>>();
    let array = Box::new(NDArray::new(tp.try_into().unwrap(), shape));

    unsafe { *out = (Box::leak(array) as *const NDArray as *const c_void).into(); }
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