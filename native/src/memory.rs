use std::{os::raw::c_void, sync::Mutex};

use lazy_static::lazy_static;
use rustc_hash::FxHashMap;

use crate::{array::NDArray, rynd_error};

lazy_static! {
    static ref REFCOUNTS: Mutex<FxHashMap<usize, usize>> = Mutex::default();
}

pub fn ptr_to_ref<'a>(ptr: *const c_void) -> &'a NDArray {
    if get_refcount(ptr).is_none() { // Avoid dereferencing an invalid array
        rynd_error!("Tried to use deleted array (perhaps you need to clone an array)");
    }

    unsafe { &*(ptr as *const NDArray) }
}

pub fn register_ref(ptr: *const c_void) {
    REFCOUNTS.lock().unwrap()
             .entry(ptr as usize)
             .and_modify(|i| *i += 1)
             .or_insert(1);
}

pub fn remove_ref(ptr: *const c_void) {
    REFCOUNTS.lock().unwrap()
             .entry(ptr as usize)
             .and_modify(|i| *i -= 1);
}

pub fn get_refcount(ptr: *const c_void) -> Option<usize> {
    REFCOUNTS.lock().unwrap()
             .get(&(ptr as usize))
             .cloned()
}

pub fn free_array_ptr(ptr: *const c_void) {
    remove_ref(ptr);

    if let Some(0) = get_refcount(ptr) { // Avoid double free
        REFCOUNTS.lock().unwrap().remove(&(ptr as usize));

        unsafe { std::ptr::drop_in_place(ptr as *mut NDArray) };
    }
}

pub fn register_and_leak(obj: Box<NDArray>) -> *const c_void {
    let ptr = Box::leak(obj) as *const NDArray as *const c_void;

    register_ref(ptr);

    ptr
}