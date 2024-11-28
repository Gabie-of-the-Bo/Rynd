use std::{os::raw::c_void, sync::Mutex};

use lazy_static::lazy_static;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{array::NDArray, rynd_error};

lazy_static! {
    static ref REFCOUNTS: Mutex<FxHashMap<usize, usize>> = Mutex::default();
    static ref DEPS_VIEW_ARR: Mutex<FxHashMap<usize, usize>> = Mutex::default();
    static ref DEPS_ARR_VIEW: Mutex<FxHashMap<usize, FxHashSet<usize>>> = Mutex::default();
}

pub fn ptr_to_ref<'a>(ptr: *const c_void) -> &'a mut NDArray {
    if get_refcount(ptr).is_none() { // Avoid dereferencing an invalid array
        rynd_error!("Tried to use deleted array (perhaps you need to clone an array)");
    }

    unsafe { &mut *(ptr as *mut NDArray) }
}

pub fn register_view(arr: *const c_void, view: *const c_void) {
    // Get the array pointer to which the view is pointing to
    let arr_ptr = match ptr_to_ref(arr) {
        NDArray::Owned(_) => arr,
        NDArray::View(_) => *DEPS_VIEW_ARR.lock().unwrap().get(&(arr as usize)).unwrap() as *const c_void,
    };

    DEPS_VIEW_ARR.lock().unwrap()
                .entry(view as usize)
                .or_insert(arr_ptr as usize);

    DEPS_ARR_VIEW.lock().unwrap()
                .entry(arr_ptr as usize)
                .or_default()
                .insert(view as usize);
}

fn array_has_view(arr: *const c_void) -> bool {
    DEPS_ARR_VIEW.lock().unwrap().contains_key(&(arr as usize))
}

fn remove_view(view: *const c_void) {
    if can_remove_view(view) {
        REFCOUNTS.lock().unwrap().remove(&(view as usize));

        unsafe { std::ptr::drop_in_place(view as *mut NDArray) };

        // Remove the underlying array if needed
        match DEPS_VIEW_ARR.lock().unwrap().remove(&(view as usize)) {
            Some(arr) => { 
                DEPS_ARR_VIEW.lock().unwrap().entry(arr).or_default().remove(&(view as usize));

                if DEPS_ARR_VIEW.lock().unwrap().entry(arr).or_default().is_empty() {
                    DEPS_ARR_VIEW.lock().unwrap().remove(&(arr as usize));
                }
    
                remove_array(arr as *const c_void);
            },
            None => { },
        }
    }
}

fn remove_array(arr: *const c_void) {
    if can_remove_array(arr) { // Avoid double free
        REFCOUNTS.lock().unwrap().remove(&(arr as usize));

        unsafe { std::ptr::drop_in_place(arr as *mut NDArray) };
    }
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

pub fn can_remove_array(ptr: *const c_void) -> bool {
    matches!(get_refcount(ptr), Some(0)) && !array_has_view(ptr)
}

pub fn can_remove_view(ptr: *const c_void) -> bool {
    matches!(get_refcount(ptr), Some(0))
}

pub fn free_array_ptr(ptr: *const c_void) {
    let arr = ptr_to_ref(ptr);

    remove_ref(ptr);

    match arr {
        NDArray::Owned(_) => remove_array(ptr),
        NDArray::View(_) => remove_view(ptr),
    }
}

pub fn register_and_leak(obj: Box<NDArray>) -> *const c_void {
    let ptr = Box::leak(obj) as *const NDArray as *const c_void;

    register_ref(ptr);

    ptr
}