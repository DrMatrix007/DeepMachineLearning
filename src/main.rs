#![allow(dead_code)]

use layer::Forwardable;

use crate::layer::DenseLayer;
pub mod layer;
pub mod matrix;
fn main() {
    let l = network!(DenseLayer::<1, 5>::default(), DenseLayer::<5, 2>::default());
    println!("hello world, {}", l.forward([[1]].into()));
}
