#![allow(dead_code)]

use std::io::LineWriter;

use layer::{Activation, Network};
use matrix::Matrix;

use crate::layer::{DenseLayer, LearningArgs, LinearActivation};
pub mod layer;
pub mod matrix;

#[derive(Debug)]
struct SquareActivation;

impl Activation for SquareActivation {
    fn forward<const K: usize, const N: usize>(&self, l: Matrix<K, N>) -> Matrix<K, N> {
        l.element_wise_product(&l)
    }

    fn backward<const K: usize, const N: usize>(&self, l: Matrix<K, N>) -> Matrix<K, N> {
        l*2.0
    }
}


fn main() {
    let args = LearningArgs {
        learning_rate: 0.01,
    };
    let mut net = network!(DenseLayer::<1, 1>::default(),DenseLayer::<1,1>::default());

    for _ in 0..1 {
        for i in 0..10 {
            let err =  net.forward([[i]].into()) - Matrix::from([[i*2]]);
            println!("{} {}",&err, net.fit([[i]].into(), err.clone(), &args));
        }   
    }
    println!("{:?}",net);
    println!("{}", net.forward([[1]].into()));
}
