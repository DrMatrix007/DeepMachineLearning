#![allow(dead_code, unused_imports)]

use std::default::{self};

use layer::{Activation, Network};
use matrix::Matrix;
use rand::random;

use crate::layer::{DenseLayer, LearningArgs, LinearActivation, ReLU, SigmoidActivation, Tanh};
pub mod layer;
pub mod matrix;

#[derive(Debug)]
struct SquareActivation;

impl Activation for SquareActivation {
    fn forward<const K: usize, const N: usize>(&self, l: Matrix<K, N>) -> Matrix<K, N> {
        l.element_wise_product(&l)
    }

    fn backward<const K: usize, const N: usize>(&self, l: Matrix<K, N>) -> Matrix<K, N> {
        l * 2.0
    }
}

fn main() {
    let args = LearningArgs { learning_rate: 0.1 };
    let mut net = network!(
        DenseLayer::<2, 3>::default(),
        Tanh,
        DenseLayer::<3, 1>::default(),
        Tanh,
    );
    for _ in 0..10000 {
        for i in 0..2 {
            for j in 0..2 {
                // let (j, i) = (random::<f64>() / 2.0, random::<f64>() / 2.0);
                let x = Matrix::from([[i], [j]]);
                let err =
                    net.calulate_error(x.clone(), Matrix::from([[if i == j { 1.0 } else { 0.5 }]]));
                net.fit(x, err.clone(), &args);
            }
        }
    }
    println!("{:?}", net);

    println!("{}", net.forward([[1], [0]].into()));
    println!("{}", net.forward([[0], [1]].into()));
    println!("{}", net.forward([[1], [1]].into()));
    println!("{}", net.forward([[0], [0]].into()));
}
