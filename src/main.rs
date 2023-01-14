#![allow(dead_code, unused_imports)]

use std::default::{self};

use layer::{Activation, Network};
use matrix::Matrix;
use rand::random;

use crate::layer::{DenseLayer, LearningArgs, LinearActivation, ReLU, SigmoidActivation};
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
        DenseLayer::<2, 1000>::default(),
        DenseLayer::<1000,1>::default(),
    );

    for _ in 0..1000 {
        let (j, i) = (random::<f64>()/2.0, random::<f64>()/2.0);
        let x = Matrix::from([[i], [j]]);
        let err = net.calulate_error(x.clone(), Matrix::from([[i+j]]));
        net.fit(x, err.clone(), &args);
    }
    println!("{:?}", net);

    println!("{}", net.forward([[0.6], [0.2]].into()));
}
