use std::fmt::Debug;

use crate::{layer::{Layer, LearningArgs}, matrix::Matrix};

pub trait Activation: Debug {
    fn forward<const K: usize, const N: usize>(&self, l: Matrix<K, N>) -> Matrix<K, N>;
    fn backward<const K: usize, const N: usize>(&self, l: Matrix<K, N>) -> Matrix<K, N>;
}
impl<const N: usize, T: Activation> Layer<N, N> for T {
    fn learn(
        &mut self,
        input: Matrix<N, 1>,
        output_error: Matrix<N, 1>,
        _: &LearningArgs,
    ) -> Matrix<N, 1> {
        self.backward(input).element_wise_product(&output_error)
    }

    fn forward<const K: usize>(&self, l: Matrix<N, K>) -> Matrix<N, K> {
        self.forward(l)
    }
}

#[derive(Debug)]
pub struct LinearActivation;

impl Activation for LinearActivation {
    fn forward<const K: usize, const N: usize>(&self, l: Matrix<K, N>) -> Matrix<K, N> {
        l
    }

    fn backward<const K: usize, const N: usize>(&self, _: Matrix<K, N>) -> Matrix<K, N> {
        Matrix::ones()
    }
}
#[derive(Debug)]
pub struct SigmoidActivation;

impl SigmoidActivation {
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
    fn der(x: f64) -> f64 {
        let x = Self::sigmoid(x);
        x * (1.0 - x)
    }
}
impl Activation for SigmoidActivation {
    fn forward<const K: usize, const N: usize>(&self, l: Matrix<K, N>) -> Matrix<K, N> {
        l.map(|_, v| Self::sigmoid(v))
    }

    fn backward<const K: usize, const N: usize>(&self, l: Matrix<K, N>) -> Matrix<K, N> {
        l.map(|_, v| Self::der(v))
    }
}
#[derive(Debug)]
pub struct ReLUActivation;

impl Activation for ReLUActivation {
    fn forward<const K: usize, const N: usize>(&self, l: Matrix<K, N>) -> Matrix<K, N> {
        l.map(|_, v| if v >= 0.0 { v } else { 0.0 })
    }

    fn backward<const K: usize, const N: usize>(&self, l: Matrix<K, N>) -> Matrix<K, N> {
        l.map(|_, v| if v > 0.0 { 1.0 } else { 0.0 })
    }
}


#[derive(Debug)]
pub struct LeakyReLUActivation;

impl Activation for LeakyReLUActivation {
    fn forward<const K: usize, const N: usize>(&self, l: Matrix<K, N>) -> Matrix<K, N> {
        l.map(|_, v| if v >= 0.0 { v } else { 0.1*v })
    }

    fn backward<const K: usize, const N: usize>(&self, l: Matrix<K, N>) -> Matrix<K, N> {
        l.map(|_, v| if v > 0.0 { 1.0 } else { 0.1 })
    }
}




#[derive(Debug)]
pub struct TanhActivation;

impl Activation for TanhActivation {
    fn forward<const K: usize, const N: usize>(&self, l: Matrix<K, N>) -> Matrix<K, N> {
        l.map(|_, v| v.tanh())
    }

    fn backward<const K: usize, const N: usize>(&self, l: Matrix<K, N>) -> Matrix<K, N> {
        l.map(|_, v| 1.0 - v.tanh().powi(2))
    }
}
