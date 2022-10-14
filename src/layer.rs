use rand::Rng;

use crate::matrix::{Matrix, ToMatrix};
pub struct LearningArgs {
    learning_rate: f64,
}

pub trait Layer<const N: usize, const M: usize>: Default + Sized {
    fn forward<const DATA: usize>(&self, l: Matrix<DATA, N>) -> Matrix<DATA, M>;
    fn learn(
        &mut self,
        input: Matrix<1, N>,
        output_error: Matrix<1, M>,
        args: &LearningArgs,
    ) -> Matrix<1, N>;
}

#[derive(Debug)]
pub struct Dense<const N: usize, const M: usize> {
    weights: Matrix<N, M>,
    biases: Matrix<1, M>,
}
impl<const N: usize, const M: usize> Default for Dense<N, M> {
    fn default() -> Self {
        Self {
            weights: Default::default(),
            biases: Default::default(),
        }
    }
}
impl<const N: usize, const M: usize> Layer<N, M> for Dense<N, M> {
    fn forward<const DATA: usize>(&self, l: Matrix<DATA, N>) -> Matrix<DATA, M> {
        (&l * &self.weights).add_vec(&self.biases)
    }

    fn learn(
        &mut self,
        input: Matrix<1, N>,
        output_error: Matrix<1, M>,
        args: &LearningArgs,
    ) -> Matrix<1, N> {
        let input_error = &output_error * &self.weights.transpose();
        let weights_error = &input.transpose() * &output_error;
        self.weights = &self.weights - &(&weights_error * args.learning_rate);
        self.biases = &self.biases - &(&output_error * args.learning_rate);
        input_error
    }
}
trait ActivationFunction<const N: usize>: Default {
    fn func<const P: usize, const K: usize>(d: Matrix<P, K>) -> Matrix<P, K>;
    fn der<const Q: usize, const M: usize>(d: Matrix<Q, M>) -> Matrix<Q, M>;
    fn empty() -> Self;
}

impl<const N: usize, T: ActivationFunction<N>> Layer<N, N> for T {
    fn forward<const DATA: usize>(&self, l: Matrix<DATA, N>) -> Matrix<DATA, N> {
        Self::func(l)
    }

    fn learn(
        &mut self,
        input: Matrix<1, N>,
        output_error: Matrix<1, N>,
        _: &LearningArgs,
    ) -> Matrix<1, N> {
        Self::der(input).element_wise_product(&output_error)
    }
}

#[derive(Default)]
struct Tanh<const N: usize>;
impl<const N: usize> ActivationFunction<N> for Tanh<N> {
    fn func<const P: usize, const K: usize>(d: Matrix<P, K>) -> Matrix<P, K> {
        d.apply(|x| x.tanh())
    }

    fn der<const Q: usize, const M: usize>(d: Matrix<Q, M>) -> Matrix<Q, M> {
        d.apply(|x| 1.0 - x.tanh().powi(2))
    }

    fn empty() -> Self {
        Self
    }
}

pub struct Wrapper<const N: usize, const M: usize, A: Layer<N, M>>(A);
pub trait Predictable<const INPUT: usize, const FINAL: usize> {
    fn predict<const DATA: usize>(&self, x: Matrix<DATA, INPUT>) -> Matrix<DATA, FINAL>;
}
impl<const N: usize, const M: usize, const FINAL: usize, A, B> Predictable<N, FINAL>
    for (Wrapper<N, M, A>, B)
where
    A: Layer<N, M>,
    B: Predictable<M, FINAL>,
{
    fn predict<const DATA: usize>(&self, x: Matrix<DATA, N>) -> Matrix<DATA, FINAL> {
        let a = self.0 .0.forward(x);
        self.1.predict(a)
    }
}
impl<const M: usize, const FINAL: usize, B> Predictable<M, FINAL> for (Wrapper<M, FINAL, B>,)
where
    B: Layer<M, FINAL>,
{
    fn predict<const DATA: usize>(&self, x: Matrix<DATA, M>) -> Matrix<DATA, FINAL> {
        self.0 .0.forward(x)
    }
}

trait Learnable<const N:usize,const M:usize,const FINAL:usize> {
    fn fit(
        &mut self,
        input: Matrix<1, N>,
        output_error: Matrix<1, M>,
        args: &LearningArgs,
    ) -> Matrix<1, N>;
}
impl<const N: usize, const M: usize, const FINAL: usize, A, B> Learnable<N,M, FINAL>
    for (Wrapper<N, M, A>, B)
where
    A: Layer<N, M>,
    B: Learnable<N,M, FINAL>,
{
    fn fit(
        &mut self,
        input: Matrix<1, N>,
        output_error: Matrix<1, M>,
        args: &LearningArgs,
    ) -> Matrix<1, N> {
        self.1.0.0. self.0.0.learn(input,output_error,args)
    }
}
impl<const M: usize, const FINAL: usize, B> Learnable<M, FINAL> for (Wrapper<M, FINAL, B>,)
where
    B: Layer<M, FINAL>,
{
    fn fit(
        &mut self,
        input: Matrix<1, M>,
        output_error: Matrix<1, M>,
        _: &LearningArgs,
    ) -> Matrix<1, M> {
        todo!()
    }
}


#[macro_export]
macro_rules! network {
    ($t:expr,$($other:expr),*) => {
        network_layers!($t,$($other),*)
    };
}
macro_rules! wrap_layer {
    ($t:expr) => {
        $crate::layer::Wrapper($t)
    };
}
macro_rules! network_layers {
    ($t:expr,$($other:expr),+) => {
        (wrap_layer!($t),network_layers!($($other),+))
    };
    ($t:expr)=>{
        (wrap_layer!($t),)
    };
    ()=> {()};
}

#[test]
fn test() {
    let n = network!(Dense::<2, 1>::default(), Dense::<1, 1>::default());
    let mut rng = rand::thread_rng();
    let args = &LearningArgs { learning_rate: 0.1 };
    let max = 1000000;
    let (mut a, mut b) = (rng.gen_range(0..max), rng.gen_range(0..max));
    for _ in 0..100000 {
        let error = (n.predict([[a, b]].to_matrix()) - [[a + b]].to_matrix())
            / ((a.max(b).max(1) as f64).powi(2));

        n.learn([[a, b]].to_matrix(), error, args);
        (a, b) = (rng.gen_range(0..max), rng.gen_range(0..max));
    }
    let error = [[a + b]].to_matrix() - n.predict([[a, b]].to_matrix());
    if !error.max().unwrap_or(0.0).is_finite() {
        panic!();
    }
    println!("{:?} {:?}", n, error);
}
