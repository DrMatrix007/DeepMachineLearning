use rand::Rng;

use crate::matrix::{Matrix, ToMatrix};
struct LearningArgs {
    learning_rate: f64,
}

trait Layer<const N: usize, const M: usize>: Default {
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
trait ActivationFunction: Default {
    fn func<const P: usize, const K: usize>(d: Matrix<P, K>) -> Matrix<P, K>;
    fn der<const N: usize, const M: usize>(d: Matrix<N, M>) -> Matrix<N, M>;
    fn empty() -> Self;
}

impl<const N: usize, T: ActivationFunction> Layer<N, N> for T {
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
struct Tanh;
impl ActivationFunction for Tanh {
    fn func<const P: usize, const K: usize>(d: Matrix<P, K>) -> Matrix<P, K> {
        d.apply(|x, _| x.tanh())
    }

    fn der<const N: usize, const M: usize>(d: Matrix<N, M>) -> Matrix<N, M> {
        d.apply(|x, _| 1.0 - x.tanh().powi(2))
    }

    fn empty() -> Self {
        Self
    }
}

#[macro_export]
macro_rules! network {
    ($t:expr,$($other:expr),+) => {
        ($t,network!($($other),*))
    };
    ($t:expr)=>{($t,)};
}

#[test]
fn test() {
    let nn = network!(Dense::<5, 5>::default(),Tanh, Dense::<5, 5>::default());
    let mut n = Dense::<2,1>::default();
    let mut rng = rand::thread_rng();
    let args = &LearningArgs { learning_rate: 0.1 };
    let max = 1000000;
    let (mut a,mut b) = (rng.gen_range(0..max),rng.gen_range(0..max));
    for i in 0..1000
     {
        let error = (n.forward([[a,b]].to_matrix()) - [[a+b]].to_matrix()) / ((a.max(b) as f64).powi(2));
        
        n.learn([[a,b]].to_matrix(), error, args);
        (a,b) = (rng.gen_range(0..max),rng.gen_range(0..max));
    }
    let error = [[a+b]].to_matrix() - n.forward([[a,b]].to_matrix());
    println!("{:?} {:?}",n,error);
}
