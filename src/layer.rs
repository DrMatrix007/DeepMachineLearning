use std::fmt::Debug;

use crate::matrix::Matrix;

pub struct LearningArgs {
    pub learning_rate: f64,
}

#[derive(Debug)]
pub struct Wrapper<const N: usize, const M: usize, T: Layer<N, M>>(pub T);

pub trait Layer<const N: usize, const M: usize> {
    fn forward<const K: usize>(&self, l: Matrix<K, N>) -> Matrix<K, M>;
    fn backward<const K: usize>(&self, l: Matrix<K, M>) -> Matrix<K, N>;

    fn learn(
        &mut self,
        input: Matrix<1, N>,
        output_error: Matrix<1, M>,
        args: &LearningArgs,
    ) -> Matrix<1, N>;
}
pub trait Activation: Debug {
    fn forward<const K: usize, const N: usize>(&self, l: Matrix<K, N>) -> Matrix<K, N>;
    fn backward<const K: usize, const N: usize>(&self, l: Matrix<K, N>) -> Matrix<K, N>;
}
impl<const N: usize, T: Activation> Layer<N, N> for T {
    fn forward<const K: usize>(&self, l: Matrix<K, N>) -> Matrix<K, N> {
        self.forward(l)
    }

    fn backward<const K: usize>(&self, l: Matrix<K, N>) -> Matrix<K, N> {
        self.backward(l)
    }

    fn learn(
        &mut self,
        input: Matrix<1, N>,
        output_error: Matrix<1, N>,
        _: &LearningArgs,
    ) -> Matrix<1, N> {
        self.backward(input).element_wise_product(&output_error)
    }
}

#[derive(Debug)]
pub struct DenseLayer<const N: usize, const M: usize> {
    weights: Matrix<N, M>,
}

impl<const N: usize, const M: usize> Layer<N, M> for DenseLayer<N, M> {
    fn forward<const K: usize>(&self, l: Matrix<K, N>) -> Matrix<K, M> {
        l * &self.weights
    }

    fn backward<const K: usize>(&self, l: Matrix<K, M>) -> Matrix<K, N> {
        l * self.weights.trasnpose()
    }

    fn learn(
        &mut self,
        input: Matrix<1, N>,
        output_error: Matrix<1, M>,
        args: &LearningArgs,
    ) -> Matrix<1, N> {
        let input_error = &output_error * self.weights.trasnpose();
        let weights_error = input.trasnpose() * &output_error;
        self.weights = &self.weights - &weights_error * args.learning_rate;
        input_error
    }
}

impl<const N: usize, const M: usize> Default for DenseLayer<N, M> {
    fn default() -> Self {
        Self {
            weights: Matrix::generate(|| ((rand::random::<f64>()) - 0.5) / (N * M) as f64),
        }
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
pub struct ReLU;

impl Activation for ReLU {
    fn forward<const K: usize, const N: usize>(&self, l: Matrix<K, N>) -> Matrix<K, N> {
        l.map(|_, v| if v > 0.0 { v } else { 0.0 })
    }

    fn backward<const K: usize, const N: usize>(&self, l: Matrix<K, N>) -> Matrix<K, N> {
        l.map(|_, v| if v > 0.0 { 1.0 } else { 0.0 })
    }
}

#[derive(Debug)]
pub struct Tanh;

impl Activation for Tanh {
    fn forward<const K: usize, const N: usize>(&self, l: Matrix<K, N>) -> Matrix<K, N> {
         l.map(|_,v|v.tanh())
    }

    fn backward<const K: usize, const N: usize>(&self, l: Matrix<K, N>) -> Matrix<K, N> {
        l.map(|_,v|1.0-v.tanh().powi(2))
    }
}


pub trait Network<const N: usize, const M: usize> {
    fn forward<const K: usize>(&self, l: Matrix<K, N>) -> Matrix<K, M>;
    fn backward<const K: usize>(&self, l: Matrix<K, M>) -> Matrix<K, N>;

    fn fit(
        &mut self,
        input: Matrix<1, N>,
        output_error: Matrix<1, M>,
        args: &LearningArgs,
    ) -> Matrix<1, N>;

    fn calulate_error<const K: usize>(&self, x: Matrix<K, N>, y: Matrix<K, M>) -> Matrix<K, M> {
        let loss = self.forward(x) - y;
        2.0  * loss
    }
}

impl<const N: usize, const M: usize, A: Layer<N, M>> Network<N, M> for (A,) {
    fn forward<const K: usize>(&self, l: Matrix<K, N>) -> Matrix<K, M> {
        Layer::forward(&self.0, l)
    }

    fn backward<const K: usize>(&self, l: Matrix<K, M>) -> Matrix<K, N> {
        Layer::backward(&self.0, l)
    }

    fn fit(
        &mut self,
        input: Matrix<1, N>,
        output_error: Matrix<1, M>,
        args: &LearningArgs,
    ) -> Matrix<1, N> {
        self.0.learn(input, output_error, args)
    }
}

impl<const N: usize, const M: usize, const K: usize, A: Layer<N, K>, B: Network<K, M>> Network<N, M>
    for (Wrapper<N, K, A>, B)
{
    fn forward<const P: usize>(&self, l: Matrix<P, N>) -> Matrix<P, M> {
        self.1.forward(self.0 .0.forward(l))
    }

    fn backward<const P: usize>(&self, l: Matrix<P, M>) -> Matrix<P, N> {
        self.0 .0.backward(self.1.backward(l))
    }

    fn fit(
        &mut self,
        input: Matrix<1, N>,
        output_error: Matrix<1, M>,
        args: &LearningArgs,
    ) -> Matrix<1, N> {
        self.0 .0.learn(
            input.clone(),
            self.1.fit(self.0 .0.forward(input), output_error, args),
            args,
        )
    }
}

#[macro_export]
macro_rules! network {
    ($e:expr $(,)?) => {
        ($e,)
    };
    ($e:expr,$($es:expr),* $(,)?) => {
        ($crate::layer::Wrapper($e),network!($($es),*))
    };

}
