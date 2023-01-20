use std::fmt::Debug;

use crate::{matrix::Matrix, optimizers::Optimizer};

pub struct LearningArgs {
    pub learning_rate: f64,
    pub epochs: i32,
    pub single_epochs: i32,
}

#[derive(Debug)]
pub struct Wrapper<const N: usize, const M: usize, T: Layer<N, M>>(pub T);

pub trait Layer<const N: usize, const M: usize> {
    fn forward<const K: usize>(&self, l: Matrix<N, K>) -> Matrix<M, K>;

    fn learn(
        &mut self,
        input: Matrix<N, 1>,
        output_error: Matrix<M, 1>,
        args: &LearningArgs,
    ) -> Matrix<N, 1>;
}

#[derive(Debug)]
pub struct DenseLayer<const N: usize, const M: usize,Args, Op:Optimizer<N,M,Args>> {
    weights: Matrix<M, N>,
    biases: Matrix<M, 1>,
    optimizer:Op,
}
impl<const N: usize, const M: usize,Args, Op:Optimizer<N,M,Args>> DenseLayer<N, M,Args,Op> {
    fn add_bias<const K: usize, const P: usize>(
        m: &Matrix<P, K>,
        b: &Matrix<P, 1>,
    ) -> Matrix<P, K> {
        m.map(|(x, _), val| val + b[(x, 0)])
    }
}

impl<const N: usize, const M: usize,Args, Op:Optimizer<N,M,Args>> Layer<N, M> for DenseLayer<N, M,Args,Op> {
    fn forward<const K: usize>(&self, l: Matrix<N, K>) -> Matrix<M, K> {
        Self::add_bias(&(l * &self.weights), &self.biases)
    }
    fn learn(
        &mut self,
        input: Matrix<N, 1>,
        output_error: Matrix<M, 1>,
        args: &LearningArgs,
    ) -> Matrix<N, 1> {
        let input_error = &output_error * self.weights.trasnpose();
        let weights_error = input.trasnpose() * &output_error;
        self.weights = &self.weights - &weights_error * args.learning_rate;
        self.biases = &self.biases - (output_error * args.learning_rate);
        input_error
    }
}

impl<const N: usize, const M: usize> Default for DenseLayer<N, M> {
    fn default() -> Self {
        Self {
            weights: Matrix::generate(|| ((rand::random::<f64>()) - 0.5) / (N * M) as f64),
            biases: Matrix::generate(|| ((rand::random::<f64>()) - 0.5) / M as f64),
        }
    }
}

pub trait NetworkLayer<const N: usize, const M: usize> {
    fn forward<const K: usize>(&self, l: Matrix<N, K>) -> Matrix<M, K>;

    fn fit(
        &mut self,
        input: Matrix<N, 1>,
        output_error: Matrix<M, 1>,
        args: &LearningArgs,
    ) -> Matrix<N, 1>;

    fn calulate_error<const K: usize>(&self, x: Matrix<N, K>, y: Matrix<M, K>) -> Matrix<M, K> {
        let loss = self.forward(x) - y;
        2.0 * loss
    }
}

impl<const N: usize, const M: usize, A: Layer<N, M>> NetworkLayer<N, M> for (A,) {
    fn forward<const K: usize>(&self, l: Matrix<N, K>) -> Matrix<M, K> {
        self.0.forward(l)
    }

    fn fit(
        &mut self,
        input: Matrix<N, 1>,
        output_error: Matrix<M, 1>,
        args: &LearningArgs,
    ) -> Matrix<N, 1> {
        self.0.learn(input, output_error, args)
    }
}

impl<const N: usize, const M: usize, const K: usize, A: Layer<N, K>, B: NetworkLayer<K, M>>
    NetworkLayer<N, M> for (Wrapper<N, K, A>, B)
{
    fn forward<const P: usize>(&self, l: Matrix<N, P>) -> Matrix<M, P> {
        self.1.forward(self.0 .0.forward(l))
    }

    fn fit(
        &mut self,
        input: Matrix<N, 1>,
        output_error: Matrix<M, 1>,
        args: &LearningArgs,
    ) -> Matrix<N, 1> {
        self.0 .0.learn(
            input.clone(),
            self.1.fit(self.0 .0.forward(input), output_error, args),
            args,
        )
    }
}

#[derive(Debug)]
pub struct Network<const N: usize, const M: usize, T: NetworkLayer<N, M>>(T);

impl<const N: usize, const M: usize, T: NetworkLayer<N, M>> Network<N, M, T> {
    pub fn new(t: T) -> Self {
        Self(t)
    }
    fn calculate_error<const K: usize>(&self, x: &Matrix<N, K>, y: &Matrix<M, K>) -> Matrix<M, K> {
        2.0 * (self.predict(x.clone()) - y)
    }
    pub fn fit<const K: usize>(&mut self, x: Matrix<N, K>, y: Matrix<M, K>, args: &LearningArgs) {
        for e in 0..args.epochs {
            for i in 0..K {
                for _ in 0..args.single_epochs {
                    let x = x.sub(i);
                    let y = y.sub(i);
                    for _e in 0..args.epochs {
                        self.0.fit(x.clone(), self.calculate_error(&x, &y), args);
                    }
                }
            }

            if e % 10 == 0 {
                println!("epochs: {}", e);
            }
        }
    }
    pub fn predict<const K: usize>(&self, x: Matrix<N, K>) -> Matrix<M, K> {
        self.0.forward(x)
    }
}

#[macro_export]
macro_rules! network_layers {
    ($e:expr $(,)?) => {
        ($e,)
    };
    ($e:expr,$($es:expr),* $(,)?) => {
        ($crate::layer::Wrapper($e),network_layers!($($es),*))
    };

}
#[macro_export]
macro_rules! network {
    ($e:expr $(,)?) => {
        $crate::layer::Network::new(network_layers!($e,))
    };
    ($($es:expr),* $(,)?) => {
        $crate::layer::Network::new(network_layers!($($es),*))
    };

}
