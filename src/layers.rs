use std::{fmt::Debug, marker::PhantomData};

use crate::{
    activation::Activation,
    matrix::Matrix,
    optimizers::{OptimizationArgs, Optimizer, OptimizerState},
};

pub struct LearningArgs<OpArgs: OptimizationArgs> {
    pub epochs: i32,
    pub single_epochs: i32,
    pub op_args: OpArgs,
}

// #[derive(Debug)]
// pub struct Wrapper<const N: usize, const M: usize, T: Layer<N, M>>(pub T);

#[derive(Debug)]
pub struct DenseLayer<const N: usize, const M: usize, Act: Activation, Op: Optimizer> {
    weights: Matrix<M, N>,
    biases: Matrix<M, 1>,
    optimizer: Op::State<M, N>,
    activation: Act,
}
impl<const N: usize, const M: usize, Act: Activation, Op: Optimizer> DenseLayer<N, M, Act, Op> {
    fn add_bias<const K: usize, const P: usize>(
        m: &Matrix<P, K>,
        b: &Matrix<P, 1>,
    ) -> Matrix<P, K> {
        m.map(|(x, _), val| val + b[(x, 0)])
    }
    fn forward<const K: usize>(&self, l: Matrix<N, K>) -> Matrix<M, K> {
        self.activation
            .forward(Self::add_bias(&(l * &self.weights), &self.biases))
    }
    fn learn(
        &mut self,
        input: Matrix<N, 1>,
        output_error: Matrix<M, 1>,
        args: &LearningArgs<Op::Args>,
    ) -> Matrix<N, 1> {
        let output_error = output_error.element_wise_product(
            &self
                .activation
                .backward(Self::add_bias(&(&input * &self.weights), &self.biases)),
        );
        let (a, b) = self.optimizer.update(
            &self.weights,
            &self.biases,
            &(input.trasnpose() * &output_error),
            &output_error,
            &args.op_args,
        );
        self.weights = a;
        self.biases = b;

        output_error * self.weights.trasnpose()
    }
}

impl<const N: usize, const M: usize, Act: Activation, Op: Optimizer> Default
    for DenseLayer<N, M, Act, Op>
{
    fn default() -> Self {
        Self {
            weights: Matrix::generate(|| ((rand::random::<f64>()) - 0.5) / (N * M) as f64),
            biases: Matrix::generate(|| ((rand::random::<f64>()) - 0.5) / M as f64),
            optimizer: Op::get_state(),
            activation: Act::default(),
        }
    }
}

pub trait NetworkLayer<const N: usize, const M: usize, Op: Optimizer> {
    fn forward<const K: usize>(&self, l: Matrix<N, K>) -> Matrix<M, K>;

    fn fit(
        &mut self,
        input: Matrix<N, 1>,
        output_error: Matrix<M, 1>,
        args: &LearningArgs<Op::Args>,
    ) -> Matrix<N, 1>;

    fn calulate_error<const K: usize>(&self, x: Matrix<N, K>, y: Matrix<M, K>) -> Matrix<M, K> {
        let loss = self.forward(x) - y;
        2.0 * loss
    }
}

impl<const N: usize, const M: usize, Act: Activation, Op: Optimizer> NetworkLayer<N, M, Op>
    for (DenseLayer<N, M, Act, Op>,)
{
    fn forward<const K: usize>(&self, l: Matrix<N, K>) -> Matrix<M, K> {
        self.0.forward(l)
    }

    fn fit(
        &mut self,
        input: Matrix<N, 1>,
        output_error: Matrix<M, 1>,
        args: &LearningArgs<Op::Args>,
    ) -> Matrix<N, 1> {
        self.0.learn(input, output_error, args)
    }
}

impl<
        const N: usize,
        const M: usize,
        const K: usize,
        Act: Activation,
        Op: Optimizer,
        B: NetworkLayer<K, M, Op>,
    > NetworkLayer<N, M, Op> for (DenseLayer<N, K, Act, Op>, B)
{
    fn forward<const P: usize>(&self, l: Matrix<N, P>) -> Matrix<M, P> {
        self.1.forward(self.0.forward(l))
    }

    fn fit(
        &mut self,
        input: Matrix<N, 1>,
        output_error: Matrix<M, 1>,
        args: &LearningArgs<Op::Args>,
    ) -> Matrix<N, 1> {
        self.0.learn(
            input.clone(),
            self.1.fit(self.0.forward(input), output_error, args),
            args,
        )
    }
}

#[derive(Debug)]
pub struct Network<Op: Optimizer, const N: usize, const M: usize, T: NetworkLayer<N, M, Op>>(
    T,
    PhantomData<Op>,
);

impl<const N: usize, const M: usize, Op: Optimizer, T: NetworkLayer<N, M, Op>>
    Network<Op, N, M, T>
{
    pub fn new(t: T) -> Self {
        Self(t, Default::default())
    }
    fn calculate_error<const K: usize>(&self, x: &Matrix<N, K>, y: &Matrix<M, K>) -> Matrix<M, K> {
        2.0 * (self.predict(x.clone()) - y)
    }
    pub fn fit<const K: usize>(
        &mut self,
        x: Matrix<N, K>,
        y: Matrix<M, K>,
        args: &LearningArgs<Op::Args>,
    ) {
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
    ($op:ty,($x:tt,$y:tt,$act:ty) $(,($xs:tt,$ys:tt,$acts:ty))+ $(,)?) => {
        ($crate::layers::DenseLayer::<$x,$y,$act,$op>::default(),network_layers!($op,$(($xs,$ys,$acts),)+))
    };

    ($op:ty,($x:tt,$y:tt,$act:ty) $(,)?) => {
        ($crate::layers::DenseLayer::<$x,$y,$act,$op>::default(),)
    };
}
#[macro_export]
macro_rules! get_first {
    ($e:tt $(,$es:tt)* $(,)?) => {
        ($e)
    };
}

#[macro_export]
macro_rules! get_last {
    ($e:tt $(,$es:tt)+ $(,)?) => {
        $crate::get_last!($($es,)*)
    };

    ($e:tt $(,)?) => {
        ($e)
    };
}

#[macro_export]
macro_rules! network {
    (optimizer: $op:ty, layers: [$(($x:tt,$y:tt,$act:ty)),* $(,)?]) => {
        $crate::layers::Network::<$op,{$crate::get_first!($($x,)*)},{$crate::get_last!($($y,)*)},_>::new(network_layers!($op,$(($x,$y,$act),)*))
    };

}
