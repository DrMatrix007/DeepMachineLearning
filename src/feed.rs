use crate::{
    activation::ActivationFunction,
    matrix::{Matrix, Number},
    network::Layer,
};

pub trait FeedableForward<const M: usize, const FINAL: usize> {
    type Func: ActivationFunction;
    fn feed_forward<const DATA_T: usize>(
        &self,
        data: Matrix<DATA_T, M>,
    ) -> Matrix<DATA_T, FINAL>;
}
impl<const M: usize, const FINAL: usize, F: ActivationFunction>
    FeedableForward<M, FINAL> for Layer<M, FINAL, F>
{
    type Func = F;
    fn feed_forward<const DATA_T: usize>(
        &self,
        data: Matrix<DATA_T, M>,
    ) -> Matrix<DATA_T, FINAL> {
        (data * self.weights)
            .add_vec(self.biases)
            .apply(Self::Func::f)
    }
}

impl<
        const N: usize,
        const M: usize,
        const FINAL: usize,
        F: ActivationFunction,
        A: FeedableForward<M, FINAL>,
    > FeedableForward<N, FINAL> for (Layer<N, M, F>, A)
{
    fn feed_forward<const DATA_T: usize>(
        &self,
        data: Matrix<DATA_T, N>,
    ) -> Matrix<DATA_T, FINAL> {
        self.1.feed_forward(
            (data * self.0.weights)
                .add_vec(self.0.biases)
                .apply(Self::Func::f),
        )
    }
    type Func = F;
}

#[derive(Clone, Copy, Debug)]
pub struct LearningArgs {
    pub learning_rate: f64,
}
pub trait Learnable<const M: usize, const START: usize, T: Number = f64> {
    type Func: ActivationFunction<T>;
    fn learn(
        &mut self,
        x: Matrix<1, M, T>,
        y: Matrix<1, START, T>,
        args: &LearningArgs,
    ) -> Matrix<1, M, T>;
}

impl<const M: usize, const START: usize, F: ActivationFunction> Learnable<M, START>
    for Layer<M, START, F>
{
    type Func = F;
    fn learn(&mut self, x: Matrix<1, M>, y: Matrix<1, START>, args: &LearningArgs) -> Matrix<1, M> {
        let a = (x * self.weights).add_vec(self.biases).apply(Self::Func::f);
        let dc = y - a;
        let delta = dc.element_wise_product(a.apply(Self::Func::der));
        let ans = (delta * self.weights.transpose()).element_wise_product(x.apply(Self::Func::der));
        // println!("{} {}",a,  a.apply(Self::Func::der));
        self.weights = self.weights + (x.transpose() * delta) * args.learning_rate;
        self.biases = self.biases + delta * args.learning_rate;
        ans
    }
}

impl<
        const N: usize,
        const M: usize,
        const START: usize,
        F: ActivationFunction,
        A: Learnable<M, START>,
    > Learnable<N, START> for (Layer<N, M, F>, A)
{
    type Func = F;

    fn learn(&mut self, x: Matrix<1, N>, y: Matrix<1, START>, args: &LearningArgs) -> Matrix<1, N> {
        let a = (x * self.0.weights).add_vec(self.0.biases).apply(Self::Func::f);
        let delta = self.1.learn(a, y, args);

        let dot = delta * self.0.weights.transpose();
        let ans = dot.element_wise_product(x.apply(Self::Func::der));

        self.0.weights = self.0.weights + (x.transpose() * delta) * args.learning_rate;
        self.0.biases = self.0.biases + delta * args.learning_rate;

        ans
    }
}

// pub trait Learnable<const M: usize, const START: usize, T: Number> {
//     type Func: ActivationFunction<T>;
//     fn learn<const DATA_T: usize>(
//         &self,
//         x: Matrix<DATA_T,M,T>,
//         y: Matrix<DATA_T,START, T>,
//         args:LearningArgs
//     ) -> Matrix<DATA_T, M, T>;
// }
//
// impl<const M: usize, const START: usize, T: Number, F: ActivationFunction<T>>
//     Learnable<M, START, T> for Layer<M, START, F, T>
// {
//     type Func = F;
//     fn learn<const DATA_T: usize>(
//         &self,
//         x: Matrix<DATA_T,M,T>,
//         y: Matrix<DATA_T, START, T>,
//         args:LearningArgs
//     ) -> Matrix<DATA_T, M, T> {
//         let ans = (x*self.weights).apply(Self::Func::f);
//         let delta = ans-y;
//
//         ( delta*self.weights.transpose()).apply(Self::Func::der)
//     }
// }
//
// impl<
//         const N: usize,
//         const M: usize,
//         const START: usize,
//         T: Number,
//         F: ActivationFunction<T>,
//         A: Learnable<M, START, T>,
//     > Learnable<N, START, T> for (Layer<N, M, F, T>, A)
// {
//     type Func = F;
//
//     fn learn<const DATA_T: usize>(
//         &self,
//         x: Matrix<DATA_T,N,T>,
//         y: Matrix<DATA_T, START, T>,
//         args:LearningArgs
//     ) -> Matrix<DATA_T, N, T> {
//         let delta = self.1.learn((x*self.0.weights).apply(Self::Func::f),y,args);
//         (delta*self.0.weights.transpose()).apply(Self::Func::der)
//     }
// }
