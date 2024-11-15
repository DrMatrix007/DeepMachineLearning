use matrix::Matrix;

use crate::{
    activation::{LeakyReLUActivation, SigmoidActivation, TanhActivation},
    layers::{DenseLayer, LearningArgs},
    optimizers::{AdamOptimizer, AdamOptimizerArgs, GradientDecentOptimizer, GradientDecentOptimizerArgs},
};
pub mod activation;
pub mod layers;
pub mod matrix;
pub mod optimizers;
fn main() {
    // let args = LearningArgs::<AdamOptimizerArgs> {
    //     learning_rate: 0.1,
    //     epochs: 10,
    //     single_epochs: 100,
    //     op_args: AdamOptimizerArgs {
    //         beta1: 0.9,
    //         beta2: 0.999,
    //         epsilon: 1e-8,
    //         eta: 0.01,
    //     },
    // };
    let args = LearningArgs {
        epochs: 100,
        single_epochs: 10,
        op_args: GradientDecentOptimizerArgs {
            learning_rate: 0.01, 
        },
    };
    let mut net = network! {
            optimizer: GradientDecentOptimizer,
            layers: [
                 (2,10,TanhActivation),
                 (10,1,LeakyReLUActivation)
            ]
    };
    let mut x = Matrix::<2, 4>::default();
    let mut y = Matrix::<1, 4>::default();

    for i in 0..2 {
        for j in 0..2 {
            x.set_sub(j + i * 2, &Matrix::from([[i as f64, j as f64]]));
            y.set_sub(
                j + i * 2,
                &Matrix::from([[if i != j { 1 } else { -1 }]]),
            )
        }
    }

    net.fit(x, y, &args);
    // println!("{:?}", net);
    println!("{}", net.predict([[0, 1]].into()));
    println!("{}", net.predict([[1, 0]].into()));
    println!("{}", net.predict([[1, 1]].into()));
    println!("{}", net.predict([[0, 0]].into()));
}
