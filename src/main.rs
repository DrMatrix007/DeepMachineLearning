use matrix::Matrix;

use crate::{layer::{DenseLayer, LearningArgs}, activation::LeakyReLUActivation};
pub mod layer;
pub mod matrix;
pub mod activation;
pub mod optimizers;
fn main() {
    let args = LearningArgs {
        learning_rate: 0.1,
        epochs: 100,
        single_epochs: 100,
    };
    let mut net = network!(
        DenseLayer::<2, 9>::default(),
        LeakyReLUActivation,
        DenseLayer::<9, 1>::default(),
        LeakyReLUActivation
    );
    let mut x = Matrix::<2, 4>::default();
    let mut y = Matrix::<1, 4>::default();

    for i in 0..2 {
        for j in 0..2 {
            x.set_sub(j + i * 2, &Matrix::from([[i as f64, j as f64]]));
            y.set_sub(
                j + i * 2,
                &Matrix::from([[if i == j { -5.0 } else { 0.5 }]]),
            )
        }
    }
    net.fit(x, y, &args);
    println!("{:?}", net);

    println!("{}", net.predict([[0, 1]].into()));
    println!("{}", net.predict([[1, 0]].into()));
    println!("{}", net.predict([[1, 1]].into()));
    println!("{}", net.predict([[0, 0]].into()));
}
