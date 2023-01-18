use matrix::Matrix;

use crate::layer::{DenseLayer, LearningArgs, TanhActivation};
pub mod layer;
pub mod matrix;

fn main() {
    let args = LearningArgs {
        learning_rate: 0.5,
        epochs: 10,
        single_epochs: 10,
    };
    let mut net = network!(
        DenseLayer::<2, 5>::default(),
        DenseLayer::<5, 1>::default(),
    );
    const MAX:usize = 5;

    let mut x = Matrix::<2, 25>::default();
    let mut y = Matrix::<1, 25>::default();
    for i in 0..MAX {
        for j in 0..MAX {
            x.set_sub(i + j * MAX, &Matrix::from([[i as f64, j as f64]]));
            y.set_sub(
                i + j * MAX,
                // &Matrix::from([[if i == j { 1.0 } else { -1.0 }]]),
                &Matrix::from([[(i+j) as f64]]),
            )
        }
    }
    net.fit(x, y, &args);
    println!("{:?}", net);

    println!("{}", net.predict([[0, 1]].into()));
    println!("{}", net.predict([[1, 0]].into()));
    println!("{}", net.predict([[1, 1]].into()));
    println!("{}", net.predict([[0, 0]].into()));
    println!("{}", net.predict([[100, 156]].into()));
}
