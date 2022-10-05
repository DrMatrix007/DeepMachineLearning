#![allow(dead_code, unused_imports)]

use std::f64::consts::PI;

use rand::Rng;

use crate::{
    activation::{Linear, ReLU, Sigmoid, Tanh},
    feed::{FeedableForward, Learnable, LearningArgs},
    matrix::ToMatrix,
    network::Layer,
};

pub mod activation;
pub mod feed;
pub mod matrix;
pub mod network;

// network!(MyNetwork; (2, Sigmoid), (20, Sigmoid), (20, Sigmoid));
fn main() {
    let args = LearningArgs {
        learning_rate: 1.,
    };

    let mut rng = rand::thread_rng();
    let mut x: f64;
    let mut y: f64;
    let mut nn = network_layers!(auto(); (2),(8,Tanh),(1,Sigmoid));
    // {
    //     let max = 10000;
    //     let mut val;
    //     let mut avg = 0.;
    //     for i in 0..max {
    //         val = i as f64 / max as f64;
    //         avg += ((val).exp() - nn.feed_forward([[val]].to_matrix())[(0, 0)]) / max as f64;
    //     }
    //     // println!("avg: {}",avg);
    //     println!("from: {}%", avg * 100.);
    // }

    for _ in 0..1000000 {
        x = rng.gen_range(0..=1) as f64;
        y = rng.gen_range(0..=1) as f64;
        // XOR
        nn.learn(
            [[x, y]].to_matrix().transpose(),
            [[if x == y { 0. } else { 1. }]].to_matrix(),
            &args,
        );
    }

    {
        println!("nn: {:?}", nn);
        let data = [[1., 1.], [1., 0.], [0., 1.], [0., 0.]]
            .to_matrix()
            .transpose();
        println!("{}", (nn.feed_forward(data)))
    }
}
