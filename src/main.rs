#![allow(dead_code, unused_imports)]

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
        learning_rate: 0.001,
    };

    let mut rng = rand::thread_rng();
    let mut x: f64;
    loop {
        let mut nn = network_layers!(random(-1.,1.); (1),(8,ReLU),(8,ReLU),(1,Linear));
        {
            let max = 10000;
            let mut val;
            let mut avg = 0.;
            for i in 0..max {
                val = i as f64 / max as f64;
                avg += ((val).exp() - nn.feed_forward([[val]].to_matrix())[(0, 0)]) / max as f64;
            }
            // println!("avg: {}",avg);
            println!("from: {}%", avg * 100.);
        }
        for _ in 0..1000000 {
            x = rng.gen_range(0.0..1.);
            nn.learn([[x]].to_matrix(), [[x.exp()]].to_matrix(), &args);
        }

        {
            let max = 10000;
            let mut val;
            let mut avg = 0.;
            for i in 0..max {
                val = i as f64 / max as f64;
                avg += ((val).exp() - nn.feed_forward([[val]].to_matrix())[(0, 0)]) / max as f64;
            }
            // println!("avg: {}",avg);
            println!("to: {}", avg);
        }
    }
}
