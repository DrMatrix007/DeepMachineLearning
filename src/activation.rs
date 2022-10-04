use std::{f64::consts::E, fmt::Debug};

use crate::matrix::Number;

pub trait ActivationFunction<T: Number = f64> {
    fn f(x: T) -> T;
    fn der(x: T) -> T;
}
#[derive(Debug)]
pub struct Sigmoid;
impl ActivationFunction<f64> for Sigmoid {
    fn f(x: f64) -> f64 {
        1. / (1. + E.powf(-x))
    }

    fn der(x: f64) -> f64 {
        let y = Self::f(x);
        y * (1. - y)
    }
}

#[derive(Debug)]
pub struct Tanh;
impl ActivationFunction<f64> for Tanh {
    fn f(x: f64) -> f64 {
        x.tanh()
    }

    fn der(x: f64) -> f64 {
        1.0-x*x
    }
}
#[derive(Debug)]
pub struct ReLU;
impl<T: Number> ActivationFunction<T> for ReLU {
    fn f(x: T) -> T {
        if x > 0.0.into() {
            x
        } else {
            0.0.into()
        }
    }

    fn der(x: T) -> T {
        if x >= 0.0.into() {
            1.0.into()
        } else {
            0.0.into()
        }
    }
}
#[derive(Debug)]
pub struct Linear;
impl<T:Number> ActivationFunction<T> for Linear {
    fn f(x: T) -> T {
        x
    }

    fn der(_: T) -> T {
        1.0.into()
    }
}