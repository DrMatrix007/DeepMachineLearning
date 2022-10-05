use std::marker::PhantomData;

use rand::{distributions::Standard, prelude::Distribution};

use crate::{
    activation::ActivationFunction,
    matrix::{Matrix, Number},
};

#[derive(Debug)]
pub struct Layer<const N: usize, const M: usize, F: ActivationFunction<f64>> {
    pub weights: Matrix<N, M>,
    pub biases:Matrix<1,M>,
    l: PhantomData<F>,
}
impl<const N: usize, const M: usize, F: ActivationFunction<f64>> Layer<N, M, F>
where
{
    pub fn new(m: Matrix<N, M>,b:Matrix<1,M>) -> Self {
        Self {
            weights: m,
            biases:b,
            l: Default::default(),
        }
    }
    pub fn random(l: f64, h: f64) -> Self {
        Self {
            weights: Matrix::random(l, h),
            biases: Matrix::random(l, h),
            l: Default::default(),
        }
    }
    pub fn auto() -> Self {
        Self {
            weights: Matrix::random(-6.0/((M+N) as f64).sqrt(), 6.0/((M+N) as f64).sqrt()),
            biases: Matrix::random(-6.0/((N+M) as f64).sqrt(), 6.0/((M+N) as f64).sqrt()),
            l:Default::default()
        }
    }
    pub fn zero() -> Self {
        Default::default()
    }
}
impl<const N: usize, const M: usize, F: ActivationFunction<f64>> Default
    for Layer<N, M, F>
{
    fn default() -> Self {
        Self {
            weights: Default::default(),
            biases:Default::default(),
            l: Default::default(),
        }
    }
}
#[macro_export]
macro_rules! network_layers_type {
    [($x:tt,$l1:ty);($y:tt,$l2:ty);$(($z:tt,$l3:ty));*]=>{
        ($crate::Layer<$x,$y,$l1> , network_layers_type![($y,$l2);$(($z,$l3));*])
    };
    [($x:tt,$l1:ty);($y:tt,$l2:ty)] => {
        $crate::Layer<$x,$y,$l1>
    };
}
#[macro_export]
macro_rules! network_layers {
    [$f:ident;($x:expr),($y:expr,$l2:ty),$(($z:expr,$l3:ty)),*]=>{
        ($crate::Layer::<$x,$y,$l2>::$f($($data:expr),*) ,network_layers![$f(),($y),$(($z,$l3)),*],)
    };
    [$f:ident($($data:expr),*);($x:expr),($y:expr,$l2:ty),$(($z:expr,$l3:ty)),*]=>{
        ($crate::Layer::<$x,$y,$l2>::$f($($data),*) ,network_layers![$f($($data),*);($y),$(($z,$l3)),*],)
    };
    [$f:ident($($data:expr),*);($x:expr),($y:expr,$l2:ty)] => {
        $crate::Layer::<$x,$y,$l2>::$f($($data),*)
    };
    [($x:expr),($y:expr,$l2:ty),$(($z:expr,$l3:ty)),*]=>{
        network_layers!(default();($x),($y,$l2),$(($z,$l3)),*)
    };

}

#[macro_export]
macro_rules! test {
    ($(($layers:expr,$activation:expr));*) => {
        ($($layers,$activation),*)
    };
}

#[macro_export]
macro_rules! network {
    ($name:ident; $(($layers:expr,$activation:ty)),*) => {
        struct $name {
            network:network_layers_type![$(($layers, $activation));*],
        }
        impl $name {
            fn new() -> Self {
                Self{
                    network:network_layers![default;$(($layers,$activation));*],
                }
            }
        }
    };
}

// impl<const N:usize,const M:usize,const FINAL:usize,T:Number> FeedableForward<N,FINAL,T> for (Matrix<N,M,T>,Matrix<M,FINAL,T>) {
//     fn feed<const DATA_T:usize>(&self,data:Matrix::<DATA_T,N,T>) -> Matrix<DATA_T, FINAL ,T>  {
//         data*self.0*self.1
//     }
// }
