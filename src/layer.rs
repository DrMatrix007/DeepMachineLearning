use crate::matrix::Matrix;

pub struct Wrapper<const N:usize,const M:usize,T:Layer<N,M>>(pub T);

pub trait Layer<const N: usize, const M: usize> {
    fn forward<const K: usize>(&self, l: Matrix<K, N>) -> Matrix<K, M>;
    fn backward<const K: usize>(&self, l: Matrix<K, M>) -> Matrix<K, N>;
}
pub trait Activation<const N: usize> {
    fn forward<const K: usize>(&self, l: Matrix<K, N>) -> Matrix<K, N>;
    fn backward<const K: usize>(&self, l: Matrix<K, N>) -> Matrix<K, N>;
}
impl<const N: usize, T: Activation<N>> Layer<N, N> for T {
    fn forward<const K: usize>(&self, l: Matrix<K, N>) -> Matrix<K, N> {
        self.forward(l)
    }

    fn backward<const K: usize>(&self, l: Matrix<K, N>) -> Matrix<K, N> {
        self.backward(l)
    }
}

pub struct DenseLayer<const N: usize, const M: usize> {
    weights: Matrix<N, M>,
}

impl<const N: usize, const M: usize> Layer<N, M> for DenseLayer<N, M> {
    fn forward<const K: usize>(&self, l: Matrix<K, N>) -> Matrix<K, M> {
        l * &self.weights
    }

    fn backward<const K: usize>(&self, l: Matrix<K, M>) -> Matrix<K, N> {
        l * self.weights.trasnpose()
    }
}

impl<const N: usize, const M: usize> Default for DenseLayer<N, M> {
    fn default() -> Self {
        Self {
            weights: Default::default(),
        }
    }
}

pub struct LinearActivation;

impl<const M: usize> Activation<M> for LinearActivation {
    fn forward<const K: usize>(&self, l: Matrix<K, M>) -> Matrix<K, M> {
        l
    }

    fn backward<const K: usize>(&self, _: Matrix<K, M>) -> Matrix<K, M> {
        Matrix::ones()
    }
}

pub trait Forwardable<const N: usize, const M: usize> {
    fn forward<const K: usize>(&self, l: Matrix<K, N>) -> Matrix<K, M>;
    fn backward<const K: usize>(&self, l: Matrix<K, M>) -> Matrix<K, N>;
}

impl<const N: usize, const M: usize, A: Layer<N, M>> Forwardable<N, M> for (A,) {
    fn forward<const K: usize>(&self, l: Matrix<K, N>) -> Matrix<K, M> {
        Layer::forward(&self.0,l)
    }
    
    fn backward<const K: usize>(&self, l: Matrix<K, M>) -> Matrix<K, N> {
        Layer::backward(&self.0,l)
    }
}

impl<const N: usize, const M: usize, const K:usize,A: Layer<N, K>,B:Forwardable<K,M>> Forwardable<N, M> for (Wrapper<N,K,A>,B) {
    fn forward<const P: usize>(&self, l: Matrix<P, N>) -> Matrix<P, M> {
        self.1.forward(self.0.0.forward(l))
    }

    fn backward<const P: usize>(&self, l: Matrix<P, M>) -> Matrix<P, N> {
        self.0.0.backward(self.1.backward(l))
    }
}


#[macro_export]
macro_rules! network {
    ($e:expr) => {
        ($e,)
    };
    ($e:expr,$($es:expr),*) => {
        ($crate::layer::Wrapper($e),network!($($es),*))
    };
    
}