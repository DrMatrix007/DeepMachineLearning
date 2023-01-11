use std::ops::{Index, IndexMut};

pub struct Matrix<const N: usize, const M: usize>(pub(self) Vec<f64>);

impl<const N: usize, const M: usize> Matrix<N, M> {
    pub fn new() -> Self {
        Self(Vec::with_capacity(N * M))
    }
}

impl<const N: usize, const M: usize> Index<(usize,usize)> for Matrix<N, M> {
    type Output = f64;

    fn index(&self, index: (usize,usize)) -> &Self::Output {
        return self.0.get(index.0+index.1*M).unwrap();
    }
}

impl<const N: usize, const M: usize> IndexMut<(usize,usize)> for Matrix<N, M> {
    fn index_mut(&mut self, index: (usize,usize)) -> &mut Self::Output {
        return self.0.get_mut(index.0+index.1*M).unwrap();   
    }
}
pub trait ToMatrix<const N: usize, const M: usize> {
    fn to_matrix(self) -> Matrix<N, M>;
}

impl<T:Into<f64>+Clone,const N: usize, const M: usize> ToMatrix<N, M> for [[T; N]; M] {
    fn to_matrix(self) -> Matrix<N, M> {
        Matrix(self.concat().into_iter().map(|x|x.into()).collect())
    }
}
