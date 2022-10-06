use std::{
    ops::{Add, Div, Index, IndexMut, Mul, Sub},
    vec,
};
macro_rules! impl_matrix_with_name {
    ($($a:tt)+) => {
        impl<const N: usize, const M: usize> Add<Self> for $($a)+<N, M> {
            type Output = Matrix<N, M>;
        
            fn add(self, rhs: Self) -> Self::Output {
                self.apply(|x, pos| x + rhs[pos])
            }
        }
        
        impl<const N: usize, const M: usize> Sub<Self> for $($a)+<N, M> {
            type Output = Matrix<N, M>;
        
            fn sub(self, rhs: Self) -> Self::Output {
                self.apply(|x, pos| x - rhs[pos])
            }
        }
        
        impl<const N: usize, const M: usize, T: Into<f64> + Copy> Mul<T> for $($a)+<N, M> {
            type Output = Matrix<N, M>;
        
            fn mul(self, rhs: T) -> Self::Output {
                self.apply(|x, _| x * rhs.into())
            }
        }
        
        impl<const N: usize, const M: usize, T: Into<f64> + Copy> Div<T> for $($a)+<N, M> {
            type Output = Matrix<N, M>;
        
            fn div(self, rhs: T) -> Self::Output {
                self.apply(|x, _| x / rhs.into())
            }
        }
        
        impl<const N: usize, const M: usize, const K: usize> Mul<$($a)+<M, K>> for $($a)+<N, M> {
            type Output = Matrix<N, K>;
            fn mul(self, rhs: $($a)+<M, K>) -> Self::Output {
                let mut ans = Matrix::default();
        
                for x in 0..N {
                    for y in 0..K {
                        for z in 0..M {
                            ans[(y, x)] += self[(z, x)] * rhs[(y, z)];
                        }
                    }
                }
                ans
            }
        }
        
    };
}
#[derive(Debug)]
pub struct Matrix<const N: usize, const M: usize>(Vec<Vec<f64>>);
impl<const N: usize, const M: usize> Matrix<N, M> {

    pub fn element_wise_product(&self,rhs:&Matrix<N,M>) -> Self {
        let mut ans = Matrix::default();
        for i in 0..N {
            for j in 0..M {
                ans[(j,i)] = self[(j,i)]*rhs[(j,i)];
            }
        }
        ans
    }

    pub fn apply(&self, f: impl Fn(f64, (usize, usize)) -> f64) -> Self {
        let f = &f;
        Matrix(
            self.0
                .iter()
                .enumerate()
                .map(|(y, v)| {
                    v.iter()
                        .copied()
                        .enumerate()
                        .map(|(x, v)| f(v, (y, x)))
                        .collect()
                })
                .collect(),
        )
    }
    pub fn apply_mut(&self, mut f: impl FnMut(f64, (usize, usize)) -> f64) -> Self {
        let mut a = Matrix::default();
        for i in a.0.iter_mut().enumerate() {
            for x in i.1.iter_mut().enumerate() {
                *x.1 = f(*x.1, (i.0, x.0))
            }
        }
        a
    }

    pub fn add_vec(&self, biases: &Matrix<1, M>) -> Matrix<N, M> {
        self.apply(|v,(y,x)|v+biases[(0,x)])
    }
    pub fn transpose(&self) -> Matrix<M, N> {
        let mut ans = Matrix::default();
        for x in 0..N {
            for y in 0..M {
                ans[(x, y)] = self[(y, x)];
            }
        }
        ans
    }
}
impl<const N: usize, const M: usize> Default for Matrix<N, M> {
    fn default() -> Self {
        Self(vec![vec![0.0; N]; M])
    }
}
impl<const N: usize, const M: usize> Index<(usize, usize)> for Matrix<N, M> {
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.0[index.0][index.1]
    }

    type Output = f64;
}
impl<const N: usize, const M: usize> IndexMut<(usize, usize)> for Matrix<N, M> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.0[index.0][index.1]
    }
}
impl_matrix_with_name!(Matrix);
impl_matrix_with_name!(&Matrix);
pub trait ToMatrix<const N: usize, const M: usize> {
    fn to_matrix(&self) -> Matrix<N, M>;
}
impl<T: Into<f64> + Copy, const N: usize, const M: usize> ToMatrix<N, M> for [[T; M]; N] {
    fn to_matrix(&self) -> Matrix<N, M> {
        let mut ans = Matrix::default();
        for x in 0..N {
            for y in 0..M {
                ans[(y, x)] = self[x][y].into();
            }
        }

        ans
    }
}

