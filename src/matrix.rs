use std::{
    ops::{Add, Div, Index, IndexMut, Mul, Sub},
    vec,
};
macro_rules! impl_matrix_with_name {
    ($($a:tt)+) => {
        impl<const N: usize, const M: usize> Add<Self> for $($a)+<N, M> {
            type Output = Matrix<N, M>;
        }
    }
}
macro_rules! impl_matrix_one {
    ($($a:tt)+) => {
        impl<const N: usize, const M: usize, T: Into<f64> + Copy> Mul<T> for $($a)+<N, M> {
            type Output = Matrix<N, M>;

            fn mul(self, rhs: T) -> Self::Output {
                self.apply(|x| x * rhs.into())
            }
        }

        impl<const N: usize, const M: usize, T: Into<f64> + Copy> Div<T> for $($a)+<N, M> {
            type Output = Matrix<N, M>;

            fn div(self, rhs: T) -> Self::Output {
                self.apply(|x| x / rhs.into())
            }
        }

    };
}
macro_rules! impl_matrix_for_matrix {
    (($($a:tt)+),($($b:tt)+)) => {
        impl<const N: usize, const M: usize> Add<$($b)+<N,M>> for $($a)+<N, M> {
            type Output = Matrix<N, M>;

            fn add(self, rhs: $($b)+<N,M>) -> Self::Output {
                self.apply_pos(|x, pos| x + rhs[pos])
            }
        }

        impl<const N: usize, const M: usize> Sub<$($b)+<N,M>> for $($a)+<N, M> {
            type Output = Matrix<N, M>;

            fn sub(self, rhs: $($b)+<N,M>) -> Self::Output {
                self.apply_pos(|x, pos| x - rhs[pos])
            }
        }


        impl<const N: usize, const M: usize, const K: usize> Mul<$($b)+<M, K>> for $($a)+<N, M> {
            type Output = Matrix<N, K>;
            fn mul(self, rhs: $($b)+<M, K>) -> Self::Output {
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
#[derive(Debug,Clone)]
pub struct Matrix<const N: usize, const M: usize>(Vec<Vec<f64>>);
impl<const N: usize, const M: usize> Matrix<N, M> {
    pub fn element_wise_product(&self, rhs: &Matrix<N, M>) -> Self {
        let mut ans = Matrix::default();
        for i in 0..N {
            for j in 0..M {
                ans[(j, i)] = self[(j, i)] * rhs[(j, i)];
            }
        }
        ans
    }

    pub fn apply_pos(&self, f: impl Fn(f64, (usize, usize)) -> f64) -> Self {
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
    pub fn apply(&self, f: impl Fn(f64) -> f64) -> Self {
        let f = &f;
        Matrix(
            self.0
                .iter()
                .map(|vec| vec.iter().copied().map(f).collect())
                .collect(),
        )
    }
    pub fn apply_mut(&self, mut f: impl FnMut(f64) -> f64) -> Self {
        let mut a = Matrix::default();
        for i in &mut a.0 {
            for x in i {
                *x = f(*x)
            }
        }
        a
    }
    pub fn apply_pos_mut(&self, mut f: impl FnMut(f64, (usize, usize)) -> f64) -> Self {
        let mut a = Matrix::default();
        for i in a.0.iter_mut().enumerate() {
            for x in i.1.iter_mut().enumerate() {
                *x.1 = f(*x.1, (i.0, x.0))
            }
        }
        a
    }

    pub fn add_vec(&self, biases: &Matrix<1, M>) -> Matrix<N, M> {
        self.apply_pos(|v, (y, _)| v + biases[(y, 0)])
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
    pub fn max(&self) -> Option<f64> {
        let v = self.0
            .iter()
            .map(|vec| vec.iter().copied().fold(f64::NEG_INFINITY, f64::max))
            .fold(f64::NEG_INFINITY, f64::max);
        if v.is_finite() {
            Some(v)
        }else {
            None
        }
    }
    pub fn min(&self) -> Option<f64> {
        let v = self.0
        .iter()
        .map(|vec| vec.iter().copied().fold(f64::INFINITY, f64::min))
        .fold(f64::INFINITY, f64::min);
    if v.is_finite() {
        Some(v)
    }else {
        None
    }
}

    pub fn abs(&self) -> Self {
        self.apply(|x|x.abs())
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
impl_matrix_for_matrix!((Matrix),(Matrix));
impl_matrix_for_matrix!((Matrix),(&Matrix));
impl_matrix_for_matrix!((&Matrix),(Matrix));
impl_matrix_for_matrix!((&Matrix),(&Matrix));

impl_matrix_one!(&Matrix);
impl_matrix_one!(Matrix);
// impl<const N:usize,const M:usize> Mul<Matrix<N,M>> for 


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
