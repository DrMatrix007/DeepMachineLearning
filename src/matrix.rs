use std::{
    fmt::Display,
    ops::{Index, IndexMut},
    vec::IntoIter,
};

#[derive(Clone,Debug)]
pub struct Matrix<const N: usize, const M: usize>(pub(self) Vec<f64>);

impl<const N: usize, const M: usize> Display for Matrix<N, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;

        for (pos, x) in self.iter() {
            if pos.0 == 0 {
                write!(f, "[{},", x)?
            } else if pos.0 == M - 1 {
                if pos.1 == N - 1 {
                    write!(f, "{}]", x)?;
                } else {
                    writeln!(f, "{}]", x)?;
                }
            } else {
                write!(f, "{},", x)?;
            }
        }

        write!(f, "]")
    }
}

impl<const N: usize, const M: usize> Default for Matrix<N, M> {
    fn default() -> Self {

        Self(std::iter::repeat(0.0).take(N * M).collect())
    }
}

impl<const N: usize, const M: usize> Matrix<N, M> {
    pub fn into_iter(&self) -> IntoIter<((usize, usize), &f64)> {
        self.0
            .iter()
            .enumerate()
            .map(|(pos, x)| ((pos % N, pos / N), x))
            .collect::<Vec<_>>()
            .into_iter()
    }
    pub fn iter(&self) -> impl Iterator<Item = ((usize, usize), &f64)> {
        self.0
            .iter()
            .enumerate()
            .map(|(pos, x)| ((pos % M, pos / M), x))
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = ((usize, usize), &mut f64)> {
        self.0
            .iter_mut()
            .enumerate()
            .map(|(pos, x)| ((pos %N, pos/N), x))
    }

    pub fn trasnpose(&self) -> Matrix<M, N> {
        let mut ans = Matrix::default();
        for (pos, i) in ans.iter_mut() {
            *i = self[(pos.1, pos.0)];
        }
        ans
    }

    pub fn ones() -> Matrix<N, M> {
        let v = std::iter::repeat(1.0).take(N * M).collect::<Vec::<_>>();
        Matrix(v)
    }

    pub(crate) fn element_wise_product(&self, x: &Matrix<N, M>) -> Matrix<N, M> {
        let mut ans = Matrix::default();

        for(pos,i) in ans.iter_mut() {
            *i = x[pos] * self[pos];
        }
        ans
    }
}

impl<const N: usize, const M: usize> Index<(usize, usize)> for Matrix<N, M> {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.0[index.0 + index.1 * N]
    }
}

impl<const N: usize, const M: usize> IndexMut<(usize, usize)> for Matrix<N, M> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.0[index.0 + index.1 * N]
    }
}
impl<T, const N: usize, const M: usize> From<[[T; N]; M]> for Matrix<N, M> where T: Into<f64> + Clone {
    fn from(val: [[T; N]; M]) -> Self {
        Matrix(val.concat().into_iter().map(|x| x.into()).collect())
    }
}

// impl<T: Into<f64> + Clone, const N: usize> ToMatrix<N, 1> for [T; N] {
//     fn to_matrix(self) -> Matrix<N, 1> {
//         Matrix(self.concat().into_iter().map(|x| x.into()).collect())
//     }
// }

mod add {
    use std::ops::Add;

    use super::Matrix;
    impl<const N: usize, const M: usize> Add<Self> for &Matrix<N, M> {
        type Output = Matrix<N, M>;

        fn add(self, rhs: Self) -> Self::Output {
            let mut ans = Matrix::default();
            for (a, b) in ans.iter_mut() {
                *b = self[a] + rhs[a];
            }
            ans
        }
    }
    impl<const N: usize, const M: usize> Add<Matrix<N, M>> for Matrix<N, M> {
        type Output = Matrix<N, M>;

        fn add(self, rhs: Matrix<N, M>) -> Self::Output {
            &self + &rhs
        }
    }
    impl<const N: usize, const M: usize> Add<&Matrix<N, M>> for Matrix<N, M> {
        type Output = Matrix<N, M>;

        fn add(self, rhs: &Matrix<N, M>) -> Self::Output {
            &self + rhs
        }
    }
    impl<const N: usize, const M: usize> Add<Matrix<N, M>> for &Matrix<N, M> {
        type Output = Matrix<N, M>;

        fn add(self, rhs: Matrix<N, M>) -> Self::Output {
            self + &rhs
        }
    }
}
mod sub {
    use std::ops::Sub;

    use super::Matrix;

    impl<const N: usize, const M: usize> Sub<&Matrix<N, M>> for &Matrix<N, M> {
        type Output = Matrix<N, M>;

        fn sub(self, rhs: &Matrix<N, M>) -> Self::Output {
            let mut ans = Matrix::default();
            for (a, b) in ans.iter_mut() {
                *b = self[a] - rhs[a];
            }
            ans
        }
    }
    impl<const N: usize, const M: usize> Sub<Matrix<N, M>> for Matrix<N, M> {
        type Output = Matrix<N, M>;

        fn sub(self, rhs: Matrix<N, M>) -> Self::Output {
            &self - &rhs
        }
    }
    impl<const N: usize, const M: usize> Sub<&Matrix<N, M>> for Matrix<N, M> {
        type Output = Matrix<N, M>;

        fn sub(self, rhs: &Matrix<N, M>) -> Self::Output {
            &self - rhs
        }
    }
    impl<const N: usize, const M: usize> Sub<Matrix<N, M>> for &Matrix<N, M> {
        type Output = Matrix<N, M>;

        fn sub(self, rhs: Matrix<N, M>) -> Self::Output {
            self - &rhs
        }
    }
}
mod mul {
    use std::ops::Mul;

    use super::Matrix;

    impl<const N: usize, const M: usize, const K: usize> Mul<&Matrix<M, K>> for &Matrix<N, M> {
        type Output = Matrix<N, K>;

        fn mul(self, rhs: &Matrix<M, K>) -> Self::Output {
            let mut ans = Matrix::default();

            for (pos, x) in ans.iter_mut() {
                *x = 0.0;
                for m in 0..M {
                    *x += self[(pos.0, m)] * rhs[(m, pos.1)];
                }
            }
            ans
        }
    }
    impl<const N: usize, const M: usize, const K: usize> Mul<Matrix<M, K>> for &Matrix<N, M> {
        type Output = Matrix<N, K>;

        fn mul(self, rhs: Matrix<M, K>) -> Self::Output {
            self * &rhs
        }
    }
    impl<const N: usize, const M: usize, const K: usize> Mul<&Matrix<M, K>> for Matrix<N, M> {
        type Output = Matrix<N, K>;

        fn mul(self, rhs: &Matrix<M, K>) -> Self::Output {
            &self * rhs
        }
    }
    impl<const N: usize, const M: usize, const K: usize> Mul<Matrix<M, K>> for Matrix<N, M> {
        type Output = Matrix<N, K>;

        fn mul(self, rhs: Matrix<M, K>) -> Self::Output {
            &self * &rhs
        }
    }

    impl<const N: usize, const M: usize> Mul<f64> for &Matrix<N, M> {
        type Output = Matrix<N, M>;

        fn mul(self, rhs: f64) -> Self::Output {
            let mut ans = Matrix::default();
            for (pos, i) in ans.iter_mut() {
                *i = self[pos] * rhs;
            }
            ans
        }
    }
    impl<const N: usize, const M: usize> Mul<f64> for Matrix<N, M> {
        type Output = Matrix<N, M>;

        fn mul(self, rhs: f64) -> Self::Output {
            &self * rhs
        }
    }
    impl<const N: usize, const M: usize> Mul<Matrix<N, M>> for f64 {
        type Output = Matrix<N, M>;

        fn mul(self, rhs: Matrix<N, M>) -> Self::Output {
            rhs * self
        }
    }
    impl<const N: usize, const M: usize> Mul<&Matrix<N, M>> for f64 {
        type Output = Matrix<N, M>;

        fn mul(self, rhs: &Matrix<N, M>) -> Self::Output {
            rhs * self
        }
    }
}
