use std::{
    fmt::Display,
    ops::{Index, IndexMut},
    vec::IntoIter,
};

#[derive(Clone, Debug)]
pub struct Matrix<const N: usize, const M: usize>(pub(self) Vec<f64>);

impl<const N: usize, const M: usize> FromIterator<Matrix<N, 1>> for Matrix<N, M> {
    fn from_iter<T: IntoIterator<Item = Matrix<N, 1>>>(iter: T) -> Self {
        Matrix(iter.into_iter().flat_map(|x| x.0).collect())
    }
}

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
        Self::zeros()
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
            .map(|(pos, x)| ((pos % N, pos / N), x))
    }

    pub fn trasnpose(&self) -> Matrix<M, N> {
        let mut ans = Matrix::default();
        for (pos, i) in ans.iter_mut() {
            *i = self[(pos.1, pos.0)];
        }
        ans
    }
    pub fn generate(f: impl FnMut() -> f64) -> Self {
        Self(std::iter::repeat_with(f).take(N * M).collect())
    }
    pub fn ones() -> Matrix<N, M> {
        Self::generate(|| 1.0)
    }

    pub fn element_wise_product(&self, x: &Matrix<N, M>) -> Matrix<N, M> {
        let mut ans = Matrix::default();

        for (pos, i) in ans.iter_mut() {
            *i = x[pos] * self[pos];
        }
        ans
    }

    pub fn zeros() -> Matrix<N, M> {
        Self::generate(|| 0.0)
    }

    pub fn map(&self, mut f: impl FnMut((usize, usize), f64) -> f64) -> Matrix<N, M> {
        let mut ans = self.clone();
        for (pos, val) in ans.iter_mut() {
            *val = f(pos, *val);
        }
        ans
    }
    pub fn max(&self) -> f64 {
        self.iter().map(|(_, y)| *y).fold(f64::NAN, f64::max)
    }
    pub fn sub_matrices_vertically(&self) -> impl Iterator<Item = Matrix<N, 1>> + '_ {
        (0..M).map(|offset| {
            let mut v = self
                .0
                .iter()
                .skip(offset * N)
                .take(N)
                .cloned()
                .collect::<Vec<_>>();
            v.resize(N, 0.0);
            Matrix::<N, 1>(v)
        })
    }

    pub fn sub(&self, i: usize) -> Matrix<N, 1> {
        if i < M {
            Matrix(self.0.iter().skip(i * N).take(N).copied().collect())
        } else {
            panic!("access bounds out of range");
        }
    }
    pub fn set_sub(&mut self, i: usize, m: &Matrix<N, 1>) {
        for x in 0..N {
            self[(x, i)] = m[(x, 0)];
        }
    }

    pub fn powi(&self, arg: i32) -> Self {
        self.map(|_, y| y.powi(arg))
    }
    pub fn powf(&self, arg: f64) -> Self {
        self.map(|_, y| y.powf(arg))
    }
    pub fn add(&self, arg: f64) -> Self {
        self.map(|_, x| x + arg)
    }
    pub fn sqrt(&self) -> Self {
        self.map(|_, x| x.sqrt())
    }
    pub fn mul_element_wise(&self, m: &Matrix<N, M>) -> Self {
        self.map(|pos, x| x * m[pos])
    }
    pub fn div_element_wise(&self, m: &Matrix<N, M>) -> Self {
        self.map(|pos, x| x / m[pos])
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
impl<T, const N: usize, const M: usize> From<[[T; N]; M]> for Matrix<N, M>
where
    T: Into<f64> + Clone,
{
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

    impl<const N: usize, const M: usize, const K: usize> Mul<&Matrix<K, N>> for &Matrix<N, M> {
        type Output = Matrix<K, M>;

        fn mul(self, rhs: &Matrix<K, N>) -> Self::Output {
            let mut ans = Matrix::default();

            for (pos, x) in ans.iter_mut() {
                *x = 0.0;
                for n in 0..N {
                    *x += self[(n, pos.1)] * rhs[(pos.0, n)];
                }
            }
            ans
        }
    }
    impl<const N: usize, const M: usize, const K: usize> Mul<Matrix<K, N>> for &Matrix<N, M> {
        type Output = Matrix<K, M>;

        fn mul(self, rhs: Matrix<K, N>) -> Self::Output {
            self * &rhs
        }
    }
    impl<const N: usize, const M: usize, const K: usize> Mul<&Matrix<K, N>> for Matrix<N, M> {
        type Output = Matrix<K, M>;

        fn mul(self, rhs: &Matrix<K, N>) -> Self::Output {
            &self * rhs
        }
    }
    impl<const N: usize, const M: usize, const K: usize> Mul<Matrix<K, N>> for Matrix<N, M> {
        type Output = Matrix<K, M>;

        fn mul(self, rhs: Matrix<K, N>) -> Self::Output {
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
mod div {
    use std::ops::Div;

    use super::Matrix;

    impl<const N: usize, const M: usize> Div<f64> for &Matrix<N, M> {
        type Output = Matrix<N, M>;

        fn div(self, rhs: f64) -> Self::Output {
            let mut ans = Matrix::default();
            for (pos, i) in ans.iter_mut() {
                *i = self[pos] / rhs;
            }
            ans
        }
    }
    impl<const N: usize, const M: usize> Div<f64> for Matrix<N, M> {
        type Output = Matrix<N, M>;

        fn div(self, rhs: f64) -> Self::Output {
            &self / rhs
        }
    }
}
