use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Range, Sub, SubAssign},
};

use rand::{
    distributions::{
        uniform::{SampleRange, SampleUniform},
        Standard,
    },
    prelude::Distribution,
    Rng,
};

pub trait Number:
    Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Default
    + AddAssign<Self>
    + SubAssign<Self>
    + DivAssign<Self>
    + MulAssign<Self>
    + Display
    + Copy
    + Clone
    + From<f64>
    + Into<f64>
    + PartialOrd<Self>
    + SampleUniform
{
}
macro_rules! impl_matrix {
    ($a:ty,$i:tt) => {
        impl<
        const N: usize,
        const M: usize,
        T:Number
        > $a for Matrix<N,M,T>
        $i

    };

}
impl<
        T: Add<Output = Self>
            + Sub<Output = Self>
            + Mul<Output = Self>
            + Div<Output = Self>
            + Default
            + AddAssign<Self>
            + SubAssign<Self>
            + DivAssign<Self>
            + MulAssign<Self>
            + Display
            + Copy
            + Clone
            + From<f64>
            + Into<f64>
            + PartialOrd<Self>
            + SampleUniform,
    > Number for T
where
    T: From<f64>,
    Standard: Distribution<T>,
    T: SampleUniform,
    Range<T>: SampleRange<T>,
{
}

#[derive(Clone, Copy, Debug)]
pub struct Matrix<const N: usize, const M: usize, T: Number = f64>(pub [[T; N]; M]);

impl<const N: usize, const M: usize, T: Number> Matrix<N, M, T> {
    pub fn new(arrs: [[impl Into<T>; N]; M]) -> Self {
        Self(arrs.map(|x| x.map(|y| y.into())))
    }

    pub fn apply(&self, f: impl Fn(T) -> T) -> Self {
        let f = &f;
        Matrix::<N, M, T>::new(self.0.map(|x| x.map(f)))
    }
    pub fn apply_mut(&self, mut f: impl FnMut(T) -> T) -> Self {
        let mut a = *self;
        for i in &mut a.0 {
            for x in i {
                *x = f(*x)
            }
        }
        a
    }
    pub fn random(l: T, h: T) -> Self {
        let mut rng = rand::thread_rng();
        Matrix::default().apply_mut(|_| rng.gen_range(l..h))
    }
    pub fn transpose(&self) -> Matrix<M, N, T> {
        let mut ans = Matrix::default();
        for x in 0..N {
            for y in 0..M {
                ans[(x, y)] = self[(y, x)];
            }
        }
        ans
    }
    pub fn add_vec(&self,rhs:Matrix<1,M,T>) -> Self {
        let mut ans = Matrix::<N, M, T>::default();
        for x in 0..N {
            for y in 0..M {
                ans.0[y][x] = self.0[y][x] + rhs.0[y][0];
            }
        }
        ans
    }
    pub fn element_wise_product(&self,rhs:Matrix<N,M,T>) -> Self {
        let mut ans = Matrix::default();
        for i in 0..N {
            for j in 0..M {
                ans[(j,i)] = self[(j,i)]*rhs[(j,i)];
            }
        }
        ans
    }
    pub fn mul_arg(&self,rhs:T) -> Self {
        self.apply(|x|x*rhs)
    }

    pub fn div_arg(&self,rhs:T) -> Self {
        self.apply(|x|x/rhs)
    }
}
impl_matrix!(Default, {
    fn default() -> Self {
        Self([[T::zero(); N]; M])
    }
});
impl_matrix!(Index<(usize, usize)>, {
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.0[index.0][index.1]
    }
});
impl_matrix!(IndexMut<(usize, usize)>, {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.0[index.0][index.1]
    }
});
impl_matrix!(Add<Self>, {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let mut ans = Matrix::<N, M, T>::default();
        for x in 0..N {
            for y in 0..M {
                ans.0[y][x] = self.0[y][x] + rhs.0[y][x];
            }
        }
        ans
    }
});


impl_matrix!(Sub<Self>, {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut ans = Matrix::<N, M, T>::default();
        for x in 0..N {
            for y in 0..M {
                ans.0[y][x] = self.0[y][x] - rhs.0[y][x];
            }
        }
        ans
    }
});

impl<const N:usize,const M:usize,T:Number> Mul<T> for Matrix<N,M,T> {
    type Output=Matrix<N,M,T>;

    fn mul(self, rhs: T) -> Self::Output {
        self.apply(|x|x*rhs)
    }
}

impl<const N:usize,const M:usize,T:Number> Div<T> for Matrix<N,M,T> {
    type Output=Matrix<N,M,T>;

    fn div(self, rhs: T) -> Self::Output {
        self.apply(|x|x/rhs)
    }
}

impl<const N: usize, const M: usize, const K: usize, T: Number> Mul<Matrix<M, K, T>>
    for Matrix<N, M, T>
{
    type Output = Matrix<N, K, T>;
    fn mul(self, rhs: Matrix<M, K, T>) -> Self::Output {
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

impl_matrix!(Display, {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "\n[")?;

        for y in 0..M {
            write!(f, "[")?;

            for x in 0..N {
                write!(f, "{}, ", self[(y, x)])?;
            }

            write!(f, "{}", if y != M - 1 { "],\n " } else { "]" })?;
        }
        write!(f, "]")
    }
});
impl_matrix!(From<[[T; N]; M]>, {
    fn from(a: [[T; N]; M]) -> Self {
        Self(a)
    }
});

trait Zero {
    fn zero() -> Self;
}

impl<T: From<f64>> Zero for T {
    fn zero() -> Self {
        (0.).into()
    }
}

pub trait ToMatrix<const N: usize, const M: usize, T: Number> {
    fn to_matrix(&self) -> Matrix<N, M, T>;
}
impl<const N: usize, const M: usize, T: Number> ToMatrix<N, M, T> for [[T; N]; M] {
    fn to_matrix(&self) -> Matrix<N, M, T> {
        Matrix(*self)
    }
}
