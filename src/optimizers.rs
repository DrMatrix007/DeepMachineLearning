use crate::matrix::Matrix;

pub trait OptimizationArgs {}

pub trait Optimizer<const N: usize, const M: usize, Args: OptimizationArgs>: Default {
    fn update(
        &mut self,
        w: &Matrix<N, M>,
        b: &Matrix<M, 1>,
        dw: &Matrix<N, M>,
        db: &Matrix<M, 1>,
    ) -> (Matrix<N, M>, Matrix<M, 1>);
}

pub struct AdamOptimizerArgs;

impl OptimizationArgs for AdamOptimizerArgs {}

pub struct AdamOptimizer<const N: usize, const M: usize> {
    t: i32,
    m_dw: Matrix<N, M>,
    m_db: Matrix<M, 1>,
    v_dw: Matrix<N, M>,
    v_db: Matrix<M, 1>,
}

impl<const N: usize, const M: usize> Default for AdamOptimizer<N, M> {
    fn default() -> Self {
        Self {
            t: 1,
            m_dw: Default::default(),
            m_db: Default::default(),
            v_dw: Default::default(),
            v_db: Default::default(),
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            eta: 0.01,
        }
    }
}

impl<const N: usize, const M: usize> Optimizer<N, M, AdamOptimizerArgs> for AdamOptimizer<N, M> {
    fn update(
        &mut self,
        w: &Matrix<N, M>,
        b: &Matrix<M, 1>,
        dw: &Matrix<N, M>,
        db: &Matrix<M, 1>,
    ) -> (Matrix<N, M>, Matrix<M, 1>) {
        self.m_dw = &self.m_dw * self.beta1 + (1.0 - self.beta1) * dw;
        self.m_db = &self.m_db * self.beta1 + (1.0 - self.beta1) * db;

        self.v_dw = &self.v_dw * self.beta2 + (1.0 - self.beta2) * dw.powi(2);
        self.m_db = &self.v_db * self.beta2 + (1.0 - self.beta2) * db;

        let m_dw_c = &self.m_dw / (1.0 - self.beta1.powi(self.t));
        let m_db_c = &self.m_db / (1.0 - self.beta1.powi(self.t));

        let v_dw_c = &self.v_dw / (1.0 - self.beta2.powi(self.t));
        let v_db_c = &self.v_db / (1.0 - self.beta2.powi(self.t));

        (
            w - (self.eta * (m_dw_c)).div_element_wise(&v_dw_c.sqrt().add(self.epsilon)),
            b - (self.eta * (m_db_c)).div_element_wise(&v_db_c.sqrt().add(self.epsilon)),
        )
    }
}
