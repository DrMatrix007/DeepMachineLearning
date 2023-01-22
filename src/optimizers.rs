use crate::matrix::Matrix;

pub trait OptimizationArgs {}

pub trait Optimizer : Default {
    type Args: OptimizationArgs;
    type State<const N:usize, const M:usize> : OptimizerState<N,M,Self::Args>;

    fn getOptimizer<const N:usize,const M:usize>() -> Self::State<N,M>;
}

pub trait OptimizerState<const N: usize, const M: usize,Args:OptimizationArgs>: Default {
    fn update(
        &mut self,
        w: &Matrix<N, M>,
        b: &Matrix<M, 1>,
        dw: &Matrix<N, M>,
        db: &Matrix<M, 1>,
        args: &Args,
    ) -> (Matrix<N, M>, Matrix<M, 1>);

    fn new() -> Self;
}


#[derive(Default)]
pub struct AdamOptimizer;

impl Optimizer for AdamOptimizer {
    type Args = AdamOptimizerArgs;

    type State<const N:usize, const M:usize>  = AdamOptimizerState<N,M>;

    fn getOptimizer<const N:usize,const M:usize>() -> Self::State<N,M> {
        Default::default()
    }
}



pub struct AdamOptimizerArgs {
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub eta: f64,
}

impl OptimizationArgs for AdamOptimizerArgs {}

pub struct AdamOptimizerState<const N: usize, const M: usize> {
    t: i32,
    m_dw: Matrix<N, M>,
    m_db: Matrix<M, 1>,
    v_dw: Matrix<N, M>,
    v_db: Matrix<M, 1>,
}

impl<const N: usize, const M: usize> Default for AdamOptimizerState<N, M> {
    fn default() -> Self {
        Self {
            t: 1,
            m_dw: Default::default(),
            m_db: Default::default(),
            v_dw: Default::default(),
            v_db: Default::default(),
        }
    }
}

impl<const N: usize, const M: usize> OptimizerState<N, M,AdamOptimizerArgs > for AdamOptimizerState<N, M> {
    fn update(
        &mut self,
        w: &Matrix<N, M>,
        b: &Matrix<M, 1>,
        dw: &Matrix<N, M>,
        db: &Matrix<M, 1>,
        args: &AdamOptimizerArgs,
    ) -> (Matrix<N, M>, Matrix<M, 1>) {
        self.m_dw = &self.m_dw * args.beta1 + (1.0 - args.beta1) * dw;
        self.m_db = &self.m_db * args.beta1 + (1.0 - args.beta1) * db;

        self.v_dw = &self.v_dw * args.beta2 + (1.0 - args.beta2) * dw.powi(2);
        self.m_db = &self.v_db * args.beta2 + (1.0 - args.beta2) * db;

        let m_dw_c = &self.m_dw / (1.0 - args.beta1.powi(self.t));
        let m_db_c = &self.m_db / (1.0 - args.beta1.powi(self.t));

        let v_dw_c = &self.v_dw / (1.0 - args.beta2.powi(self.t));
        let v_db_c = &self.v_db / (1.0 - args.beta2.powi(self.t));

        (
            w - (args.eta * (m_dw_c)).div_element_wise(&v_dw_c.sqrt().add(args.epsilon)),
            b - (args.eta * (m_db_c)).div_element_wise(&v_db_c.sqrt().add(args.epsilon)),
        )
    }

    fn new() -> Self {
        Default::default()
    }
}
