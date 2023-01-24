use std::fmt::Debug;

use crate::matrix::Matrix;

pub trait OptimizationArgs {}

pub trait Optimizer: Default + Debug {
    type Args: OptimizationArgs;
    type State<const N: usize, const M: usize>: OptimizerState<N, M, Self::Args>;

    fn get_state<const N: usize, const M: usize>() -> Self::State<N, M>;
}

pub trait OptimizerState<const N: usize, const M: usize, Args: OptimizationArgs>: Default {
    fn update(
        &mut self,
        w: &Matrix<N, M>,
        b: &Matrix<N, 1>,
        dw: &Matrix<N, M>,
        db: &Matrix<N, 1>,
        args: &Args,
    ) -> (Matrix<N, M>, Matrix<N, 1>);

    fn new() -> Self;
}

#[derive(Default, Debug)]
pub struct AdamOptimizer;

impl Optimizer for AdamOptimizer {
    type Args = AdamOptimizerArgs;

    type State<const N: usize, const M: usize> = AdamOptimizerState<N, M>;

    fn get_state<const N: usize, const M: usize>() -> Self::State<N, M> {
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

#[derive(Debug)]
pub struct AdamOptimizerState<const N: usize, const M: usize> {
    t: i32,
    m_dw: Matrix<N, M>,
    m_db: Matrix<N, 1>,
    v_dw: Matrix<N, M>,
    v_db: Matrix<N, 1>,
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

impl<const N: usize, const M: usize> OptimizerState<N, M, AdamOptimizerArgs>
    for AdamOptimizerState<N, M>
{
    fn update(
        &mut self,
        w: &Matrix<N, M>,
        b: &Matrix<N, 1>,
        dw: &Matrix<N, M>,
        db: &Matrix<N, 1>,
        args: &AdamOptimizerArgs,
    ) -> (Matrix<N, M>, Matrix<N, 1>) {
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

#[derive(Debug, Default)]
pub struct GradientDecentOptimizer;

impl Optimizer for GradientDecentOptimizer {
    type Args = GradientDecentOptimizerArgs;

    type State<const N: usize, const M: usize> = GradientDecentOptimizerState<N, M>;

    fn get_state<const N: usize, const M: usize>() -> Self::State<N, M> {
        GradientDecentOptimizerState::default()
    }
}

#[derive(Default,Debug)]
pub struct GradientDecentOptimizerState<const N: usize, const M: usize>;

impl<const N: usize, const M: usize> OptimizerState<N, M, GradientDecentOptimizerArgs>
    for GradientDecentOptimizerState<N, M>
{
    fn update(
        &mut self,
        w: &Matrix<N, M>,
        b: &Matrix<N, 1>,
        dw: &Matrix<N, M>,
        db: &Matrix<N, 1>,
        args: &GradientDecentOptimizerArgs,
    ) -> (Matrix<N, M>, Matrix<N, 1>) {
        (w - args.learning_rate * dw, b - args.learning_rate * db)
    }

    fn new() -> Self {
        Self {}
    }
}
pub struct GradientDecentOptimizerArgs {
    pub learning_rate: f64,
}

impl OptimizationArgs for GradientDecentOptimizerArgs {}
