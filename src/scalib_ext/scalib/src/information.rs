use crate::lda::LDA;
use crate::rlda::{RLDAClusteredModel, RLDA};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use std::sync::Arc;

/// Implementation of information estimator
///
/// Estimates the bounds on the information (perceived, or training) based on the
/// bounded probabilities obtained with RLDAClusteredModel.
/// It assumes a model that profiles a uniform variable.
///
/// Perceived information is obtained using fresh traces, i.e. not used for training
/// Training information is obtained using traces used for training.
///
/// It estimates the information using
///     I(X,L) = H(X) + 1/n sum_i=0^n log2(Pr(X=x_i,l_i))
/// Pr(X=x_i,l_i) is given by RLDAClusteredModel.
/// Then, confidence intervals can be built using the deviation and the number of traces.
pub struct ItEstimator {
    /// RLDAClusteredModel used for the estimation of the PI bounds
    model: Arc<RLDAClusteredModel>,
    /// Maximum number of classes to take out of the clusters.
    max_popped_classes: usize,
    /// Number of traces accumulated.
    n: usize,
    /// Sum of the lower bound of probabilities
    sum_prs_l: f64,
    /// Sum of the upper bound of probabilities
    sum_prs_h: f64,
    /// Sum of the squared lower bound of probabilities
    sum_prs2_l: f64,
    /// Sum of the squared upper bound of probabilities
    sum_prs2_h: f64,
}

impl ItEstimator {
    pub fn new(model: Arc<RLDAClusteredModel>, max_popped_classes: usize) -> Self {
        Self {
            model: model.clone(),
            sum_prs_l: 0.0f64,
            sum_prs_h: 0.0f64,
            sum_prs2_l: 0.0f64,
            sum_prs2_h: 0.0f64,
            n: 0,
            max_popped_classes,
        }
    }
    /// Obtains the bounds and accumulates the probabilities.
    /// traces has shape (nt,ns) and labels has shape (nt)
    pub fn fit_u(&mut self, traces: ArrayView2<i16>, labels: ArrayView1<u64>) {
        let (prs_l, prs_h) = self
            .model
            .bounded_prs(traces, labels, self.max_popped_classes);
        let sum_sum_sq = |array: &Array1<f64>| {
            array
                .iter()
                .cloned()
                .map(f64::log2)
                .fold((0.0, 0.0), |(acc, acc_sq), x| (acc + x, acc_sq + x * x))
        };
        let (sum_l, sum_l_sq) = sum_sum_sq(&prs_l);
        let (sum_h, sum_h_sq) = sum_sum_sq(&prs_h);
        self.sum_prs_l += sum_l;
        self.sum_prs_h += sum_h;
        self.sum_prs2_l += sum_l_sq;
        self.sum_prs2_h += sum_h_sq;
        self.n += prs_l.shape()[0];
    }
    /// Get the estimation of the information based on the bounded probabilities.
    /// Returns an upper and lower bound on the estimated information
    pub fn get_information(&self) -> (f64, f64) {
        let nbits = self.model.coefs.shape()[1] as f64 - 1.0;
        let pi_l = nbits + self.sum_prs_l / self.n as f64;
        let pi_h = nbits + self.sum_prs_h / self.n as f64;
        return (pi_l, pi_h);
    }
    /// Get the standard deviation of the information for the lower and upper bound.
    /// Returns the lower and upper deviation, and the number of traces used for the estimation.
    pub fn get_deviation(&self) -> (f64, f64, usize) {
        let dev_l = f64::sqrt(
            self.sum_prs2_l / self.n as f64 - f64::powi(self.sum_prs_l / self.n as f64, 2),
        );
        let dev_h = f64::sqrt(
            self.sum_prs2_h / self.n as f64 - f64::powi(self.sum_prs_h / self.n as f64, 2),
        );
        return (dev_l, dev_h, self.n);
    }
}

pub enum LDAModel {
    LDA(Arc<LDA>),
    RLDA(Arc<RLDA>),
}
/// Implementation of information solver
/// Computes the perceived or training information exactly.
/// It computes the information using
///     I(X,L) = H(X) + 1/n sum_i=0^n log2(Pr(X=x_i,l_i))
/// This method is suited for models with a small number of classes
pub struct ItSolver {
    model: LDAModel,
    n: usize,
    sum_prs: f64,
    sum_prs2: f64,
}

impl ItSolver {
    pub fn new(model: LDAModel) -> Self {
        Self {
            model: match model {
                LDAModel::LDA(a) => LDAModel::LDA(a.clone()),
                LDAModel::RLDA(a) => LDAModel::RLDA(a.clone()),
            },
            n: 0,
            sum_prs: 0.0f64,
            sum_prs2: 0.0f64,
        }
    }

    pub fn fit_u(&mut self, traces: ArrayView2<i16>, var_num: Option<usize>) {
        let prs = match &self.model {
            LDAModel::LDA(m) => m.predict_proba(traces),
            LDAModel::RLDA(m) => {
                m.predict_proba(traces, var_num.expect("RLDA requires a variable index"))
            }
        };

        let sum_sum_sq = |array: &Array2<f64>| {
            array
                .iter()
                .cloned()
                .map(f64::log2)
                .fold((0.0, 0.0), |(acc, acc_sq), x| (acc + x, acc_sq + x * x))
        };
        let (sum, sum_sq) = sum_sum_sq(&prs);
        self.sum_prs += sum;
        self.sum_prs2 += sum_sq;
        self.n += prs.shape()[0];
    }

    pub fn get_information(&self) -> f64 {
        let entropy = match &self.model {
            LDAModel::LDA(m) => f64::log2(m.nc as f64),
            LDAModel::RLDA(m) => m.nb as f64,
        };
        entropy + (self.sum_prs / self.n as f64)
    }

    pub fn get_deviation(&self) -> (f64, usize) {
        let dev = self.sum_prs2 / self.n as f64 - f64::powi(self.sum_prs / self.n as f64, 2);
        (dev, self.n)
    }
}
