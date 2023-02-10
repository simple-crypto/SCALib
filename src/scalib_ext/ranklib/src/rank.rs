use itertools::Itertools;

use crate::histogram::Histogram;
use crate::{RankError, RankEstimation};

#[cfg(fuzzing)]
const MAX_NB_BINS: usize = 1 << 14;
#[cfg(not(fuzzing))]
const MAX_NB_BINS: usize = 1 << 29;

fn cost2bin_f(cost: f64, bin_size: f64) -> f64 {
    cost / bin_size
}
fn cost2bin(cost: f64, bin_size: f64) -> usize {
    cost2bin_f(cost, bin_size).round() as usize
}

/// A rank estimation problem.
///
/// Invariants:
/// * costs.len() == real_key.len() != 0
/// * real_key[i] < costs[i].len()
/// The rank of the key is defined as the number of integer arrays x such that
/// sum_i costs[i][x[i]] <= sum_i costs[i][real_key[i]].
///
/// The real_key has thus rank 1 if it has the minimum cost.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub(crate) struct RankProblem {
    /// For each sub-key, the cost (e.g. negative log-likelihood) of each possible value for that
    /// sub-key.
    pub costs: Vec<Vec<f64>>,
    /// Real key to rank. Value of each of the subkeys.
    pub real_key: Vec<usize>,
}

impl RankProblem {
    pub fn new(
        costs: impl Into<Vec<Vec<f64>>>,
        real_key: impl Into<Vec<usize>>,
    ) -> Result<Self, RankError> {
        let mut res = Self {
            costs: costs.into(),
            real_key: real_key.into(),
        };
        res.assert_valid()?;
        res.normalize();
        res.assert_valid()?;
        return Ok(res);
    }
    pub fn assert_valid(&self) -> Result<(), RankError> {
        if self.costs.len() != self.real_key.len() {
            Err("Not same length cost and key")?;
        } else if self.real_key.len() == 0 {
            Err("Empty key")?;
        } else if self
            .real_key
            .iter()
            .zip(self.costs.iter())
            .any(|(k, sc)| *k >= sc.len())
        {
            Err("Key value too large wrt cost")?;
        } else if !self
            .costs
            .iter()
            .flat_map(|sc| sc.iter())
            .all(|s| s.is_finite())
        {
            Err("Non-finite cost")?;
        } else if !self
            .costs
            .iter()
            .map(|sc| sc.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap())
            .sum::<f64>()
            .is_finite()
        {
            Err("Infinite max total cost")?;
        }
        Ok(())
    }
    /// Merge merge_nb subkeys together.
    pub fn merge(&self, merge_nb: usize) -> Self {
        let sub_iter = self
            .costs
            // Each chunk is merged in a single new subkey
            .chunks(merge_nb)
            .zip(self.real_key.chunks(merge_nb));
        Self::merge_inner(sub_iter)
    }
    /// Merge as much as possible, while keeping cost array of at most max_len elements
    pub fn auto_merge(&self, max_len: usize) -> Self {
        let mut rem_costs: &[_] = &self.costs;
        let mut rem_subkeys: &[_] = &self.real_key;
        let sub_iter = std::iter::from_fn(|| {
            if rem_costs.is_empty() {
                None
            } else {
                let mut cum_len = 1;
                let n = rem_costs
                    .iter()
                    .take_while(|c| {
                        cum_len *= c.len();
                        cum_len <= max_len
                    })
                    .count();
                let (yield_costs, rc) = rem_costs.split_at(n);
                let (yield_subkeys, rk) = rem_subkeys.split_at(n);
                rem_costs = rc;
                rem_subkeys = rk;
                Some((yield_costs, yield_subkeys))
            }
        });
        Self::merge_inner(sub_iter)
    }
    /// Generate a merged problem from a (cost_chunk, subkeys_chunk) iterator.
    fn merge_inner<'a>(x: impl Iterator<Item = (&'a [Vec<f64>], &'a [usize])>) -> Self {
        let (costs, real_key) = x
            .map(|(costs_chunk, real_key_chunk)| {
                (
                    // Cartesian product of costs, then summed.
                    costs_chunk
                        .iter()
                        .multi_cartesian_product()
                        .map(|x| x.into_iter().sum())
                        .collect(),
                    // Index of new key, matching cartesian product order: first sub-keys are most
                    // significant.
                    costs_chunk
                        .iter()
                        .zip(real_key_chunk.iter())
                        .fold(0, |acc, (sc, k)| acc * sc.len() + k),
                )
            })
            .unzip::<_, _, Vec<_>, Vec<_>>();
        return Self { costs, real_key };
    }
    /// Iterate over the cost for each of the subkey of the real key.
    fn key_costs<'a>(&'a self) -> impl Iterator<Item = f64> + 'a {
        self.costs
            .iter()
            .zip(self.real_key.iter())
            .map(|(sc, sk)| sc[*sk])
    }
    /// Offset all costs such that the minimum cost for each subkey is 0.0.
    fn normalize(&mut self) {
        self.costs.iter_mut().for_each(|sc| {
            sc.iter()
                .copied()
                .min_by(|a, b| a.partial_cmp(b).expect("No NaN"))
                .map(|min| sc.iter_mut().for_each(|s| *s -= min));
        });
    }
    /// total key cost
    fn key_cost(&self) -> f64 {
        self.key_costs().sum()
    }
    /// Size of bins to use to reach nb_bins bins before the real key
    fn bin_size(&self, nb_bins: usize) -> Result<f64, RankError> {
        // Bins are indexed from 0 to n_bins-1 (included), and bin with index i covers the span
        // [i*bin_size, (i+1)*bin_size[, hence the last bin covers the span
        // [(nb_bins-1)*bin_size, nb_bins*bin_size[.
        // The histogram should contain the bin corresponding to the key cost, and
        // margin = ceil(nb_subkeys/2.0) bins after it.
        // Hence key_cost < (nb_bins-margin)*bin_size.
        // We take key_cost = (nb_bins-margin-1)*bin_size.
        let nb_subkeys = self.costs.len();
        let margin = (nb_subkeys / 2) + (nb_subkeys & 0x1);
        let effective_nb_bins = nb_bins.checked_sub(margin + 1).ok_or("nb_bins too small")?;
        if effective_nb_bins == 0 {
            Err("nb_bins too small")?;
        }
        return Ok(self.key_cost() / (effective_nb_bins as f64));
    }
    /// Create a convolved histogram with nb_bins bins before the real key.
    /// Return the histogram and its bin size.
    fn build_histogram<H: Histogram>(&self, nb_bins: usize) -> Result<(H, f64), RankError> {
        let bin_size = self.bin_size(nb_bins)?;
        let hist = self
            .costs
            .iter()
            .map(|costs| {
                H::from_elems(
                    nb_bins,
                    costs
                        .iter()
                        .copied()
                        .map(|s| cost2bin(s, bin_size))
                        .filter(|bin| *bin < nb_bins),
                )
            })
            .fold(None, |acc: Option<H>, hist| {
                acc.map(|x| x.convolve(&hist)).or(Some(hist))
            })
            .expect("Some subkey");
        return Ok((hist, bin_size));
    }
    /// Get the exact rank through brute-force cost computation for all keys.
    pub fn naive_rank(&self) -> f64 {
        let problem = self.merge(self.costs.len());
        assert_eq!(problem.costs.len(), 1);
        return problem.costs[0]
            .iter()
            .filter(|sc| **sc <= problem.costs[0][problem.real_key[0]])
            .count() as f64;
    }
    /// Estimate rank using a convolution of histograms
    pub fn rank_hist<H: Histogram>(&self, nb_bins: usize) -> Result<RankEstimation, RankError> {
        if nb_bins < 1 || nb_bins > MAX_NB_BINS {
            Err("Bin count out of limits.")?;
        }
        let (hist, bin_size): (H, _) = self.build_histogram(nb_bins)?;
        return Ok(rank_in_histogram(
            self.key_costs().sum::<f64>(),
            &hist,
            bin_size,
            self.costs.len(),
        ));
    }
}

/// Count elements with lower cost
fn rank_in_histogram<H: Histogram>(
    real_key_cost: f64,
    histo: &H,
    bin_size: f64,
    nb_subkeys: usize,
) -> RankEstimation {
    // Values can be shifted by at most nb_subkeys/2 bins.
    // We add a small amount to compensate for rounding errors.
    let margin = (nb_subkeys as f64) / 2.0 + 1e-20;
    let coefs = histo.coefs_f64();
    // bound by histo.len is needed due to greedy histogram growth
    let bin_real_key = std::cmp::min(coefs.len() - 1, cost2bin(real_key_cost, bin_size));
    let bin_bound_max = std::cmp::min(
        coefs.len() - 1,
        (cost2bin_f(real_key_cost, bin_size) + margin).floor() as usize,
    );
    let bin_bound_min = std::cmp::min(
        coefs.len() - 1,
        (cost2bin_f(real_key_cost, bin_size) - margin).ceil() as usize,
    );
    debug_assert!(bin_bound_min <= bin_real_key);
    debug_assert!(bin_real_key <= bin_bound_max);
    let sum_hist = |end| coefs[..end].iter().copied().sum::<f64>();
    // We must add one to the minimum rank to include the real key.
    let rank_min = 1.0 + sum_hist(bin_bound_min);
    // It is already included in the max.
    // For the est, it might not be included, if the real key is between the real bin and the max.
    return RankEstimation::new(
        rank_min,
        rank_min.max(sum_hist(bin_real_key) + (coefs[bin_real_key] / 2.0).ceil()),
        sum_hist(bin_bound_max + 1),
    );
}

#[cfg(test)]
mod tests {
    use super::RankProblem;
    #[test]
    fn test_merge_mat() {
        let costs = vec![vec![0.0, 1.0], vec![2.0, 3.0], vec![10.0, 11.0]];
        let sp = RankProblem::new(costs.clone(), vec![0, 0, 0].as_slice()).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let real_key = vec![i, j, 0];
                let problem = RankProblem::new(costs.clone(), real_key).unwrap();
                let problem = problem.merge(2);
                assert_eq!(problem.real_key, vec![i * 2 + j, 0]);
                assert_eq!(
                    sp.costs[0][i] + sp.costs[1][j],
                    problem.costs[0][problem.real_key[0]]
                );
                assert_eq!(sp.costs[2][i], problem.costs[1][i]);
            }
        }
    }
}
