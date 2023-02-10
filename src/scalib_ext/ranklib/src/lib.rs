mod histogram;
mod rank;

#[derive(Debug)]
pub struct RankError {
    s: String,
}
impl<'a> From<&'a str> for RankError {
    fn from(s: &'a str) -> Self {
        Self { s: s.into() }
    }
}

impl std::fmt::Display for RankError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Ranking error: {}", self.s)
    }
}

impl std::error::Error for RankError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct RankEstimation {
    pub min: f64,
    pub est: f64,
    pub max: f64,
}
impl RankEstimation {
    pub fn new(min: f64, est: f64, max: f64) -> Self {
        debug_assert!(min <= est, "{min:?} {est:?} {max:?}");
        debug_assert!(est <= max, "{min:?} {est:?} {max:?}");
        Self { min, est, max }
    }
    pub fn contains(&self, rank: f64) -> bool {
        self.min <= rank && rank <= self.max
    }
    fn margin(&self) -> f64 {
        self.max / self.min
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum RankingMethod {
    Naive,
    #[cfg(feature = "hellib")]
    Hellib,
    Hist,
    #[cfg(feature = "ntl")]
    HistBigNum,
}

impl RankingMethod {
    fn rank_inner(
        &self,
        problem: &rank::RankProblem,
        nb_bin: usize,
        merge: Option<usize>,
    ) -> Result<RankEstimation, RankError> {
        let merged_problem = if let Some(merge) = merge {
            if merge < 1 || merge > problem.costs.len() {
                Err("Merge value not supported.")?;
            }
            problem.merge(merge)
        } else {
            problem.auto_merge(1 << 16)
        };
        match self {
            RankingMethod::Naive => {
                let rank = problem.naive_rank();
                Ok(RankEstimation::new(rank, rank, rank))
            }
            #[cfg(feature = "hellib")]
            RankingMethod::Hellib => {
                rank_hellib(&problem.costs, &problem.real_key, nb_bin, merge.unwrap())
            }
            RankingMethod::Hist => merged_problem.rank_hist::<histogram::F64Hist>(nb_bin),
            #[cfg(feature = "ntl")]
            RankingMethod::HistBigNum => merged_problem.rank_hist::<histogram::BigNumHist>(nb_bin),
        }
    }
    pub fn rank_nbin(
        &self,
        costs: &[Vec<f64>],
        key: &[usize],
        nb_bin: usize,
        merge: Option<usize>,
    ) -> Result<RankEstimation, RankError> {
        #[cfg(feature = "hellib")]
        let problem = if *self == RankingMethod::Hellib {
            rank::RankProblem {
                costs: costs.into(),
                real_key: key.into(),
            }
        } else {
            rank::RankProblem::new(costs, key)?
        };
        #[cfg(not(feature = "hellib"))]
        let problem = rank::RankProblem::new(costs, key)?;
        self.rank_inner(&problem, nb_bin, merge)
    }
    pub fn rank_accuracy(
        &self,
        costs: &[Vec<f64>],
        key: &[usize],
        acc: f64,
        merge: Option<usize>,
        max_nb_bin: usize,
    ) -> Result<RankEstimation, RankError> {
        let problem = rank::RankProblem::new(costs, key)?;
        for nb_bin in (4..).map(|i| 1 << i).take_while(|x| *x < max_nb_bin) {
            let res = self.rank_inner(&problem, nb_bin, merge)?;
            if res.margin() <= acc {
                return Ok(res);
            }
        }
        // We do best-effort. If we cannot reach desired accuracy, we still return the best result
        // we can have.
        return self.rank_inner(&problem, max_nb_bin, merge);
    }
}

#[cfg(feature = "hellib")]
fn rank_hellib(
    costs: &[Vec<f64>],
    key: &[usize],
    nb_bin: usize,
    merge: usize,
) -> Result<RankEstimation, RankError> {
    // Hellib only supports fixed key length and key size, I believe (otherwise you have to
    // recompile).
    if key.len() != 16 {
        Err("There must be 16 subkeys for hellib.")?;
    }
    if costs.len() != key.len() {
        Err("There must be as many costs as subkeys.")?;
    }
    if costs.iter().any(|c| c.len() != 256) {
        Err("All subkeys must be 8 bits, hence 256 costs.")?;
    }
    let scores: Vec<Vec<f64>> = costs
        .iter()
        .map(|c| c.iter().map(|x| -*x).collect())
        .collect();
    let sc_ptr: Vec<*const f64> = scores.iter().map(|x| x.as_ptr()).collect();
    let key: Vec<i32> = key.iter().map(|k| *k as i32).collect();
    let mut min = 0.0;
    let mut est = 0.0;
    let mut max = 0.0;
    unsafe {
        hel_struct::hel_top(
            sc_ptr.as_ptr(),
            key.as_ptr(),
            nb_bin as i32,
            merge as i32,
            &mut min,
            &mut est,
            &mut max,
        );
    }
    return Ok(RankEstimation { min, est, max });
}

#[cfg(feature = "hellib")]
mod hel_struct {
    #![allow(dead_code)]
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    include!(concat!(env!("OUT_DIR"), "/binding_hel_if.rs"));
}

#[cfg(any(test, fuzzing))]
mod score_example_data;
#[cfg(any(test, fuzzing))]
pub mod tests {
    use crate::{rank::RankProblem, score_example_data, RankError, RankEstimation, RankingMethod};
    use itertools::Itertools;
    static RANKING_METHODS: &[RankingMethod] = &[
        #[cfg(feature = "hellib")]
        RankingMethod::Hellib,
        RankingMethod::Hist,
        #[cfg(feature = "ntl")]
        RankingMethod::HistBigNum,
    ];
    #[test]
    fn test_rank_hellib_corpus() {
        for test in 0..score_example_data::ALL_SCORES.len() {
            println!("running test {}", test);
            let nb_sk_init = score_example_data::NB_SUBKEY_INIT;
            let nb_kv_init = score_example_data::NB_KEY_VALUE_INIT;
            let costs: Vec<Vec<f64>> = (0..nb_sk_init)
                .map(|i| {
                    (0..nb_kv_init)
                        .map(|j| -score_example_data::ALL_SCORES[test][i][j])
                        .collect()
                })
                .collect();
            for method in RANKING_METHODS {
                println!("method: {:?}", method);
                let RankEstimation { min, est, max } = method
                    .rank_nbin(
                        &costs,
                        &score_example_data::REAL_KEY,
                        score_example_data::NB_BIN,
                        Some(score_example_data::MERGE),
                    )
                    .expect("Valid problem");
                let ref_min: f64 = score_example_data::EXPECTED_RANK[test][0];
                let ref_max: f64 = score_example_data::EXPECTED_RANK[test][2];
                assert!(min <= ref_max);
                assert!(max >= ref_min);
                assert!(min <= est);
                assert!(est <= max);
                // strict checking that our new rank is better than the previous one
                assert!(min >= ref_min);
                assert!(max <= ref_max);
                #[cfg(feature = "hellib")]
                if *method == RankingMethod::Hellib {
                    let ref_est: f64 = score_example_data::EXPECTED_RANK[test][1];
                    assert_eq!(min, ref_min);
                    assert_eq!(est, ref_est);
                    assert_eq!(max, ref_max);
                }
            }
        }
    }
    #[test]
    fn test_naive_rank() {
        let costs = vec![vec![0.2, 0.8], vec![0.4, 0.3]];
        let keys_rank: Vec<(Vec<usize>, f64)> = vec![
            (vec![0, 0], 2.0),
            (vec![0, 1], 1.0),
            (vec![1, 0], 4.0),
            (vec![1, 1], 3.0),
        ];
        for (key, rank) in keys_rank.iter() {
            let problem = RankProblem::new(costs.clone(), key.clone()).unwrap();
            assert_eq!(problem.naive_rank(), *rank, "test key: {:?}", key);
        }
    }
    #[test]
    fn test_exact_value() {
        let costs = vec![vec![0.0, 1.0, 2.0, 2.0, 3.0]];
        for key in 0..costs.len() {
            let key = vec![key];
            let rank = RankProblem::new(&*costs, &*key).unwrap().naive_rank();
            for method in RANKING_METHODS {
                #[cfg(feature = "hellib")]
                if *method == RankingMethod::Hellib {
                    continue;
                }
                let rank_est = method.rank_nbin(&*costs, &*key, 8, Some(1)).unwrap();
                println!(
                    "method: {:?}, rank: {}, rank_est: {:?}",
                    method, rank, rank_est
                );
                assert_eq!(rank_est.min, rank_est.max);
                assert!(rank_est.contains(rank));
                let rank_est = method
                    .rank_accuracy(&*costs, &*key, 1.1, Some(1), 1 << 10)
                    .unwrap();
                assert_eq!(rank_est.min, rank_est.max);
                assert!(rank_est.contains(rank));
            }
        }
    }
    pub fn rank2_vs_naive(
        costs: &[Vec<f64>],
        key: &[usize],
        merge: usize,
        nb_bin: usize,
        acc_req: f64,
    ) -> Result<(), RankError> {
        let rank = RankProblem::new(costs, key)?.naive_rank();
        for method in RANKING_METHODS {
            let rank_est = method.rank_nbin(costs, key, nb_bin, Some(merge))?;
            assert!(
                rank_est.contains(rank),
                "rank: {}, rank_est: {:?}, merge: {}, nb_bin: {}, costs: {:?}, key: {:?} method: {:?}",
                rank,
                rank_est,
                merge,
                nb_bin,
                costs,
                key,
                method
            );
            let rank_est = method.rank_accuracy(costs, key, 1.0, Some(merge), 1 << 10)?;
            assert!(
                rank_est.contains(rank),
                "rank: {}, rank_est: {:?}, merge: {}, nb_bin: {}, costs: {:?}, key: {:?} method: {:?}",
                rank,
                rank_est,
                merge,
                nb_bin,
                costs,
                key,
                method
            );
            assert!(
                rank_est.max / rank_est.min <= acc_req,
                "{rank_est:?} {acc_req:?}"
            );
        }
        return Ok(());
    }
    #[test]
    fn test_rank_simple() {
        let costs = vec![vec![-0.2, -0.8], vec![-0.4, -0.3]];
        for merge in 1..=2 {
            for nb_bin in 1..100 {
                for key in costs.iter().map(|sc| 0..sc.len()).multi_cartesian_product() {
                    println!(
                        "Simple Testcase merge={}, nb_bin={}, key: {:?}",
                        merge, nb_bin, key
                    );
                    let _ = rank2_vs_naive(&costs, &key, merge, nb_bin, 2.0);
                }
            }
        }
    }
    struct TestCase {
        costs: Vec<Vec<f64>>,
        key: Vec<usize>,
        nb_bin: usize,
        merge: usize,
        acc: f64,
    }
    #[test]
    fn test_manycases() {
        let test_cases = vec![
            TestCase {
                costs: vec![
                    vec![
                        -256.0, -1.0, -16641.0, -1.0, -1.0, -1.0, -1.0, -189.0, 32511.0, -1.0,
                        -1.0, -1.0, -1.0, -1.0, 4112.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                        -20481.0, -12113.0, 13039.0, 14135.0, 55.0,
                    ],
                    vec![13621.0, 13621.0, 13621.0, 13749.0, -4113.0],
                    vec![
                        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                        -1.0, -1.0, -1.0, -1.0,
                    ],
                ],
                key: vec![0, 0, 0],
                nb_bin: 32,
                merge: 1,
                acc: 20.0,
            },
            TestCase {
                costs: vec![vec![0.0, 0.0]],
                key: vec![0],
                nb_bin: 10,
                merge: 1,
                acc: 2.0,
            },
            TestCase {
                costs: vec![vec![1.0, 0.0, 1.00001, 2.0]],
                key: vec![0],
                nb_bin: 4,
                merge: 1,
                acc: 2.0,
            },
            TestCase {
                costs: vec![
                    vec![1800.0, 1801.0, 5251.0, 0.0],
                    vec![2000.0],
                    vec![2000.0],
                ],
                key: vec![0, 0, 0],
                nb_bin: 32,
                merge: 1,
                acc: 2.0,
            },
            TestCase {
                costs: vec![vec![-1.0, 32511.0], vec![-1.0], vec![-1.0, 0.0, -26315.0]],
                key: vec![0, 0, 0],
                nb_bin: 32,
                merge: 1,
                acc: 2.0,
            },
            TestCase {
                costs: vec![vec![0.0], vec![0.0], vec![2.0, 0.0, 1.0]],
                key: vec![0, 0, 0],
                nb_bin: 4,
                merge: 1,
                acc: 1.0,
            },
            TestCase {
                costs: vec![vec![10.0, 0.0], vec![30.0, 0.0]],
                key: vec![0, 0],
                nb_bin: 4,
                merge: 1,
                acc: 2.0,
            },
            TestCase {
                costs: vec![vec![10.0, 0.0], vec![30.0, 0.0], vec![13.0, 0.0, 12.0]],
                key: vec![0, 0, 0],
                nb_bin: 4,
                merge: 1,
                acc: 2.0,
            },
        ];
        for (i, case) in test_cases.into_iter().enumerate() {
            println!("Testcase {}", i);
            let _ = rank2_vs_naive(&case.costs, &case.key, case.merge, case.nb_bin, case.acc);
        }
    }
}
