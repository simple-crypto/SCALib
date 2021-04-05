#![no_main]
use libfuzzer_sys::fuzz_target;

#[derive(arbitrary::Arbitrary, Debug, Clone)]
struct FuzzTarget {
    scores: Vec<Vec<i16>>,
    key: Vec<u16>,
    nb_bin: usize,
    merge_nb: usize,
}

impl FuzzTarget {
    fn to_problem(self) -> (Vec<Vec<f64>>, Vec<usize>, usize, usize) {
        let scores: Vec<Vec<f64>> = self
            .scores
            .into_iter()
            .map(|sc| sc.into_iter().map(|s| s as f64).collect())
            .collect();
        let key: Vec<usize> = self.key.into_iter().map(|k| k as usize).collect();
        return (scores, key, self.nb_bin, self.merge_nb);
    }
}

fuzz_target!(|target: FuzzTarget| {
    let (scores, key, nb_bin, merge_nb) = target.to_problem();
    let _ = ranklib::tests::rank2_vs_naive(&scores, &key, merge_nb, nb_bin);
});
