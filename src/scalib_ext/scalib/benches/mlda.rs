//use std::fmt::format;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform};
use scalib::multi_lda::{Class, MultiLdaAcc};

use ndarray_rand::rand::prelude::SliceRandom;
use rand_xoshiro::Xoshiro256StarStar as Prng;

type BenchMarkGroup<'a> = criterion::BenchmarkGroup<'a, criterion::measurement::WallTime>;

fn gen_traces(n: usize, ns: usize) -> Array2<i16> {
    Array2::<i16>::random((n, ns), Uniform::new(0, 100))
}
fn gen_classes(nv: usize, n: usize, nc: usize) -> Array2<u16> {
    Array2::<u16>::random((n, nv), Uniform::new(0, nc as u16))
}

fn gen_pois_with_maxpois(
    npois: usize,
    ns: u32,
    max: usize,
    nv: usize,
    rng: &mut Prng,
) -> Vec<Vec<u32>> {
    // Randomly generate a permutation out of the ns possible indexes
    let mut work: Vec<u32> = (0..ns).collect();
    work.shuffle(rng);
    // Generate the vector for each var
    let mut pois: Vec<Vec<u32>> = Vec::new();
    work.truncate(max);
    for _ in 0..nv {
        // shuffle
        work.shuffle(rng);
        // fetch
        let mut sorted_pois = work[0..npois].to_vec();
        sorted_pois.sort();
        assert!(sorted_pois.is_sorted(), "POIs not sorted");
        pois.push(sorted_pois)
    }
    pois
}

fn bench_mlda_ll_update_sums(mlda: &mut MultiLdaAcc, traces: &Array2<i16>, classes: &Array2<u16>) {
    mlda.trace_sums.update(traces.view(), classes.view());
}

fn bench_mlda_ll_update_covs(mlda: &mut MultiLdaAcc, traces: &Array2<i16>) {
    let _ = mlda.cov_pois.update(&mlda.poi_map, traces.view());
}

fn bench_mlda_ll_update(mlda: &mut MultiLdaAcc, traces: &Array2<i16>, classes: &Array2<u16>) {
    let _ = mlda.update(traces.view(), classes.view());
}

fn generate_case_data(nv: usize, nc: Class, ns: u32, n: usize) -> (Array2<i16>, Array2<u16>) {
    // Genereate the useful data
    let t = gen_traces(n, ns as usize);
    let x = gen_classes(nv, n, nc as usize);
    (t, x)
}

fn generate_inputs_mlda_ll_update(
    seed: u64,
    nc: Class,
    ns: u32,
    nv: usize,
    npois: usize,
    max_npois: usize,
    n: usize,
) -> (MultiLdaAcc, Array2<i16>, Array2<u16>) {
    // Create the prng used for POIs generation
    let mut prng = Prng::seed_from_u64(seed);
    let pois: Vec<Vec<u32>> = gen_pois_with_maxpois(npois, ns, max_npois, nv, &mut prng);
    // Genereate the useful data
    let (t, x) = generate_case_data(nv, nc, ns, n);
    // Instanciate the mlda object
    let mlda = MultiLdaAcc::new(ns, nc, pois).unwrap();
    // Return
    (mlda, t, x)
}

fn bench_mlda_ll_update_sums_inner(
    seed: u64,
    nc: Class,
    ns: u32,
    nv: usize,
    npois: usize,
    max_npois: usize,
    n: usize,
    group: &mut BenchMarkGroup,
) {
    // Sums
    group.bench_with_input(
        BenchmarkId::new(
            format!(
                "INNER-SUMS nv:{} ; ns:{} ; npois:{} [max:{}]",
                nv, ns, npois, max_npois
            ),
            nv,
        ),
        &nv,
        |b, _| {
            let (mut mlda, traces, classes) =
                generate_inputs_mlda_ll_update(seed, nc, ns, nv, npois, max_npois, n);
            b.iter(|| {
                bench_mlda_ll_update_sums(&mut mlda, &traces, &classes);
            })
        },
    );
}

fn bench_mlda_ll_update_scatterpairs_inner(
    seed: u64,
    nc: Class,
    ns: u32,
    nv: usize,
    npois: usize,
    max_npois: usize,
    n: usize,
    group: &mut BenchMarkGroup,
) {
    // Sums
    group.bench_with_input(
        BenchmarkId::new(
            format!(
                "INNER-PAIRS nv:{} ; ns:{} ; npois:{} [max:{}]",
                nv, ns, npois, max_npois
            ),
            nv,
        ),
        &nv,
        |b, _| {
            let (mut mlda, traces, _classes) =
                generate_inputs_mlda_ll_update(seed, nc, ns, nv, npois, max_npois, n);
            b.iter(|| {
                bench_mlda_ll_update_covs(&mut mlda, &traces);
            })
        },
    );
}

fn bench_mlda_ll_update_global_inner(
    seed: u64,
    nc: Class,
    ns: u32,
    nv: usize,
    npois: usize,
    max_npois: usize,
    n: usize,
    group: &mut BenchMarkGroup,
) {
    // Sums
    group.bench_with_input(
        BenchmarkId::new(
            format!(
                "GLOB nv:{} ; ns:{} ; npois:{} [max:{}]",
                nv, ns, npois, max_npois
            ),
            nv,
        ),
        &nv,
        |b, _| {
            let (mut mlda, traces, classes) =
                generate_inputs_mlda_ll_update(seed, nc, ns, nv, npois, max_npois, n);
            b.iter(|| {
                bench_mlda_ll_update(&mut mlda, &traces, &classes);
            })
        },
    );
}

fn bench_mlda_ll_update_all(
    seed: u64,
    nc: Class,
    ns: u32,
    nv: usize,
    npois: usize,
    max_npois: usize,
    n: usize,
    group: &mut BenchMarkGroup,
) {
    bench_mlda_ll_update_sums_inner(seed, nc, ns, nv, npois, max_npois, n, group);
    bench_mlda_ll_update_scatterpairs_inner(seed, nc, ns, nv, npois, max_npois, n, group);
    bench_mlda_ll_update_global_inner(seed, nc, ns, nv, npois, max_npois, n, group);
    println!("\n");
}

fn bench_mlda_update_top(c: &mut Criterion) {
    let seed = 0;
    let nc = 256;
    let n = 10000;
    let mut group = c.benchmark_group("MLDA update low-level");
    bench_mlda_ll_update_all(seed, nc, 10000, 1, 1, 10000, n, &mut group);
    bench_mlda_ll_update_all(seed, nc, 10000, 1, 1000, 10000, n, &mut group);
    bench_mlda_ll_update_all(seed, nc, 10000, 100, 1, 10000, n, &mut group);
    bench_mlda_ll_update_all(seed, nc, 10000, 100, 1000, 10000, n, &mut group);
    bench_mlda_ll_update_all(seed, nc, 10000, 1000, 1, 10000, n, &mut group);
    bench_mlda_ll_update_all(seed, nc, 10000, 1000, 1000, 10000, n, &mut group);
}

fn bench_univariate_mlda_update(nc: Class, ns: u32, n: usize, group: &mut BenchMarkGroup) {
    group.bench_with_input(
        BenchmarkId::new(format!("UNI-UPDATE-NEW nv:{} ; ns:{}", 1, ns), ns),
        &ns,
        |b, _| {
            // Pois, which is in fact all the time samples
            let pois: Vec<Vec<u32>> = vec![(0..ns).into_iter().collect()];
            // Creation of the Mlda instance
            let mut mlda = MultiLdaAcc::new(ns, nc, pois).unwrap();
            // Generate the inputs
            let (t, x) = generate_case_data(1, nc, ns, n);
            b.iter(|| {
                let _ = mlda.update(t.view(), x.view());
            })
        },
    );
}

fn bench_univariate_update(nc: Class, ns: u32, n: usize, group: &mut BenchMarkGroup) {
    bench_univariate_mlda_update(nc, ns, n, group);
}

fn bench_univariate_mlda_solve(nc: Class, ns: u32, n: usize, p: u32, group: &mut BenchMarkGroup) {
    group.bench_with_input(
        BenchmarkId::new(
            format!("UNI-SOLVE-NEW nv:{} ; ns:{} ; ndim:{}", 1, ns, p),
            p,
        ),
        &p,
        |b, _| {
            // Pois, which is in fact all the time samples
            let pois: Vec<Vec<u32>> = vec![(0..ns).into_iter().collect()];
            // Creation of the Mlda instance
            let mut mlda = MultiLdaAcc::new(ns, nc, pois).unwrap();
            // Generate the inputs
            let (t, x) = generate_case_data(1, nc, ns, n);
            let _ = mlda.update(t.view(), x.view());
            let config = scalib::Config::no_progress();
            b.iter(|| {
                let _mldares = mlda.lda(p, &config);
            })
        },
    );
}

fn bench_univariate_solve(nc: Class, ns: u32, n: usize, p: u32, group: &mut BenchMarkGroup) {
    bench_univariate_mlda_solve(nc, ns, n, p, group);
    println!("\n");
}

fn bench_univariate_mlda_predict(nc: Class, ns: u32, n: usize, p: u32, group: &mut BenchMarkGroup) {
    group.bench_with_input(
        BenchmarkId::new(
            format!("UNI-PREDICT-NEW nv:{} ; ns:{} ; ndim:{}", 1, ns, p),
            p,
        ),
        &p,
        |b, _| {
            // Pois, which is in fact all the time samples
            let pois: Vec<Vec<u32>> = vec![(0..ns).into_iter().collect()];
            // Creation of the Mlda instance
            let mut mlda = MultiLdaAcc::new(ns, nc, pois).unwrap();
            // Generate the inputs
            let (t, x) = generate_case_data(1, nc, ns, n);
            let _ = mlda.update(t.view(), x.view());
            let config = scalib::Config::no_progress();
            let ldares = mlda.lda(p, &config).unwrap();
            b.iter(|| {
                let _prs = ldares.predict_proba(t.view());
            })
        },
    );
}

fn bench_univariate_predict(nc: Class, ns: u32, n: usize, p: u32, group: &mut BenchMarkGroup) {
    bench_univariate_mlda_predict(nc, ns, n, p, group);
    println!("\n");
}

fn bench_univariate(c: &mut Criterion) {
    let mut group = c.benchmark_group("LDA vs MLDA");
    let ns = 1000;
    let nc = 256;
    let n = 10000;
    bench_univariate_update(nc, ns, n, &mut group);
    bench_univariate_solve(nc, ns, n, 1, &mut group);
    bench_univariate_solve(nc, ns, n, 8, &mut group);
    bench_univariate_predict(nc, ns, n, 1, &mut group);
    bench_univariate_predict(nc, ns, n, 8, &mut group);
}

criterion_group! {
    name = benches;
    // This can be any expression that returns a `Criterion` object.
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_mlda_update_top, bench_univariate
}
criterion_main!(benches);
