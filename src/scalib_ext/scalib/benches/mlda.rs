use std::fmt::format;

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
fn gen_classes(np: usize, n: usize, nc: usize) -> Array2<u16> {
    Array2::<u16>::random((np, n), Uniform::new(0, nc as u16))
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
        pois.push(work[0..npois].to_vec())
    }
    pois
}

fn bench_mlda_init_inner(
    nc: Class,
    ns: u32,
    nv: usize,
    npois: usize,
    max_npois: usize,
    group: &mut BenchMarkGroup,
) {
    // Generate t
    let mut prng = Prng::seed_from_u64(0);
    // First, generate the POis
    let pois: Vec<Vec<u32>> = gen_pois_with_maxpois(npois, ns, max_npois, nv, &mut prng);
    group.bench_with_input(
        BenchmarkId::new(
            format!("nv:{} ; npois:{} [max: {}]", nv, npois, max_npois),
            nv,
        ),
        &nv,
        |b, _| b.iter(|| MultiLdaAcc::new(ns, nc, pois.clone())),
    );
}

fn bench_mlda_init(c: &mut Criterion) {
    let maxpois = 4000;
    let nc = 256;
    for ns in [1000000, 10000] {
        let mut group = c.benchmark_group(format!("MLDA init ns:{}", ns));
        // Dense
        bench_mlda_init_inner(nc, ns, 100, 1000, maxpois, &mut group);
        // Sparse
        bench_mlda_init_inner(nc, ns, 1000000, 10, maxpois, &mut group);
    }
}

criterion_group! {
    name = benches;
    // This can be any expression that returns a `Criterion` object.
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_mlda_init
}
criterion_main!(benches);
