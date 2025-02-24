use std::fmt::format;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::Group;
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
    group.bench_function(
        BenchmarkId::new(
            format!("nv:{} ; npois:{} [max: {}]", nv, npois, max_npois),
            nv,
        ),
        |b| {
            // Generate t
            let mut prng = Prng::seed_from_u64(0);
            // First, generate the POis
            let pois: Vec<Vec<u32>> = gen_pois_with_maxpois(npois, ns, max_npois, nv, &mut prng);
            b.iter(|| MultiLdaAcc::new(ns, nc, pois.clone()))
        },
    );
}

fn bench_mlda_init(c: &mut Criterion) {
    let nc = 256;
    let mut group = c.benchmark_group(format!("MLDA init {}", 0));
    // Dense
    bench_mlda_init_inner(nc, 1000000, 100, 1000, 4000, &mut group);
    // Sparse
    bench_mlda_init_inner(nc, 1000000, 100, 10, 4000, &mut group);
    // Sparser
    bench_mlda_init_inner(nc, 1000000, 100, 10, 100000, &mut group);
    // Sparcer ++ / small load
    bench_mlda_init_inner(nc, 1000000, 10000, 1, 1000000, &mut group);
    // Sparcer ++ / big load ---> Disabled due to memory shortage
    //bench_mlda_init_inner(nc, 1000000, 10000, 10, 1000000, &mut group);
}

fn bench_mlda_update_sums_inner(
    nc: Class,
    ns: u32,
    nv: usize,
    npois: usize,
    max_npois: usize,
    n: usize,
    group: &mut BenchMarkGroup,
) {
    group.bench_with_input(
        BenchmarkId::new(
            format!(
                "ns:{} ; nv:{} ; npois:{} [max:{}]",
                ns, nv, npois, max_npois
            ),
            nv,
        ),
        &nv,
        |b, _| {
            // Create the prng used for POIs generation
            let mut prng = Prng::seed_from_u64(0);
            let pois: Vec<Vec<u32>> = gen_pois_with_maxpois(npois, ns, max_npois, nv, &mut prng);
            // Genereate the useful data
            let t = gen_traces(n, ns as usize);
            let x = gen_classes(nv, n, nc as usize);
            // Instanciate the mlda object
            let mut mlda = MultiLdaAcc::new(ns, nc, pois).unwrap();
            b.iter(|| {
                mlda.state
                    .trace_sums
                    .update(&mlda.conf.trace_sums, t.view(), x.view())
            })
        },
    );
}

fn bench_mlda_update_sums(c: &mut Criterion) {
    let nc = 256;
    let n = 10000;
    let mut group = c.benchmark_group("MLDA update sum");
    bench_mlda_update_sums_inner(nc, 10000, 10, 1, 1000, n, &mut group);
    // disabled because too long to execute (clone_row_major mostly)
    //bench_mlda_update_sums_inner(nc, 100000, 10, 1, 1000, n, &mut group);
    // disabled because too long to execute (clone_row_major mostly)
    //bench_mlda_update_sums_inner(nc, 10000, 50000, 1, 1000, n, &mut group);
    // disabled because too long to execute (clone_row_major mostly)
    //bench_mlda_update_sums_inner(nc, 100000, 50000, 1, 1000, n, &mut group);
}

fn bench_mlda_update_covs_inner(
    nc: Class,
    ns: u32,
    nv: usize,
    npois: usize,
    max_npois: usize,
    n: usize,
    group: &mut BenchMarkGroup,
) {
    group.bench_with_input(
        BenchmarkId::new(
            format!(
                "ns:{} ; nv:{} ; npois:{} [max:{}]",
                ns, nv, npois, max_npois
            ),
            nv,
        ),
        &nv,
        |b, _| {
            // Create the prng used for POIs generation
            let mut prng = Prng::seed_from_u64(0);
            let pois: Vec<Vec<u32>> = gen_pois_with_maxpois(npois, ns, max_npois, nv, &mut prng);
            // Genereate the useful data
            let t = gen_traces(n, ns as usize);
            // Instanciate the mlda object
            let mut mlda = MultiLdaAcc::new(ns, nc, pois).unwrap();
            b.iter(|| {
                mlda.state
                    .cov_acc
                    .update(&mlda.conf.poi_map, &mlda.conf.cov_pois, t.view())
            })
        },
    );
}

fn bench_mlda_update_covs(c: &mut Criterion) {
    let nc = 256;
    let n = 10000;
    let mut group = c.benchmark_group("MLDA update cov");
    // Passing, most time allocating paris_matrix with from_elem
    bench_mlda_update_covs_inner(nc, 10000, 10, 1, 1000, n, &mut group);
    //bench_mlda_update_covs_inner(nc, 10000, 10, 100, 1000, n, &mut group);
    //// Long duration, most time allocating paris_matrix with from_elem
    //bench_mlda_update_covs_inner(nc, 100000, 10, 1, 1000, n, &mut group);
    //bench_mlda_update_covs_inner(nc, 100000, 10, 100, 1000, n, &mut group);
    //// Long duration, most time allocating paris_matrix with from_elem
    //bench_mlda_update_covs_inner(nc, 10000, 50000, 1, 1000, n, &mut group);
    //bench_mlda_update_covs_inner(nc, 10000, 50000, 100, 1000, n, &mut group);
    //// Long duration, most time allocating paris_matrix with from_elem
    //bench_mlda_update_covs_inner(nc, 100000, 50000, 1, 1000, n, &mut group);
    //bench_mlda_update_covs_inner(nc, 100000, 50000, 100, 1000, n, &mut group);
}

criterion_group! {
    name = benches;
    // This can be any expression that returns a `Criterion` object.
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_mlda_init, bench_mlda_update_sums, bench_mlda_update_covs
}
criterion_main!(benches);
