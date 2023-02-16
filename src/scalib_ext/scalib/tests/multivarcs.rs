// This files tests the internal representations of multivarcs.
//
// It first computes the central sums with multivarcs and
// the compare the results with a reference 2 passes algorithm.
//
// It tests for:
//   - different d's
//   - feed multivarcs with multiple batches
//   - number of samples (ns)
//   - number of traces nc
use itertools::izip;
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_xoshiro::Xoshiro256StarStar;
use scalib::mttest;
extern crate approx;

fn gen_problem(
    n: usize,
    ns: i16,
    order: usize,
    npois: usize,
    nc: u16,
) -> (Array2<i16>, Array2<u32>, Array1<u16>) {
    let seed = 42;
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
    let traces = Array2::<i16>::random_using((n, ns as usize), Uniform::new(0, ns), &mut rng);
    let pois = Array2::<u32>::random_using((order, npois), Uniform::new(0, ns as u32), &mut rng);
    let y = Array1::<u16>::random_using((n,), Uniform::new(0, nc), &mut rng);
    return (traces, pois, y);
}

#[test]
fn multivarcs_simple() {
    let order = 2;
    let ns = 100;
    let n = 50;
    let nc = 2;
    let npois = 20;

    let (traces, pois, y) = gen_problem(n, ns, order, npois, nc);

    let mut ttacc = mttest::MultivarCSAcc::new(pois.view(), order);
    ttacc.update(traces.view(), y.view());
    let cs: Array3<f64> = ttacc.cs.to_owned();

    test_cs(
        traces.view(),
        y.view(),
        cs.view(),
        ttacc.mean.view(),
        pois.view(),
        ttacc.combis,
        ns as usize,
        nc as usize,
    );
}

#[test]
fn multivarcs_simple_chuncks() {
    let order = 2;
    let ns = 100;
    let n = 40;
    let nc = 2;
    let npois = 20;
    let step = 10;

    let (traces, pois, y) = gen_problem(n, ns, order, npois, nc);

    let mut ttacc = mttest::MultivarCSAcc::new(pois.view(), order);

    izip!(
        traces.axis_chunks_iter(Axis(0), step),
        y.axis_chunks_iter(Axis(0), step)
    )
    .for_each(|(traces, y)| {
        ttacc.update(traces, y);
    });

    let cs: Array3<f64> = ttacc.cs.to_owned();
    test_cs(
        traces.view(),
        y.view(),
        cs.view(),
        ttacc.mean.view(),
        pois.view(),
        ttacc.combis,
        ns as usize,
        nc as usize,
    );
}
#[test]
fn multivarcs_simple_chuncks_step1() {
    let order = 2;
    let ns = 100;
    let n = 40;
    let nc = 2;
    let npois = 20;
    let step = 1;

    let (traces, pois, y) = gen_problem(n, ns, order, npois, nc);

    let mut ttacc = mttest::MultivarCSAcc::new(pois.view(), order);

    izip!(
        traces.axis_chunks_iter(Axis(0), step),
        y.axis_chunks_iter(Axis(0), step)
    )
    .for_each(|(traces, y)| {
        ttacc.update(traces, y);
    });

    let cs: Array3<f64> = ttacc.cs.to_owned();
    test_cs(
        traces.view(),
        y.view(),
        cs.view(),
        ttacc.mean.view(),
        pois.view(),
        ttacc.combis,
        ns as usize,
        nc as usize,
    );
}
fn test_cs(
    traces: ArrayView2<i16>,
    y: ArrayView1<u16>,
    cs: ArrayView3<f64>,
    mean_ref: ArrayView3<f64>,
    pois: ArrayView2<u32>,
    combi: Vec<Vec<usize>>,
    ns: usize,
    nc: usize,
) {
    let epsi = 1e-4;
    for i in 0..nc {
        let traces_0: Vec<ArrayView1<i16>> = izip!(traces.outer_iter(), y.iter())
            .filter(|(_, y)| **y == i as u16)
            .map(|(t, _)| t)
            .collect();

        let mut t0 = Array2::<i16>::zeros((traces_0.len(), ns as usize));
        for (i, t) in traces_0.iter().enumerate() {
            t0.slice_mut(s![i, ..]).assign(t);
        }
        let mut t0 = t0.mapv(|x| x as f64);
        let n_traces = t0.shape()[0];
        let mean: Array1<f64> = t0.mean_axis(Axis(0)).unwrap();

        izip!(
            mean_ref.slice(s![i as usize, .., ..]).outer_iter(),
            pois.outer_iter()
        )
        .for_each(|(mean_ref, pois)| {
            println!("New vector {:#?} {:#?}", mean_ref.shape(), pois.shape());
            izip!(mean_ref, pois).for_each(|(x, poi)| {
                println!("{} {}", x, mean[*poi as usize]);
                approx::assert_abs_diff_eq!(*x, mean[*poi as usize], epsilon = epsi);
            });
        });

        //centering
        t0.axis_iter_mut(Axis(0)).for_each(|mut x| x -= &mean);

        // test centered sums
        let mut tests = 0;
        izip!(cs.slice(s![i as usize, .., ..]).outer_iter(), combi.iter()).for_each(
            |(m, combi)| {
                println!("New Combi {:#?}", combi);
                izip!(m.iter(), pois.axis_iter(Axis(1))).for_each(|(m, poi)| {
                    let mut test = Array1::<f64>::ones((n_traces,));
                    for c in combi.iter() {
                        test = &test * &t0.slice(s![.., poi[*c] as usize]);
                    }
                    println!("{} {}", test.sum(), m);
                    approx::assert_abs_diff_eq!(test.sum(), m, epsilon = epsi);
                    tests += 1;
                });
            },
        );
    }
}
