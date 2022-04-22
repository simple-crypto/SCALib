use itertools::izip;
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
//use ndarray_stats::QuantileExt;
use scalib::ttest;

#[test]
fn ttestacc_simple() {
    let order = 3;
    let ns = 100;
    let n = 100;

    let traces = Array2::<i16>::random((n, ns as usize), Uniform::new(0, ns));
    let y = Array1::<u16>::random((n,), Uniform::new(0, 2));

    let mut ttacc = ttest::TtestAcc::new(ns as usize, order);
    ttacc.update(traces.view(), y.view());
    let moments: Array3<f64> = ttacc.moments.to_owned();

    test_moments(
        traces.view(),
        y.view(),
        moments.view(),
        order as usize,
        ns as usize,
    );
}

#[test]
fn ttestacc_simple_chuncks() {
    let order = 2;
    let ns = 100;
    let n = 2000;
    let step = 10;

    let traces = Array2::<i16>::random((n, ns as usize), Uniform::new(0, ns));
    let y = Array1::<u16>::random((n,), Uniform::new(0, 2));

    // perform ttacc
    let mut ttacc = ttest::TtestAcc::new(ns as usize, order);
    izip!(
        traces.axis_chunks_iter(Axis(0), step),
        y.axis_chunks_iter(Axis(0), step)
    )
    .for_each(|(traces, y)| {
        ttacc.update(traces, y);
    });
    let moments: Array3<f64> = ttacc.moments.to_owned();
    test_moments(
        traces.view(),
        y.view(),
        moments.view(),
        order as usize,
        ns as usize,
    );
}

fn test_moments(
    traces: ArrayView2<i16>,
    y: ArrayView1<u16>,
    moments: ArrayView3<f64>,
    order: usize,
    ns: usize,
) {
    // Testing on set 0
    // t0
    let epsi = 1e-4;
    for i in 0..2 {
        let traces_0: Vec<ArrayView1<i16>> = izip!(traces.outer_iter(), y.iter())
            .filter(|(_, y)| **y == i)
            .map(|(t, _)| t)
            .collect();
        let mut t0 = Array2::<i16>::zeros((traces_0.len(), ns as usize));
        for (i, t) in traces_0.iter().enumerate() {
            t0.slice_mut(s![i, ..]).assign(t);
        }
        let mut t0 = t0.mapv(|x| x as f64);
        let mean: Array1<f64> = t0.mean_axis(Axis(0)).unwrap();
        //centering
        t0.axis_iter_mut(Axis(0)).for_each(|mut x| x -= &mean);

        // testing the mean:
        let mm0 = moments.slice(s![i as usize, .., ..]);
        let test: Array1<f64> = mm0.slice(s![0 as usize, ..]).to_owned();
        let mut dif = &mean - &test;
        dif.mapv_inplace(|x| x.abs());
        for d in dif.iter() {
            assert!(*d < epsi, "failed on mean {}", *d);
        }

        // test centered sums
        for j in 2..(order + 1) {
            let cs = t0.mapv(|x| x.powi(j as i32));
            let reference = cs.sum_axis(Axis(0));

            let test: Array1<f64> = mm0.slice(s![j - 1 as usize, ..]).to_owned();
            let mut dif = &reference - &test;
            dif.mapv_inplace(|x| x.abs());
            for d in dif.iter() {
                assert!(*d < epsi, "failed on order {}", j);
            }
        }
    }
}
