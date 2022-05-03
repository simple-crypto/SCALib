use itertools::izip;
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
//use ndarray_stats::QuantileExt;
use scalib::mttest;

#[test]
fn ttestacc_simple() {
    let order = 2;
    let ns = 100;
    let n = 50;
    let nc = 2;
    let npois = 20;

    let traces = Array2::<i16>::random((n, ns as usize), Uniform::new(0, ns));
    let pois = Array2::<u32>::random((order, npois), Uniform::new(0, ns as u32));
    let y = Array1::<u16>::random((n,), Uniform::new(0, nc));

    let mut ttacc = mttest::MultivarMomentAcc::new(pois.view(), order);
    ttacc.update(traces.view(), y.view());
    let moments: Array3<f64> = ttacc.moments.to_owned();

    test_cs(
        traces.view(),
        y.view(),
        moments.view(),
        ttacc.mean.view(),
        pois.view(),
        ttacc.combis,
        ns as usize,
        nc as usize,
    );
}

#[test]
fn ttestacc_simple_chuncks() {
    let order = 2;
    let ns = 100;
    let n = 40;
    let nc = 2;
    let npois = 20;
    let step = 10;

    let traces = Array2::<i16>::random((n, ns as usize), Uniform::new(0, ns));
    let pois = Array2::<u32>::random((order, npois), Uniform::new(0, ns as u32));
    let y = Array1::<u16>::random((n,), Uniform::new(0, nc));

    let mut ttacc = mttest::MultivarMomentAcc::new(pois.view(), order);

    izip!(
        traces.axis_chunks_iter(Axis(0), step),
        y.axis_chunks_iter(Axis(0), step)
    )
    .for_each(|(traces, y)| {
        ttacc.update(traces, y);
    });

    let moments: Array3<f64> = ttacc.moments.to_owned();
    test_cs(
        traces.view(),
        y.view(),
        moments.view(),
        ttacc.mean.view(),
        pois.view(),
        ttacc.combis,
        ns as usize,
        nc as usize,
    );
}
#[test]
fn ttestacc_simple_chuncks_step1() {
    let order = 2;
    let ns = 100;
    let n = 40;
    let nc = 2;
    let npois = 20;
    let step = 1;

    let traces = Array2::<i16>::random((n, ns as usize), Uniform::new(0, ns));
    let pois = Array2::<u32>::random((order, npois), Uniform::new(0, ns as u32));
    let y = Array1::<u16>::random((n,), Uniform::new(0, nc));

    let mut ttacc = mttest::MultivarMomentAcc::new(pois.view(), order);

    izip!(
        traces.axis_chunks_iter(Axis(0), step),
        y.axis_chunks_iter(Axis(0), step)
    )
    .for_each(|(traces, y)| {
        ttacc.update(traces, y);
    });

    let moments: Array3<f64> = ttacc.moments.to_owned();
    test_cs(
        traces.view(),
        y.view(),
        moments.view(),
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
    moments: ArrayView3<f64>,
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
                assert!((x - mean[*poi as usize]).abs() < epsi);
            });
        });

        //centering
        t0.axis_iter_mut(Axis(0)).for_each(|mut x| x -= &mean);

        // test centered sums
        let mut tests = 0;
        izip!(
            moments.slice(s![i as usize, .., ..]).outer_iter(),
            combi.iter()
        )
        .for_each(|(m, combi)| {
            println!("New Combi {:#?}",combi); 
            izip!(m.iter(), pois.axis_iter(Axis(1))).for_each(|(m, poi)| {
                let mut test = Array1::<f64>::ones((n_traces,));
                for c in combi.iter() {
                    test = &test * &t0.slice(s![.., poi[*c] as usize]);
                }
                println!("{} {}",test.sum(),m);
                assert!((test.sum() - m).abs() < epsi);
                tests += 1;
            });
        });
    }
}
