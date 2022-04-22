use itertools::{izip};
use ndarray::{Array1, Array2, ArrayView1,s, Axis, Array3};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
//use ndarray_stats::QuantileExt;
use scalib::ttest;

#[test]
fn ttestacc_simple() {
    let d = 2;
    let ns = 100;
    let n = 2;

    let traces = Array2::<i16>::random((n, ns as usize), Uniform::new(0, ns));
    let y = Array1::<u16>::random((n,), Uniform::new(0, 2));

    // t0
    let traces_0 : Vec<ArrayView1<i16>>= izip!(traces.outer_iter(), y.iter())
        .filter(|(_, y)| **y == 0)
        .map(|(t, _)| t)
        .collect();
    let mut t0 = Array2::<i16>::zeros((traces_0.len(),ns as usize));
    for (i,t) in traces_0.iter().enumerate(){
        t0.slice_mut(s![i,..]).assign(t);
    }
    let t0 = t0.mapv(|x| x as f64);

    // t1
    let traces_1 : Vec<ArrayView1<i16>>= izip!(traces.outer_iter(), y.iter())
        .filter(|(_, y)| **y == 1)
        .map(|(t, _)| t)
        .collect();
    let mut t1 = Array2::<i16>::zeros((traces_1.len(),ns as usize));
    for (i,t) in traces_1.iter().enumerate(){
        t1.slice_mut(s![i,..]).assign(t);
    }
    let t1 = t1.mapv(|x| x as f64);

    let u0 : Array1<f64> = t0.mean_axis(Axis(0)).unwrap();
    let u1 = t1.mean_axis(Axis(0)).unwrap();

    let v0 : Array1<f64> = t0.var_axis(Axis(0),0.0);
    let v1 = t1.var_axis(Axis(0),0.0);

    // perform ttacc
    let mut ttacc = ttest::TtestAcc::new(ns as usize, d);
    ttacc.update(traces.view(), y.view());
    let moments : Array3<f64> = ttacc.get_moments();

    let mm0 : Array2<f64> = moments.slice(s![0 as usize,..,..]).to_owned();
    let mm1 : Array2<f64> = moments.slice(s![1 as usize,..,..]).to_owned();

    // test mean 0
    let test : Array1<f64> = mm0.slice(s![0 as usize, ..]).to_owned();
    let mut dif = &u0 - &test;
    dif.mapv_inplace(|x| x.abs());
    for d in dif.iter(){
        assert!(*d < 1e-7);
    }

    // test mean 1
    let test : Array1<f64> = mm1.slice(s![0 as usize, ..]).to_owned();
    let mut dif = &u1 - &test;
    dif.mapv_inplace(|x| x.abs());
    for d in dif.iter(){
        assert!(*d < 1e-7);
    }

    // test var 0
    let test : Array1<f64> = mm0.slice(s![1 as usize, ..]).to_owned();
    let mut dif = &v0 - &test;
    dif.mapv_inplace(|x| x.abs());
    for d in dif.iter(){
        assert!(*d < 1e-7);
    }

    // test var 1
    let test : Array1<f64> = mm1.slice(s![1 as usize, ..]).to_owned();
    let mut dif = &v1 - &test;
    dif.mapv_inplace(|x| x.abs());
    for d in dif.iter(){
        assert!(*d < 1e-7);
    }
}

#[test]
fn ttestacc_simple_chuncks() {
    let d = 2;
    let ns = 100;
    let n = 2;
    let step = 10;

    let traces = Array2::<i16>::random((n, ns as usize), Uniform::new(0, ns));
    let y = Array1::<u16>::random((n,), Uniform::new(0, 2));

    // perform ttacc
    let mut ttacc = ttest::TtestAcc::new(ns as usize, d);
    izip!(traces.axis_chunks_iter(Axis(0),step),
            y.axis_chunks_iter(Axis(0),step)).
        for_each(|(traces,y)|{
            ttacc.update(traces, y);
        });
    let moments : Array3<f64> = ttacc.get_moments();

    // compute references
    // t0
    let traces_0 : Vec<ArrayView1<i16>>= izip!(traces.outer_iter(), y.iter())
        .filter(|(_, y)| **y == 0)
        .map(|(t, _)| t)
        .collect();
    let mut t0 = Array2::<i16>::zeros((traces_0.len(),ns as usize));
    for (i,t) in traces_0.iter().enumerate(){
        t0.slice_mut(s![i,..]).assign(t);
    }
    let t0 = t0.mapv(|x| x as f64);

    // t1
    let traces_1 : Vec<ArrayView1<i16>>= izip!(traces.outer_iter(), y.iter())
        .filter(|(_, y)| **y == 1)
        .map(|(t, _)| t)
        .collect();
    let mut t1 = Array2::<i16>::zeros((traces_1.len(),ns as usize));
    for (i,t) in traces_1.iter().enumerate(){
        t1.slice_mut(s![i,..]).assign(t);
    }
    let t1 = t1.mapv(|x| x as f64);

    let u0 : Array1<f64> = t0.mean_axis(Axis(0)).unwrap();
    let u1 = t1.mean_axis(Axis(0)).unwrap();

    let v0 : Array1<f64> = t0.var_axis(Axis(0),0.0);
    let v1 = t1.var_axis(Axis(0),0.0);

    let mm0 : Array2<f64> = moments.slice(s![0 as usize,..,..]).to_owned();
    let mm1 : Array2<f64> = moments.slice(s![1 as usize,..,..]).to_owned();

    // test mean 0
    let test : Array1<f64> = mm0.slice(s![0 as usize, ..]).to_owned();
    let mut dif = &u0 - &test;
    dif.mapv_inplace(|x| x.abs());
    for d in dif.iter(){
        assert!(*d < 1e-7);
    }

    // test mean 1
    let test : Array1<f64> = mm1.slice(s![0 as usize, ..]).to_owned();
    let mut dif = &u1 - &test;
    dif.mapv_inplace(|x| x.abs());
    for d in dif.iter(){
        assert!(*d < 1e-7);
    }

    // test var 0
    let test : Array1<f64> = mm0.slice(s![1 as usize, ..]).to_owned();
    let mut dif = &v0 - &test;
    dif.mapv_inplace(|x| x.abs());
    for d in dif.iter(){
        assert!(*d < 1e-7);
    }

    // test var 1
    let test : Array1<f64> = mm1.slice(s![1 as usize, ..]).to_owned();
    let mut dif = &v1 - &test;
    dif.mapv_inplace(|x| x.abs());
    for d in dif.iter(){
        assert!(*d < 1e-7);
    }
}
