use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
};
use ndarray::{s, Array1, Array2};
use std::fmt;
#[inline(always)]
fn fwht(a: &mut [f64], len: usize) {
    let mut h = 1;
    while h < len {
        for mut i in 0..(len / (2 * h) as usize) {
            i *= 2 * h;
            for j in i..(i + h) {
                let x = a[j];
                let y = a[j + h];
                a[j] = x + y;
                a[j + h] = x - y;
            }
        }
        h *= 2;
    }
}
#[inline(always)]
fn fwht_2(a: &mut [f64]) {
    fwht(a, 2);
}
#[inline(always)]
fn fwht_4(a: &mut [f64]) {
    fwht(a, 4);
}
#[inline(always)]
fn fwht_8(a: &mut [f64]) {
    fwht(a, 8);
}
#[inline(always)]
fn fwht_16(a: &mut [f64]) {
    fwht(a, 16);
}
#[inline(always)]
fn fwht_32(a: &mut [f64]) {
    fwht(a, 32);
}
#[inline(always)]
fn fwht_64(a: &mut [f64]) {
    fwht(a, 64);
}
#[inline(always)]
fn fwht_128(a: &mut [f64]) {
    fwht(a, 128);
}
#[inline(always)]
fn fwht_256(a: &mut [f64]) {
    fwht(a, 256);
}
#[inline(always)]
fn fwht_nc(a: &mut [f64], nc: usize) {
    if nc == 2 {
        fwht_2(a);
    } else if nc == 4 {
        fwht_4(a);
    } else if nc == 8 {
        fwht_8(a);
    } else if nc == 16 {
        fwht_16(a);
    } else if nc == 32 {
        fwht_32(a);
    } else if nc == 64 {
        fwht_64(a);
    } else if nc == 128 {
        fwht_128(a);
    } else if nc == 256 {
        fwht_256(a);
    } else {
        fwht(a, nc);
    }
}
pub fn xors(inputs: &mut Vec<&mut Array2<f64>>, nc: usize) {
    for i in 0..inputs[0].shape()[0] {
        let mut acc = Array1::<f64>::ones(nc);

        inputs.iter_mut().for_each(|input| {
            let mut input = input.slice_mut(s![i, ..]);
            let input_fwt_s = input.as_slice_mut().unwrap();
            fwht_nc(input_fwt_s, nc);
            input_fwt_s
                .iter_mut()
                .for_each(|x| *x = if f64::abs(*x) == 0.0 { 1E-50 } else { *x });
            acc.zip_mut_with(&input, |x, y| *x = *x * y);
            acc /= acc.sum();
        });

        inputs.iter_mut().for_each(|input| {
            let mut input = input.slice_mut(s![i, ..]);
            input.zip_mut_with(&acc, |x, y| *x = *y / *x);
            let input_fwt_s = input.as_slice_mut().unwrap();
            fwht_nc(input_fwt_s, nc);
            let s = input.iter().fold(0.0, |acc, x| acc + x);
            input.iter_mut().for_each(|x| *x = *x / s);
        });
    }
}

fn xors_bench(c: &mut Criterion) {
    for ni in [2].iter() {
        let id = fmt::format(format_args!("n_inputs_{}", *ni));
        let mut group = c.benchmark_group(id);
        for nc in [2, 4, 8, 16, 32, 64, 128, 256].iter() {
            group.bench_with_input(BenchmarkId::new("xors", nc), nc, |b, nc| {
                b.iter(|| {
                    let mut inputs: Vec<Array2<f64>> = (0..*ni as usize)
                        .map(|_| Array2::<f64>::ones((100, *nc)))
                        .collect();
                    let mut inputs_ptx: Vec<&mut Array2<f64>> =
                        inputs.iter_mut().map(|x| x).collect();
                    xors(&mut inputs_ptx, *nc)
                });
            });
        }
        group.finish();
    }
}

fn alternate_measurement() -> Criterion {
    Criterion::default().sample_size(50)
}

criterion_group!(name=benches;
                config = alternate_measurement();
                targets=xors_bench);
criterion_main!(benches);
