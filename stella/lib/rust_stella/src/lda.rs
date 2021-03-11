use ndarray::{s, Array2, ArrayView1, ArrayView2, ArrayViewMut2,Zip};
use ndarray_stats::CorrelationExt;
use rayon::prelude::*;
pub fn get_projection_lda(
    x: ArrayView2<i16>,
    y: ArrayView1<u16>,
    sb: &mut ArrayViewMut2<f64>,
    sw: &mut ArrayViewMut2<f64>,
    nk: usize,
) {
    let n = x.shape()[1];
    let mut c_means = Array2::<f64>::zeros((nk, n));

    // compute class means
    c_means
        .outer_iter_mut()
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut mean)| {
            let mut n = 0;
            x.outer_iter().zip(y.outer_iter()).for_each(|(x, y)| {
                let y = y.first().unwrap();
                if (*y as usize) == i {
                    mean.zip_mut_with(&x, |mean, x| *mean += *x as f64);
                    n += 1;
                }
            });
            mean /= n as f64;
        });

    let mut x_f64 = x.mapv(|x| x as f64);
    let st = x_f64.t().cov(0.0).unwrap();
    x_f64
        .outer_iter_mut()
        .into_par_iter()
        .zip(y.outer_iter().into_par_iter())
        .for_each(|(mut x, y)| {
            let y = y.first().unwrap();
            x -= &c_means.slice(s![*y as usize, ..]);
        });
    sw.assign(&x_f64.t().cov(0.0).unwrap());
    Zip::from(sb).and(&st).and(sw).par_apply(|sb,st,sw| *sb = *st - *sw);
}
