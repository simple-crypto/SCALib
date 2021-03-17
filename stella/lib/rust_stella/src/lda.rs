use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2, Axis, Zip};
use rayon::prelude::*;
pub fn get_projection_lda(
    x: ArrayView2<i16>,
    y: ArrayView1<u16>,
    sb: &mut ArrayViewMut2<f64>,
    sw: &mut ArrayViewMut2<f64>,
    c_means_bak: &mut ArrayViewMut2<f64>,
    x_f64: &mut ArrayViewMut2<f64>,
    nk: usize,
) {
    let n = x.shape()[1];
    let ns = x.shape()[0];

    let mut c_means = Array2::<i64>::zeros((nk, n));
    let mut s = Array2::<i64>::zeros((nk, 1));

    // compute class means
    c_means
        .outer_iter_mut()
        .into_par_iter()
        .zip(s.outer_iter_mut().into_par_iter())
        .enumerate()
        .for_each(|(i, (mut mean, mut s))| {
            let mut n = 0;
            x.outer_iter().zip(y.outer_iter()).for_each(|(x, y)| {
                let y = y.first().unwrap();
                if (*y as usize) == i {
                    mean.zip_mut_with(&x, |mean, x| *mean += *x as i64);
                    n += 1;
                }
            });
            s.fill(n);
        });

    let c_means = c_means.mapv(|x| x as f64);
    let s = s.mapv(|x| x as f64);
    let mean_total = c_means.sum_axis(Axis(0)).insert_axis(Axis(1)) / (ns as f64);
    let c_means = &c_means / &s.broadcast(c_means.shape()).unwrap();
    c_means_bak.assign(&c_means);
    x_f64.zip_mut_with(&x, |x, y| *x = *y as f64);
    let mut x_f64_t = Array2::<f64>::zeros(x.t().raw_dim());
    Zip::from(&mut x_f64_t)
        .and(&x_f64.t())
        .par_apply(|x, y| *x = *y);

    let st = x_f64_t.dot(x_f64) / (ns as f64) - mean_total.dot(&mean_total.t());

    x_f64
        .outer_iter_mut()
        .into_par_iter()
        .zip(y.outer_iter().into_par_iter())
        .for_each(|(mut x, y)| {
            let y = y.first().unwrap();
            x -= &c_means.slice(s![*y as usize, ..]);
        });
    Zip::from(&mut x_f64_t)
        .and(&x_f64.t())
        .par_apply(|x, y| *x = *y);
    sw.assign(&(x_f64_t.dot(x_f64) / (ns as f64)));
    Zip::from(sb)
        .and(&st)
        .and(sw)
        .par_apply(|sb, st, sw| *sb = *st - *sw);
}
pub fn predict_proba_lda(
    x: ArrayView2<i16>,
    projection: ArrayView2<f64>,

    c_means: ArrayView2<f64>,
    psd: ArrayView2<f64>,

    prs: &mut ArrayViewMut2<f64>,
) {
    let ns_in = x.shape()[1];
    let ns_proj = projection.shape()[1];
    x.axis_chunks_iter(Axis(0), 100)
        .into_par_iter()
        .zip(prs.axis_chunks_iter_mut(Axis(0), 100).into_par_iter())
        .for_each(|(x, mut prs)| {
            let mut x_i = Array1::<f64>::zeros(ns_in);
            let mut x_proj = Array1::<f64>::zeros(ns_proj);
            let mut mu = Array1::<f64>::zeros(ns_proj);

            x.outer_iter()
                .zip(prs.outer_iter_mut())
                .for_each(|(x, mut prs)| {
                    x_i = x.mapv(|x| x as f64);
                    // project trace for subspace
                    x_proj.assign(
                        &projection
                            .axis_iter(Axis(1))
                            .map(|p| (&x_i * &p).sum())
                            .collect::<Array1<f64>>(),
                    );

                    prs.assign(
                        &c_means
                            .outer_iter()
                            .map(|c_means| {
                                mu = &c_means - &x_proj;
                                psd.dot(&mut mu).fold(0.0, |acc, x| acc + x.powi(2))
                            })
                            .collect::<Array1<f64>>(),
                    );
                    prs.mapv_inplace(|x| f64::exp(-0.5 * x));
                    prs /= prs.sum();
                });
        });
}
