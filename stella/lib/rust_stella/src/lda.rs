use ndarray::{s, Array2, Array1, ArrayView1, ArrayView2, ArrayViewMut2,Zip,Axis};
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
    let ns = x.shape()[0];

    let mut c_means = Array2::<i64>::zeros((nk, n));
    let mut s = Array2::<i64>::zeros((nk,1));
    
    // compute class means
    c_means
        .outer_iter_mut()
        .into_par_iter()
        .zip(s.outer_iter_mut().into_par_iter())
        .enumerate()
        .for_each(|(i, (mut mean,mut s))| {
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
    let mean_total = c_means.sum_axis(Axis(0)).insert_axis(Axis(1))/ (ns as f64);
    let c_means = &c_means / &s.broadcast(c_means.shape()).unwrap();
    
    let mut x_f64 = x.mapv(|x| x as f64);
    let mut x_f64_t = Array2::<f64>::zeros(x_f64.t().raw_dim());
    Zip::from(&mut x_f64_t).and(&x_f64.t()).par_apply(|mut x,y| *x = *y);
    
    // #TODO copy of x_f64 to enable blas 
    let mut st = x_f64_t.dot(&x_f64)/(ns as f64) - mean_total.dot(&mean_total.t());
   
    x_f64
        .outer_iter_mut()
        .into_par_iter()
        .zip(y.outer_iter().into_par_iter())
        .for_each(|(mut x, y)| {
            let y = y.first().unwrap();
            x -= &c_means.slice(s![*y as usize, ..]);
        });
    Zip::from(&mut x_f64_t).and(&x_f64.t()).par_apply(|mut x,y| *x = *y);
    sw.assign(&(x_f64_t.dot(&x_f64)/(ns as f64)));
    
    Zip::from(sb).and(&st).and(sw).par_apply(|sb,st,sw| *sb = *st - *sw);
}
