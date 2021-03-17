use ndarray::{Array2, ArrayView2, ArrayView3, ArrayViewMut2, ArrayViewMut3, Axis, Zip};
use rayon::prelude::*;

pub fn update_snr_only(
    x: &ArrayView2<i16>,
    y: &ArrayView2<u16>,
    s: &mut ArrayViewMut3<i64>,
    s2: &mut ArrayViewMut3<i64>,
    nc: &mut ArrayViewMut2<u64>,
) {
    s.outer_iter_mut()
        .into_par_iter()
        .zip(s2.outer_iter_mut().into_par_iter())
        .zip(nc.outer_iter_mut().into_par_iter())
        .zip(y.outer_iter().into_par_iter())
        .for_each(|(((mut s, mut s2), mut nc), y)| {
            // for each variable

            s.outer_iter_mut().into_par_iter(). // over classes
                zip(s2.outer_iter_mut().into_par_iter()).
                zip(nc.outer_iter_mut().into_par_iter()).
                enumerate().
                for_each(|(i,((mut s,mut s2),mut nc))|{
                    let mut n = 0;
                    x.outer_iter().zip(y.iter()).for_each(|(x,y)|{
                    if i == *y as usize{
                        n += 1;
                        Zip::from(&mut s)
                        .and(&mut s2)
                        .and(&x)
                        .apply(|s, s2,x| {
                            let x = *x as i64;
                            *s += x;
                            *s2 += x.pow(2);
                        });
                    }
                    });
                    nc += n;
                });
        });
}

pub fn finalize_snr_only(
    s: &ArrayView3<i64>,
    s2: &ArrayView3<i64>,
    nc: &ArrayView2<u64>,
    snr: &mut ArrayViewMut2<f64>,
) {
    s.outer_iter()
        .zip(s2.outer_iter())
        .zip(nc.outer_iter())
        .zip(snr.outer_iter_mut())
        .for_each(|(((s, s2), nc), mut snr)| {
            // for each variable
            let mut means = Array2::<f64>::zeros(s2.raw_dim());
            means
                .outer_iter_mut()
                .into_par_iter()
                .zip(nc.outer_iter().into_par_iter())
                .zip(s.outer_iter().into_par_iter())
                .for_each(|((mut means, nc), s)| {
                    let nc = *nc.first().unwrap() as f64;
                    means.zip_mut_with(&s, |x, y| *x = (*y as f64) / nc);
                });
            let mean_var = means.var_axis(Axis(0), 0.0);

            means
                .outer_iter_mut()
                .into_par_iter()
                .zip(nc.outer_iter().into_par_iter())
                .zip(s2.outer_iter().into_par_iter())
                .for_each(|((mut means, nc), s2)| {
                    let nc = *nc.first().unwrap() as f64;
                    means.zip_mut_with(&s2, |x, y| *x = ((*y as f64) / nc) - x.powi(2));
                });
            let var_mean = means.mean_axis(Axis(0)).unwrap();
            snr.assign(&(&mean_var / &var_mean));
        });
}
