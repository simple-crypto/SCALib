use lapack::*;
use ndarray::{s, Array1, Array2, Axis, Zip};
use ndarray_linalg::*;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
pub struct LDA {
    cov: Array2<f64>,
    psd: Array2<f64>,
    means: Array2<f64>,
    projection: Array2<f64>,
    n_components: usize,
    nc: usize,
}
#[pymethods]
impl LDA {
    #[new]
    fn new(_py: Python, nc: usize, n_components: usize) -> Self {
        LDA {
            cov: Array2::<f64>::zeros((1, 1)),
            psd: Array2::<f64>::zeros((1, 1)),
            means: Array2::<f64>::zeros((1, 1)),
            projection: Array2::<f64>::zeros((1, 1)),
            nc: nc,
            n_components: n_components,
        }
    }
    fn fit(&mut self, py: Python, x: PyReadonlyArray2<i16>, y: PyReadonlyArray1<u16>) {
        let x = x.as_array();
        let y = y.as_array();
        let nc = self.nc;
        let n_components = self.n_components;
        py.allow_threads(|| {
            let nk = nc;
            let ns = x.shape()[1];
            let n = x.shape()[0];

            let mut c_means = Array2::<i64>::zeros((nk, ns));
            let mut s = Array2::<i64>::zeros((nk, 1));
            let mut sb_o = Array2::<f64>::zeros((ns, ns)).reversed_axes();
            let mut sw_o = Array2::<f64>::zeros((ns, ns)).reversed_axes();

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
            let mean_total = c_means.sum_axis(Axis(0)).insert_axis(Axis(1)) / (n as f64);
            let c_means = &c_means / &s.broadcast(c_means.shape()).unwrap();

            let mut x_f64 = x.mapv(|x| x as f64);
            let mut x_f64_t = Array2::<f64>::zeros(x.t().raw_dim());
            Zip::from(&mut x_f64_t)
                .and(&x_f64.t())
                .par_apply(|x, y| *x = *y);

            let st = x_f64_t.dot(&x_f64) / (n as f64) - mean_total.dot(&mean_total.t());
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
            sw_o.assign(&(x_f64_t.dot(&x_f64) / (n as f64)));

            Zip::from(&mut sb_o)
                .and(&st)
                .and(&sw_o)
                .par_apply(|sb, st, sw| *sb = st - sw);

            let mut evals = vec![0.0; ns as usize];

            // link to lapack sysgvd
            unsafe {
                let mut i: i32 = 0;
                let itype = vec![1];
                let nwork = 1 + 6 * ns + 2 * ns * ns;
                let niwork = 3 + 5 * ns;
                let mut work = vec![0.0; nwork];
                let mut iwork = vec![0; niwork];
                dsygvd(
                    &itype,
                    b'V',
                    b'L',
                    ns as i32,
                    sb_o.as_allocated_mut().unwrap(),
                    ns as i32,
                    sw_o.as_allocated_mut().unwrap(),
                    ns as i32,
                    &mut evals,
                    &mut work,
                    nwork as i32,
                    &mut iwork,
                    niwork as i32,
                    &mut i,
                );
            }
            let mut index: Vec<(usize, f64)> = (0..evals.len()).zip(evals).collect();
            let evecs = sb_o;
            index.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
            index.reverse();
            let mut projection = Array2::<f64>::zeros((n_components, index.len()));
            index
                .iter()
                .zip(projection.axis_iter_mut(Axis(0)))
                .for_each(|((i, _), mut proj)| {
                    proj.assign(&evecs.slice(s![.., *i]));
                });

            // compute mean and cov
            let means = projection.dot(&c_means.t());
            let traces_t = projection.dot(&x_f64_t); // (n,n_components)

            let mut traces_t_t = Array2::zeros(traces_t.t().raw_dim());
            Zip::from(&mut traces_t_t)
                .and(&traces_t.t())
                .par_apply(|x, y| *x = *y);

            let cov = traces_t.dot(&traces_t_t) / (traces_t.shape()[1] as f64);

            let (s, u) = cov.eigh(UPLO::Lower).unwrap();
            let cond =
                (cov.len() as f64) * (s.fold(f64::MIN, |acc, x| acc.max(f64::abs(*x)))) * 2.3E-16;

            let pack: Vec<(usize, &f64)> = (0..s.len())
                .zip(s.into_iter())
                .filter(|(_, &s)| f64::abs(s) > cond)
                .collect();

            let mut u_s = Array2::<f64>::zeros((u.shape()[0], pack.len()));
            pack.iter()
                .zip(u_s.axis_iter_mut(Axis(1)))
                .for_each(|((i, _), mut u_s)| u_s.assign(&u.slice(s![.., *i])));

            let psigma_diag: Array1<f64> =
                pack.into_iter().map(|(_, x)| 1.0 / f64::sqrt(*x)).collect();
            let psd = &u_s * &psigma_diag.broadcast(u_s.shape()).unwrap();

            self.cov = cov.to_owned();
            self.psd = psd.to_owned();
            self.means = means.to_owned();
            self.projection = projection.t().to_owned();
        })
    }

    fn set_state<'py>(
        &mut self,
        _py: Python<'py>,
        cov: PyReadonlyArray2<f64>,
        psd: PyReadonlyArray2<f64>,
        means: PyReadonlyArray2<f64>,
        projection: PyReadonlyArray2<f64>,
        nc: usize,
        n_components: usize,
    ) {
        self.cov = cov.as_array().to_owned();
        self.psd = psd.as_array().to_owned();
        self.means = means.as_array().to_owned();
        self.projection = projection.as_array().to_owned();
        self.nc = nc;
        self.n_components = n_components;
    }

    fn predict_proba<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<i16>,
    ) -> PyResult<&'py PyArray2<f64>> {
        let x = x.as_array();
        let c_means = &self.means.t();
        let psd = &self.psd;
        let projection = &self.projection;

        let ns_in = x.shape()[1];
        let ns_proj = projection.shape()[1];
        let mut prs = Array2::<f64>::zeros((x.shape()[0], c_means.shape()[0]));
        py.allow_threads(|| {
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
        });
        Ok(&(prs.to_pyarray(py)))
    }

    fn get_cov<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        Ok(&self.cov.to_pyarray(py))
    }
    fn get_projection<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        Ok(&self.projection.to_pyarray(py))
    }
    fn get_means<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        Ok(&self.means.to_pyarray(py))
    }
    fn get_psd<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        Ok(&self.psd.to_pyarray(py))
    }
}
