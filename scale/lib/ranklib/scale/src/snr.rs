use ndarray::{Array2, Array3, Axis, Zip};
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
pub struct SNR {
    s: Array3<i64>,
    s2: Array3<i64>,
    nc: Array2<u64>,

    np: usize,
    ns: usize,
}
#[pymethods]
impl SNR {
    #[new]
    fn new(nc: usize, ns: usize, np: usize) -> Self {
        SNR {
            s: Array3::<i64>::zeros((np, nc, ns)),
            s2: Array3::<i64>::zeros((np, nc, ns)),
            nc: Array2::<u64>::zeros((np, nc)),

            ns: ns,
            np: np,
        }
    }

    fn update(
        &mut self,
        py: Python,
        traces: PyReadonlyArray2<i16>, // (len,N_sample)
        y: PyReadonlyArray2<u16>,      // (Np,len)
    ) {
        let x = traces.as_array();
        let y = y.as_array();
        let s = &mut self.s;
        let s2 = &mut self.s2;
        let nc = &mut self.nc;
        py.allow_threads(|| {
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
        });
    }
    fn get_snr<'py>(&mut self, py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        let mut snr = Array2::<f64>::zeros((self.np, self.ns));
        let s = &self.s;
        let s2 = &self.s2;
        let nc = &self.nc;
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
        Ok(&(snr.to_pyarray(py)))
    }
}
