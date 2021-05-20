use ndarray::{ArrayView2, ArrayViewMut2};

pub fn opt_dgemm(
    a: ArrayView2<f64>,
    b: ArrayView2<f64>,
    mut c: ArrayViewMut2<f64>,
    alpha: f64,
    beta: f64,
    num_threads: u32,
) {
    let (m, k) = a.dim();
    let (k2, n) = b.dim();
    assert_eq!(k, k2);
    assert_eq!((m, n), c.dim());
    if cfg!(feature = "blis") && num_threads != 0 {
        #[cfg(feature = "blis")]
        unsafe {
            let rsa = a.stride_of(ndarray::Axis(0));
            let csa = a.stride_of(ndarray::Axis(1));
            let rsb = b.stride_of(ndarray::Axis(0));
            let csb = b.stride_of(ndarray::Axis(1));
            let rsc = c.stride_of(ndarray::Axis(0));
            let csc = c.stride_of(ndarray::Axis(1));
            let mut rntm = blis_sys::rntm_t {
                auto_factor: true,
                num_threads: num_threads as i64,
                thrloop: [-1; 6usize],
                pack_a: false,
                pack_b: false,
                l3_sup: true,
                sba_pool: std::ptr::null_mut(),
                membrk: std::ptr::null_mut(),
            };
            rntm.thrloop[0] = 1;
            blis_sys::bli_dgemm_ex(
                blis_sys::trans_t_BLIS_NO_TRANSPOSE,
                blis_sys::trans_t_BLIS_NO_TRANSPOSE,
                m as blis_sys::dim_t,
                n as blis_sys::dim_t,
                k as blis_sys::dim_t,
                &alpha as *const _ as *mut _,
                a.as_ptr() as *mut _,
                rsa as blis_sys::dim_t,
                csa as blis_sys::dim_t,
                b.as_ptr() as *mut _,
                rsb as blis_sys::dim_t,
                csb as blis_sys::dim_t,
                &beta as *const _ as *mut _,
                c.as_mut_ptr(),
                rsc as blis_sys::dim_t,
                csc as blis_sys::dim_t,
                blis_sys::bli_gks_query_cntx(),
                &mut rntm,
            );
        }
    } else {
        ndarray::linalg::general_mat_mul(alpha, &a, &b, beta, &mut c);
    }
}
