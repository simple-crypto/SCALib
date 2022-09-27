use hytra::TrAdder;
use indicatif::{ProgressBar, ProgressFinish, ProgressStyle};
use std::thread;
use std::time::Duration;

pub(crate) fn with_progress<F, T>(f: F, n_iter: u64, pb_msg: &'static str) -> T
where
    F: FnOnce(&TrAdder<u64>) -> T + Send,
    T: Send,
{
    let it_cnt: TrAdder<u64> = TrAdder::new();
    let finished = std::sync::atomic::AtomicBool::new(false);
    crossbeam_utils::thread::scope(|s| {
        let finished_ref = &finished;
        let it_cnt_ref = &it_cnt;
        // spawn progress bar thread
        let pb_thread_handle = s.spawn(move |_| {
            let pb = ProgressBar::new(n_iter);
            pb.set_style(
                ProgressStyle::default_spinner()
                    .template("{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] (ETA {eta})")
                    .on_finish(ProgressFinish::AndClear),
            );
            pb.set_message(pb_msg);
            while !finished_ref.load(std::sync::atomic::Ordering::Relaxed) {
                pb.set_position(it_cnt_ref.get());
                thread::park_timeout(Duration::from_millis(50));
            }
            pb.finish_and_clear();
        });

        // spawn computing thread
        let res = s.spawn(move |_| f(it_cnt_ref)).join().unwrap();
        finished_ref.store(true, std::sync::atomic::Ordering::Relaxed);
        pb_thread_handle.thread().unpark();
        res
    })
    .unwrap()
}
