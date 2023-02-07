use hytra::TrAdder;
use indicatif::{ProgressBar, ProgressFinish, ProgressStyle};
use std::thread;
use std::time::Duration;

pub(crate) fn with_progress<F, T>(
    f: F,
    n_iter: u64,
    pb_msg: &'static str,
    config: &crate::Config,
) -> T
where
    F: FnOnce(&TrAdder<u64>) -> T + Send,
    T: Send,
{
    let it_cnt: TrAdder<u64> = TrAdder::new();
    let finished = std::sync::atomic::AtomicBool::new(false);
    thread::scope(|s| {
        let finished_ref = &finished;
        let it_cnt_ref = &it_cnt;
        // spawn progress bar thread
        let pb_thread_handle = config.show_progress.then(|| {
            s.spawn(move || {
                // Let's first wait for at least config.progress_min_time (unless
                // finished is set in the meantime).
                let start_init_wait = std::time::Instant::now();
                loop {
                    let elapsed = start_init_wait.elapsed();
                    if elapsed >= config.progress_min_time {
                        break;
                    }
                    thread::park_timeout(config.progress_min_time - elapsed);
                    if finished_ref.load(std::sync::atomic::Ordering::Acquire) {
                        return;
                    }
                }
                // Let's now create the progress bar.
                // indicatif::ProgressBar does not seem to offer a way to set the
                // start time, so we ignore the slight start timing offset.
                let pb = ProgressBar::new(n_iter)
                    .with_style(
                        ProgressStyle::default_spinner()
                            .template("{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] (ETA {eta})")
                            .unwrap(),
                    )
                    .with_finish(ProgressFinish::AndClear)
                    .with_message(pb_msg)
                    .with_position(it_cnt_ref.get());
                while !finished_ref.load(std::sync::atomic::Ordering::Acquire) {
                    pb.set_position(it_cnt_ref.get());
                    thread::park_timeout(Duration::from_millis(50));
                }
                pb.finish_and_clear();
            })
        });

        let res = f(it_cnt_ref);
        finished_ref.store(true, std::sync::atomic::Ordering::Release);
        // There is no race condition here: park always consumes the token, and
        // unpark always produces it, independently of whether the target
        // thread is running of not.
        pb_thread_handle.map(|handle| handle.thread().unpark());
        res
    })
}
