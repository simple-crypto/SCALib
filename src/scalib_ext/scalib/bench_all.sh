RUSTFLAGS="-C target-cpu=native" RAYON_NUM_THREADS=12 taskset --cpu-list 0-11 cargo bench --bench snr_update_threads -- --save-baseline 12_thread_native
RUSTFLAGS="-C target-cpu=native" taskset --cpu-list 0 cargo bench --bench snr_update -- --save-baseline single_thread_native
RAYON_NUM_THREADS=12 taskset --cpu-list 0-11 cargo bench --bench snr_update_threads -- --save-baseline 12_thread
taskset --cpu-list 0 cargo bench --bench snr_update -- --save-baseline single_thread
