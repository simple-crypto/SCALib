# Stella: Side-channel Tool-box for Embedded systems Lazy Leakage Analysis 
(btw, stella is a belgian beer too ;))

## Install

install rust

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

```
Use rust nightly version by default
```
rustup toolchain install nightl
rustup default nightly
```
Build rust code
```
cd rust_stella
cargo build --release
cd ..
```