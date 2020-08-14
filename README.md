# Stella: Side-channel Tool-box for Embedded systems Lazy Leakage Analysis 

Stella contains multiple functionalities allowing to perform side-channel analysis. It implements
various standard methodologies as well as advanced ones. 
Overall, this project implements a high-level Python interface for easy interfacing. The computational intensive 
methods are implemented with compiled languages (C, C++ or Rust) to enable multithreading and machine specific code. 

Btw, stella is a belgian beer too

## Functionalities
The similar functionalities are grouped within the same directories. 
* [evaluation/](evaluation) :
* [estimator/](estimator) :
* [sasca/](sasca):
* [examples/](examples):

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
Add this dir to your PYTHONPATH to call it from everywhere
```
export PYTHONPATH=$PYTHONPATH:/DIR/CONTAINING/STELLA
```
