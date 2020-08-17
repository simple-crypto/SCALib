# Stella: Side-channel Tool-box for Embedded systems Lazy Leakage Analysis 

Stella contains multiple functionalities allowing to perform side-channel analysis. It implements
various standard methodologies as well as advanced ones. 
Overall, this project implements a high-level Python interface for easy interfacing. The computational intensive 
methods are implemented with compiled languages (C, C++ or Rust) to enable multithreading and machine specific code.

When applicable, the library state can be updated with fresh samples. This allow to analyse large datasets or to evaluate
metric on the fly without storing them (i.e. TVLA or SNR).

Btw, stella is a belgian beer too.

## Functionalities
The similar functionalities are grouped within the same directories. 
* [evaluation/](evaluation) : Contains methods to evaluate the amount of information leaking from the device. Namely, it implements high-order univariate TVLA, (high-order) SNR and information computation.
* [estimator/](estimator) : Contains distribution estimation methods as well as dimensionality reduction techniques. These are the template constructions methods in template attacks. Other methods can also easily be interface to the project. 
* [attacks/](attacks): Contains various attacks. It includes high-order CPA (MCP-CPA) and SASCA. The latests is a generic representation of template attacks which represents the sensible variables within a factor graph.
* [postprocessing/](postprocessing) : Analysis tools for the output of attacks. At the moment, it only contains rank estimation. 
* [utils/](utils) : Contains functionalities to easy the interface with a dataset.  
* [examples/](examples) : Contains examples for multiple functionalities of Stella. Some examples are in [https://jupyter.org/](Jupyter-Notebook).

## Install
Once you have cloned the projects, many libraries have to be compiled. Before compiled those, you 
have to add the Stella directory to the PYTHONPATH (i.e. in .bashrc)
```
export PYTHONPATH=$PYTHONPATH:/DIR/CONTAINING/STELLA
```
This allows to use Stella from all your systems. You can also install the Python required packages by running
```
pip install --user -r requirements.txt
```

!! Stella has only been tested on some GNU/Linux operating systems. Feedbacks from Windows and MacOS users are welcome !! 
### Rust library
The Rust library is used in many functionalities and must there be installed by all the users.
The first step is to install rust for the user. This is done by running

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Some features of the library are only available on the nightly versions. It is set as default by running
```
rustup toolchain install nightly
rustup default nightly
```
Now that Rust is installed, you can build the code with (this can take a few minutes for the first compilation (it compiles all the dependencies)).
```
cd lib/rust_stella
cargo build --release
```

### C and C++ libraries
The SASCA and templates attacks requires C code located in [libbp](lib/libbp). To build the code, you can run in that directory
```
make
```
The second libraries is [Hel lib](https://perso.uclouvain.be/fstandae/PUBLIS/172.zip) that Stella uses for rank estimation (key enumeration is on feature to develop).
You will need the NTL Library too that you can install with different package manager 
```
sudo apt-get install libntl-dev
sudo pacman -S ntl 
```

## Contribute 

To contribute, you can do pull requests or contact Olivier Bronchain by email ( olivier dot bronchain at uclouvain dot be). We maintain the master with a stable version. develop branch aims at adding new featurs at an increased risk of bugs.


