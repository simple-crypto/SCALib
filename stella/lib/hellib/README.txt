This program is a free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation.
It is distributed without any warranty of correctness nor fitness for any particular purpose. See the GNU General Public License for more details. 

Welcome to the Histogram Evaluation Library (HEL).

Please refer to the companion paper whenever you use this source code. 
If you have a question, you can send an email to Romain Poussier : romain.poussier@uclouvain.be or poussier.romain@hotmail.fr

This file contains some information on the library.

This version does not include the threaded parallelisation.
It must be compiled with g++ (should use -O3 optimization command) by linking the NTL library (http://www.shoup.net/ntl/).
The NTL library is only used for the histogram convolution, the rest of the code is written in a C way.

The file "main_example.cpp" contains some utilization examples as in the paper.
This file is commented all along to ease the understanding.

We also provide a makefile that should work on linux if the NTL is installed in the standard repository.
