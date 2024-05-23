# "How machines learn" for Java developers 
## AKA implementation from scratch of automatic differentiation and gradient descent in Java to understand how it works

This repo is the code I've written to understand how Tensorflow/PyTorch models can "learn" using automatic differentiation and gradient descent.

I'm considering writing an article on Medium.com or some other platform like [mokabyte.it](https://www.mokabyte.it/autore/cristiano-costantini/) to talk about it.

## "Tensor" Branch
While the main branch focuses on SGD with scalar values, I'm exploring how to change the code to support N Dimensional arrays in the [this](https://github.com/cristcost/java-gradient-descent/tree/tensor) branch.

**Current status**: implemented support for Tensor operations and implemented MNIST dataset (handwritten digits) training loop in pure Java.

**Next**: Code in this branch to be refactored and styled for presentation.