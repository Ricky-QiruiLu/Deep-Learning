# RluTorch

RluTorch is my personal deep learning library. It includes a linear algebra library and interfaces for deep learning components.

1. [Linear algebra library](https://github.com/Ricky-QiruiLu/deep_learning/tree/main/RluTorch/python/needle/backend_ndarray)

    This small package creates NDArray for backend. It is a generic ND array class, and it includes all the necessary operations for network constructions.The backend could be on either CPU and GPU.

    The backend could be on either CPU and GPU. Therefore, I also implement these operations for [CPU](https://github.com/Ricky-QiruiLu/deep_learning/blob/main/RluTorch/src/ndarray_backend_cpu.cc) & [GPU](https://github.com/Ricky-QiruiLu/deep_learning/blob/main/RluTorch/src/ndarray_backend_cuda.cu) with c++.

2. [Deep learning components](https://github.com/Ricky-QiruiLu/deep_learning/tree/main/RluTorch/python/needle/backend_ndarray)

    This small package create components for deep learning networks. It is similar to PyTorch, which creates layers with Tensors. 

    - [autograd.py](https://github.com/Ricky-QiruiLu/deep_learning/blob/main/RluTorch/python/needle/autograd.py) - The core data structures for Operations and Tensors. It also create and update the computation graph for the automatic differentiation framework.

    - [init.py](https://github.com/Ricky-QiruiLu/deep_learning/blob/main/RluTorch/python/needle/init.py) - It has some general initializations for the weights.

    - [ops.py](https://github.com/Ricky-QiruiLu/deep_learning/blob/main/RluTorch/python/needle/ops.py) - It includes all the global operations (Op) for Tensors

     - [optim.py](https://github.com/Ricky-QiruiLu/deep_learning/blob/main/RluTorch/python/needle/optim.py) - It creates Optimizers for training 
