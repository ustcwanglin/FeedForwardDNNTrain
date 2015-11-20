# FeedForwardDNNTrain
=======

The aim of the FeedForwardDNNTrain project is to perform basic training of 
feed forward deep nueral network, based on CUDAMat, which provides a Python 
matrix class that performs matrix calculations on CUDA-enabled GPUs from 
Python. 

Example:

```python 
import numpy as np 
import cudamat as cm 

cm.cublas_init()

# create two random matrices and copy them to the GPU
a = cm.CUDAMatrix(np.random.rand(32, 256))
b = cm.CUDAMatrix(np.random.rand(256, 32))

# perform calculations on the GPU
c = cm.dot(a, b)
d = c.sum(axis = 0)

# copy d back to the host (CPU) and print
print(d.asarray())
```

Download
--------

You can obtain the latest release from the repository by typing:

```bash
git clone https://github.com/ustcwanglin/FeedForwardDNNTrain.git
```

Installation
------------

FeedForwardDNNTrain uses setuptools and can be installed via pip.
For details, please see [INSTALL.md](INSTALL.md).

Development
-----------

If you want to contribute new features or improvements, you're welcome to fork
FeedForwardDNNTrain on github and send us your pull requests!
Please see [CONTRIBUTE.md](CONTRIBUTE.md) if you need any help with that.
