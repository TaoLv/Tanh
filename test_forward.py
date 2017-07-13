
import time
import theano
import numpy as np

from theano import tensor as T

from mkl_tanh import mkl_tanh


def new_tanh(inp, times=1):
    x = T.dmatrix('x')
    z = mkl_tanh(x)
    f = theano.function([x], z)

    # warm up
    result = f(inp)

    tic = time.time()
    for i in range(times):
        result = f(inp)
    toc = time.time()
    return result, (toc - tic)


def theano_tanh(inp, times=1):
    x = T.dmatrix('x')
    z = T.tanh(x)
    f = theano.function([x], z)

    result = f(inp)

    tic = time.time()
    for i in range(times):
        result = f(inp)
    toc = time.time()
    return result, (toc - tic)


def numpy_tanh(inp, times=1):
    result = np.tanh(inp)

    tic = time.time()
    for i in range(times):
        result = np.tanh(inp)
    toc = time.time()
    return result, (toc - tic)


if __name__ == '__main__':
    a = np.random.rand(1024, 1024).astype(np.float64)
    loop = 1000
    res1, t1 = new_tanh(a, loop)
    res2, t2 = theano_tanh(a, loop)
    res3, t3 = numpy_tanh(a, loop)

    err1 = res1 - res2
    err2 = res1 - res3
    err3 = res2 - res3

    print(err1.max(), err2.max(), err3.max())
    print(t1, t2, t3)
