
import theano
import numpy as np

from theano import gof
from theano import tensor as T
from theano.tensor.blas import ldflags


class MKL_Tanh(gof.Op):
    __props__ = ()

    def __init__(self):
        super(MKL_Tanh, self).__init__()

    def make_node(self, inp):
        x = T.as_tensor_variable(inp)
        return gof.Apply(self, [x], [x.type()])

    def c_headers(self):
        return ['<mkl.h>']

    def c_libraries(self):
        return ldflags()

    def c_code(self, node, name, inputs, outputs, sub):
        x, = inputs
        z, = outputs

        fail = sub['fail']

        if node.inputs[0].type.dtype == 'float32':
            dtype = 's'
            d = 'float'
        elif node.inputs[0].type.dtype == 'float64':
            dtype = 'd'
            d = 'double'
        else:
            raise TypeError('Tanh: dtype error')

        ccode = """
            %(d)s* ptr = NULL;

            PyArrayObject* x_src = NULL;
            if (!PyArray_IS_C_CONTIGUOUS(%(x)s)) {
                printf(\"Warning: need cast x to C-Contiguous array\\n\");
                x_src = (PyArrayObject*) PyArray_ContiguousFromAny((PyObject*)%(x)s,
                                                PyArray_TYPE(%(x)s),
                                                PyArray_NDIM(%(x)s),
                                                PyArray_NDIM(%(x)s));

                if (!x_src) {
                    PyErr_SetString(PyExc_RuntimeError, \"MKL_Tanh: fail to cast x to C-Contiguous array\");
                    goto mkl_tanh_fail;
                }
                ptr = (%(d)s*) PyArray_DATA(x_src);
            } else {
                ptr = (%(d)s*) PyArray_DATA(%(x)s);
            }

            if (%(z)s == NULL) {
                %(z)s = (PyArrayObject*) PyArray_ZEROS(PyArray_NDIM(%(x)s), PyArray_DIMS(%(x)s), PyArray_TYPE(%(x)s), 0);
            }

            if (%(z)s == NULL) {
                PyErr_SetString(PyExc_RuntimeError, \"MKL_Tanh: fail to create output array\");
                goto mkl_tanh_fail;
            }

            vmlSetMode(vmlGetMode() & 0xFFFFFFF0 | VML_EP);
            size_t total = PyArray_SIZE(%(x)s);
            v%(dtype)sTanh(total, ptr, (%(d)s*)PyArray_DATA(%(z)s));

            mkl_tanh_fail:
            Py_XDECREF(x_src);
        """ % locals()

        return ccode

    def grad(self, inp, grads):
        x, = inp
        gz, = grads

        fw = MKL_Tanh()(x)
        output = 1. - fw ** 2
        output = output * gz
        return output

    def c_code_cache_version(self):
        return (1, 0, 0)


mkl_tanh = MKL_Tanh()
