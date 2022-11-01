
import numpy as np

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np

# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API. Therefore we recommend
# always calling "import_array" whenever you "cimport numpy"
np.import_array()

cimport cython

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = int

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int_t DTYPE_t


def getEJrandnums(int n_nums, int seed): 

    cdef np.ndarray[DTYPE_t, ndim=1] out = np.zeros(n_nums, dtype=DTYPE)
    cdef int temp1, temp2, temp3, result
    for i in range(n_nums):
        temp1 = (seed & 0xFFFF) * 0x41A7
        temp2 = (seed >> 16) * 0x41A7 + (temp1 >> 16)
        temp3 = temp2 * 2 >> 16
        temp1 = (temp1 & 0xFFFF) - 0x7FFFFFFF
        temp2 &= 0x7FFF
        temp1 += (temp2 << 16) | (temp2 >> 16) + temp3
        if (temp1 < 0):
            temp1 += 0x7FFFFFFF
        seed = temp1
        result = temp1 & 0xFFFF
        if (result == 0x8000):
            result = 0

        out[i] = result
    

    return out
