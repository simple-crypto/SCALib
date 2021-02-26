import numpy as np

class Accumulator():
    def __init__(self,l,Ns,dtype=np.int16):
        """
            This object stores the fitted inputs withing and array

            l: number of lines in the array
            Ns: number of columns
            dtype: data type of the numpy array
        """

        self._array = np.zeros((l,Ns),dtype=dtype)
        self._Ns = Ns
        self._l = l
        self._i = 0
    def fit(self,array):
        """
            fill the owned array with this one
        """
        if array.ndim == 1:
            array = np.expand_dims(array,1)
        lx,ly = array.shape
        if ly != self._Ns:
            raise Exception("Does not have the correct shape")
        m = self._i + lx
        if m > self._l:
            raise Exception("Too much data fitted")

        self._array[self._i:m,:] = array
        self._i = m

    def get(self):
        return self._array[:self._i]

    def __del__(self):
        del self._array
