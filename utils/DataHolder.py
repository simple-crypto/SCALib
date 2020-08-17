import numpy as np

class DataHolder():
    """
        DataHolder maintains an array and returns references to its rows
        incrementally. It allows to update values within many VNodes (i.e. value
        or distribution) by accessing updating a single update.
    """

    def __init__(self,data):
        """
            data: the numpy array to hold. rows of data will be distributed
        """
        m,_ = data.shape
        self._data = data
        self._l = m
        self._i = 0

    def update(self,fresh_data):
        """
            update the data held with fresh_data
        """
        if not fresh_data.shape == self._data.shape:
            raise Exception("fresh_data does not have correct shape")
        if not fresh_data.dtype == self._data.dtype:
            raise Exception("fresh data does not have the correct dtype")

        self._data[:] = fresh_data
        self._i = 0

    def reset_cnt(self):
        """
            reset pointer to the first row
        """
        self._i = 0

    def get_row(self):
        """
            return the next row of the held data
        """
        if self._i >= self._l:
            raise Exception("Too many rows are requested")
        ret = self._data[self._i]
        self._i += 1
        return ret
