import numpy as np
import threading
import queue
import time
class DataReader(threading.Thread):
    def __init__(self,files,keys,maxsize,verbose=False):
        r"""Iterator reading a list of files (.npy or .npz) with an independent
        thread.

        Parameters
        ----------
        files : list
            The list of files to read with np.load().
        keys : list
            The list of keys to be loaded from the files. If None, all the
            loaded data are returned. 
        maxsize : int
            The size of internal queue.
        verbose : bool
            Verbose flag. 
        
        Returns
        --------
        ret : tuple
            A tuple containing the data for the requested keys. 
            
        Examples
        --------
        >>> files = ["data_%d.npy"%(i) for i in range(10)]
        >>> for traces in DataReader(files,None):
                print(np.mean(traces,axis=0))
        """

        super(DataReader,self).__init__()
        self._stop_event = threading.Event()
        self.files = files
        self.queue = queue.Queue(maxsize=maxsize)
        self._verbose = verbose
        self.labels = labels

    def run(self):
        for fname in self.files:
            rets = ()
            if self._verbose: print("# DataReader: start load ",fname)
            
            read = np.load(fname,allow_pickle=True)
            if self.labels is None:
                rets = read
            else:
                for l in self.labels: rets +=(read[l],)

            if self._verbose: print("# DataReader: done load ",fname)

            while self.queue.full() and not self._stop_event.is_set():
                time.sleep(.2)

            if self._stop_event.is_set(): break

            self.queue.put(rets)
        return

    def stop(self):
        r"""Stops the reading thread. This has to be called if the iterator is
        not totally consumed.
        """
        self._stop_event.set()

    def __iter__(self):
        self._i = 0
        self.start()
        return self

    def __next__(self):
        if self._i < len(self.files):
            self._i += 1
            return self.queue.get()
        else:
            raise StopIteration
    def __del__(self):
        del self.queue
