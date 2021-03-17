import numpy as np
import threading
import queue
import time
class DataReader(threading.Thread):
    def __init__(self,files,labels,verbose=False):
        r"""Iterator reading a list of files (.npy or .npz). It starts an
        independent threads that loads the files.

        Parameters
        ----------
        files : list
            The list of files to read with np.load().
        labels : list
            The list of labels to return in the loaded file. If not specified,
            the output of np.load() is returned.
        max_depth : int
            The size of internal queue.
        verbose : bool
            Verbose flag. 
        
        Returns
        --------
        ret
            If labels is None, return the output of np.load(). Else, it returns 
            a tuple with all the labels that have been loaded.
            
        Examples
        --------
        >>> files = ["traces_%d.npy"%(i) for i in range(10)]
        >>> for traces in DataReader(files,None):
                print(np.mean(traces,axis=0))
        """

        super(DataReader,self).__init__()
        self._stop_event = threading.Event()
        self.files = files
        self.queue = queue.Queue(maxsize=2)
        self._verbose = verbose
        self.labels = labels
        self._thread_stop = False

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
        del self.queue
        return

    def stop(self):
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
