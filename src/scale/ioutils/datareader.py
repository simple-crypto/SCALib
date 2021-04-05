import numpy as np
import threading
import queue
import time
class DataReader(threading.Thread):
    def __init__(self,files,keys,maxsize=1,verbose=False):
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
        >>> #for traces in DataReader(files,None):
        >>> #    print(np.mean(traces,axis=0))
        """

        super(DataReader,self).__init__()
        self.stop_event_ = threading.Event()
        self.files_ = files
        self.queue_ = queue.Queue(maxsize=maxsize)
        self.verbose_ = verbose
        self.keys_ = keys

    def run(self):
        for fname in self.files_:
            rets = ()
            if self.verbose_: print("# DataReader: start load ",fname)
            
            read = np.load(fname,allow_pickle=True)
            if self.keys_ is None:
                rets = read
            else:
                for l in self.keys_: rets +=(read[l],)

            if self.verbose_: print("# DataReader: done load ",fname)

            while self.queue_.full() and not self.stop_event_.is_set():
                time.sleep(.2)

            if self.stop_event_.is_set(): break

            self.queue_.put(rets)
        return

    def stop(self):
        r"""Stops the reading thread. This has to be called if the iterator is
        not totally consumed.
        """
        self.stop_event_.set()

    def __iter__(self):
        self.i_ = 0
        self.start()
        return self

    def __next__(self):
        if self.i_ < len(self.files_):
            self.i_ += 1
            return self.queue_.get()
        else:
            raise StopIteration
    def __del__(self):
        del self.queue_
