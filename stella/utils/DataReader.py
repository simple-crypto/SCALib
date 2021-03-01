import threading
import queue
import numpy as np

class DataReader(threading.Thread):
    def __init__(self,file_read,max_depth=1,verbose=False):
        super(DataReader,self).__init__()
        self.file_read = file_read
        self.max_depth = max_depth
        self.queue = queue.Queue(maxsize=max_depth)
        self._verbose = verbose

    def run(self):
        for files in self.file_read:
            rets = ()
            for fname,labels in files:
                if self._verbose: print("# DataReader: start load ",fname)
                read = np.load(fname,allow_pickle=True)
                if labels is None:
                    rets +=([read],)
                else:
                    rets +=([read[l] for l in labels],)
                if self._verbose: print("# DataReader: done load ",fname)
            self.queue.put(rets)
        self.queue.put(None)
        return

    def __iter__(self):
        self._i = 0
        print("start")
        self.start()
        return self

    def __next__(self):
        if self._i < len(self.file_read):
            self._i += 1
            return self.queue.get()
        else:
            raise StopIteration
