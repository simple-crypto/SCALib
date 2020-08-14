import numpy as np
from stella.evaluation.snr import SNR
from stella.utils.DataReader import *
from stella.utils.Accumulator import * 
                
def write_snr(TRACES_PREFIX,LABELS_PREFIX,FILE_SNR,
                n_files,
                labels,batch_size=-1,Nc=256,verbose=False):
    labels = np.array_split(labels,len(labels)//batch_size) if batch_size != -1 else np.array([labels])
    snrs_labels = []
    for l in labels:
        # prepare and start the DataReader
        file_read = [[(TRACES_PREFIX+"_%d.npy"%(i),None),(LABELS_PREFIX+"_%d.npz"%(i),["labels"])]   for i in range(n_files)]
        reader = DataReader(file_read,max_depth=5)
        reader.start()
    
        i = 0
        data = reader.queue.get()
        while data is not None:
            # load the file
            traces = data[0]
            labels_f = data[1][0] 

            labels_f = list(filter(lambda x:x["label"] in l,labels_f))
            assert len(labels_f) == len(l)
            
            if i == 0:
                classes = np.zeros((len(l),len(traces[:,0])),dtype=np.uint16)
                indexes = [np.where(l==x["label"])[0][0] for x in labels_f]
                snr = SNR(Nc,len(traces[0,:]),Np=len(labels_f))
            for n,m in enumerate(indexes): classes[n,:] = labels_f[m]["val"]
            snr.fit_u(traces,classes)

            data = reader.queue.get()
            i += 1
        snrs_labels += [{"label":n,"snr":snr._SNR[i,:]} for i,n in enumerate(l)]

    for x in snrs_labels:
        print(x["label"] , str(np.max(x["snr"])))
    np.savez(FILE_SNR,snr=snrs_labels,allow_pickle=True)

def write_poi(FILE_SNR,FILE_POI,
                labels,
                selection_function):

    snrs_labels = np.load(FILE_SNR,allow_pickle=True)["snr"]
    pois_labels = list(map(lambda x: {"poi":selection_function(x["snr"]),"label":x["label"]},snrs_labels))
    np.savez(FILE_POI,poi=pois_labels,allow_pickle=True)

def build_model(TRACES_PREFIX,LABELS_PREFIX,FILE_POI,FILE_MODEL,
                n_files,
                labels,
                func,batch_size=-1):

    pois = np.load(FILE_POI,allow_pickle=True)["poi"]
    models = list(filter(lambda x: x["label"] in labels ,pois))
    models_l = np.array(list(map(lambda x: x["label"], models)))

    labels = np.array_split(labels,len(labels)//batch_size) if batch_size != -1 else np.array([labels])

    for l in labels:
        # prepare and start the DataReader
        file_read = [[(TRACES_PREFIX+"_%d.npy"%(i),None),(LABELS_PREFIX+"_%d.npz"%(i),["labels"])]   for i in range(n_files)]
        reader = DataReader(file_read,max_depth=5)
        reader.start()
 
        indexes_m = [np.where(x==models_l)[0][0] for x in l]
        i = 0
        data = reader.queue.get()
        while data is not None:

            # load the file
            traces = data[0]
            labels_f = data[1][0] 

            labels_f = list(filter(lambda x:x["label"] in l,labels_f))
            assert len(labels_f) == len(l)
            
            if i == 0:
                indexes = [np.where(l==x["label"])[0][0] for x in labels_f]
                for i_m, la_f in zip(indexes_m,labels_f): 
                    models[i_m]["acc_val"] = Accumulator(len(traces[:,0])*n_files,1,dtype=np.uint16)
                    models[i_m]["acc_traces"] = Accumulator(len(traces[:,0])*n_files,len(models[i_m]["poi"]),dtype=np.int16)

            for i_m,la_f in zip(indexes_m,labels_f): models[i_m]["acc_val"].fit(la_f["val"].astype(np.uint16))
            for i_m in indexes_m:
                models[i_m]["acc_traces"].fit(traces[:,models[i_m]["poi"]])
            i+= 1
            data = reader.queue.get()

        for i_m in indexes_m:
            m = models[i_m]
            t = m.pop("acc_traces").get()
            l = m.pop("acc_val").get()[:,0]
            m["model"] = func(t,l,m["label"])
            del t,l
    
    np.savez(FILE_MODEL,model=models,allow_pickle=True)

