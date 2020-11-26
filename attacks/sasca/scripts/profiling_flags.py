import numpy as np
import time
from tqdm import tqdm
from stella.evaluation.snr import SNR
from stella.utils.DataReader import *
from stella.utils.Accumulator import *
from sklearn.model_selection import KFold
import sklearn.preprocessing 
def write_snr(TRACES_PREFIX,LABELS_PREFIX,FILE_SNR,
                n_files,
                labels,batch_size=-1,Nc=256,verbose=False,axis_chunks=1,
                traces_extension=".npy",traces_label=None):
    """
        Compute SNR by iterating over files
        - TRACES_PREFIX: the prefix for the traces
        - LABELS_PREFIX: the prefix for the corresponding labels
        - FILE_SNR: file to write the SNR
        - n_files: number of files to compute the SNR on
        - labels: the labels of the variables to profile
        - batch_size: maximum number of variables to store the SNR 
            at the same time in memory. -1 means that all variables at computed 
            in a single pass.
        - Nc: number of classes
        - Verbose: display SNR variable to the standard output
    """
    labels = np.array_split(labels,(len(labels)//batch_size) +1) if batch_size != -1 else np.array([labels])

    snrs_labels = []
    for l in labels:
        if verbose: print("# Batch with ",len(l),"labels")
        # prepare and start the DataReader
        file_read = [[(TRACES_PREFIX+"_%d"%(i)+traces_extension,traces_label),(LABELS_PREFIX+"_%d.npz"%(i),["labels"])]   for i in range(n_files)]
        reader = DataReader(file_read,max_depth=2)
        reader.start()

        i = 0
        data = reader.queue.get()
        while data is not None:
            # load the file
            traces = data[0][0]
            labels_f = data[1][0]

            labels_f = list(filter(lambda x:x["label"] in l,labels_f))
            assert len(labels_f) == len(l)

            if i == 0:
                labels_0 = list(map(lambda x: x["label"],labels_f))
                classes = np.zeros((len(l),len(traces[:,0])),dtype=np.uint16)
                snr = SNR(Nc,len(traces[0,:]),Np=len(labels_f))
            else: #assert current file structured as the first one
                for la_0,la_f in zip(labels_0,labels_f): assert la_0 == la_f["label"]

            for j,la in enumerate(labels_f):
                classes[j,:] = la["val"]

            if i == 0 and traces.dtype != np.int16:
                print("# Caution: traces are casted from ",traces.dtype," to np.int16")
            if traces.dtype != np.int16:
                traces = traces.astype(np.int16)

            snr.fit_u(traces,classes,nchunks=axis_chunks)

            data = reader.queue.get()
            i += 1

        for i,n in enumerate(labels_f):
            at_inf = np.where(~np.isfinite(snr._SNR[i,:]))[0]
 
            if len(at_inf)>0:
                print("# !!!!!!!!! CAUTION, not finite SNR at index",at_inf,". SNR set to zero")
                snr._SNR[i,at_inf] = 0
            if verbose:
                print("# ",n["label"], "max SNR",np.max(snr._SNR[i,:]))
        snrs_labels += [{"label":n["label"],"snr":snr._SNR[i,:]} for i,n in enumerate(labels_f)]

    np.savez(FILE_SNR,snr=snrs_labels,allow_pickle=True)

def write_poi(FILE_SNR,FILE_POI,
                labels,
                selection_function):

    snrs_labels = np.load(FILE_SNR,allow_pickle=True)["snr"]
    pois_labels = list(map(lambda x: {"poi":selection_function(x["snr"]),"label":x["label"]},snrs_labels))
    np.savez(FILE_POI,poi=pois_labels,allow_pickle=True)

def eval_model(TRACES_PREFIX,LABELS_PREFIX,FILE_EVAL,FILE_MODEL,
                n_files,
                verbose=False,
                traces_extension=".npy",traces_label=None,normalize=False):

    models = np.load(FILE_MODEL,allow_pickle=True)["model"]
    for m in models: m["information"] = 0

    # prepare and start the DataReader
    file_read = [[(TRACES_PREFIX+"_%d"%(i)+traces_extension,traces_label),(LABELS_PREFIX+"_%d.npz"%(i),["labels"])]   for i in range(n_files)]
    reader = DataReader(file_read,max_depth=2)
    reader.start()

    i = 0
    data = reader.queue.get()

    while data is not None:
        # load the file
        traces = data[0][0]
        labels = data[1][0]

        ns,_ = traces.shape

        labels_f = list(map(lambda x:x["label"],labels))
        assert len(labels_f) > len(models)

        for m in tqdm(models,desc="# eval models"):
            index = labels_f.index(m["label"])
            t = traces[:,m["poi"]]
            if normalize:
                t = ((t - np.mean(t,axis=0))/np.std(t,axis=0))
                #t = sklearn.preprocessing.normalize(t)
            prs = m["model"].predict_proba(t)
            prs[np.where(prs<1E-200)] = 1E-200
            nb = len(prs[0,:])
            m["nb"] = np.log2(nb)
            m["information"] += m["nb"] + np.mean(np.log2(prs[np.arange(ns),labels[index]["val"]]))
        i+= 1
        del traces,labels
        data = reader.queue.get()
    for m in models:m["information"]/=i

    out = []
    for m in models: out.append({"label":m["label"],"information":m["information"]})

    if verbose:
        for o in out:
            print("label", o["label"],"-> information %.3f"%(o["information"]))

    np.savez(FILE_EVAL,information=out,allow_pickle=True)



def build_model(TRACES_PREFIX,LABELS_PREFIX,FILE_POI,FILE_MODEL,
                n_files,
                labels,
                func,
                verbose=False,
                batch_size=-1,
                traces_extension=".npy",traces_label=None,normalize=False):

    pois = np.load(FILE_POI,allow_pickle=True)["poi"]
    pois_l = list(map(lambda x:x["label"],pois))
    nlabels = len(labels)
    labels = np.array_split(labels,(len(labels)//batch_size)+1) if batch_size != -1 else np.array([labels])


    models = []
    done_models = 0
    for l in labels:

        # prepare and start the DataReader
        file_read = [[(TRACES_PREFIX+"_%d"%(i)+traces_extension,traces_label),(LABELS_PREFIX+"_%d.npz"%(i),["labels"])]   for i in range(n_files)]
        reader = DataReader(file_read,max_depth=2)
        reader.start()

        i = 0
        data = reader.queue.get()
        while data is not None:

            # load the file
            traces = data[0][0]
            labels_f = data[1][0]

            labels_f = list(filter(lambda x:x["label"] in l,labels_f))
            assert len(labels_f) == len(l)

            if i == 0:
                models_l = labels_f
                for m_f in models_l:
                    index = pois_l.index(m_f["label"])
                    m_f["poi"] = pois[index]["poi"]
                    m_f["acc_val"] = Accumulator(len(traces[:,0])*n_files,1,dtype=np.uint16)
                    m_f["acc_traces"] = Accumulator(len(traces[:,0])*n_files,len(m_f["poi"]),dtype=traces.dtype)
            else: # check that all the files as structed in the same way as the first one
                for m,la_f in zip(models_l,labels_f): assert m["label"] == la_f["label"]

            for m,la_f in zip(models_l,labels_f): m["acc_val"].fit(la_f["val"].astype(np.uint16))
            for m in models_l: m["acc_traces"].fit(traces[:,m["poi"]])

            i+= 1
            del traces,labels_f
            data = reader.queue.get()

        for m in models_l:
            t = m.pop("acc_traces").get()
            if normalize:
                t = ((t - np.mean(t,axis=0))/np.std(t,axis=0))
                #t = sklearn.preprocessing.normalize(t)
            l = m.pop("acc_val").get()[:,0]
            m.pop("val")
            start = time.time()
            m["model"] = func(t,l,m["label"])
            done_models +=1
            if verbose:
                print("# Done model (%d/%d)"%(done_models,nlabels),m["label"]," Elapsed time %.3f"%(time.time()-start))
            del t,l

        models += models_l

    np.savez(FILE_MODEL,model=models,allow_pickle=True)

def estimate_pi(TRACES_PREFIX,LABELS_PREFIX,FILE_POI,
                PI_PREFIX,
                n_files,
                labels,
                kfold=10,
                npts=10,
                Nb = 8,
                batch_size=-1,verbose=False,
                traces_extension=".npy",traces_label=None):

    pois = np.load(FILE_POI,allow_pickle=True)["poi"]
    pois_l = list(map(lambda x:x["label"],pois))

    labels = np.array_split(labels,(len(labels)//batch_size)+1) if batch_size != -1 else np.array([labels])

    for l in labels:
        l_ls = list(map(lambda x:x["label"],l))

        # prepare and start the DataReader
        file_read = [[(TRACES_PREFIX+"_%d"%(i)+traces_extension,traces_label),(LABELS_PREFIX+"_%d.npz"%(i),["labels"])]   for i in range(n_files)]
        reader = DataReader(file_read,max_depth=2)
        reader.start()

        i = 0
        data = reader.queue.get()
        while data is not None:

            # load the file
            traces = data[0][0]
            labels_f = data[1][0]

            labels_f = list(filter(lambda x:x["label"] in l_ls,labels_f))
            assert len(labels_f) == len(l)

            if i == 0:
                models_l = labels_f
                for m_f in models_l:
                    index = pois_l.index(m_f["label"])
                    m_f["poi"] = pois[index]["poi"]
                    m_f["acc_val"] = Accumulator(len(traces[:,0])*n_files,1,dtype=np.uint16)
                    m_f["acc_traces"] = Accumulator(len(traces[:,0])*n_files,len(m_f["poi"]),dtype=traces.dtype)

                    index = l_ls.index(m_f["label"])
                    m_f["models"] = l[index]["models"]

            else: # check that all the files as structed in the same way as the first one
                for m,la_f in zip(models_l,labels_f): assert m["label"] == la_f["label"]

            for m,la_f in zip(models_l,labels_f): m["acc_val"].fit(la_f["val"].astype(np.uint16))
            for m in models_l: m["acc_traces"].fit(traces[:,m["poi"]])

            i+= 1
            data = reader.queue.get()

        # all data have been loaded. Computing PI
        for m in models_l:
            # Extract the labels and traces
            t = m.pop("acc_traces").get()
            l = m.pop("acc_val").get()[:,0]
            m.pop("val")

            # init the storage for PI
            for single_model in m["models"]: single_model["pi"] = np.zeros((kfold,npts))
            for single_model in m["models"]: single_model["ntrain"] = np.zeros((kfold,npts))

            kf = KFold(n_splits=kfold)
            start = time.time()
            if verbose: print("\nvariable",m["label"], "with tags",[s["method_tag"] for s in m["models"]])

            for fold,(train_index, test_index) in enumerate(kf.split(l)):
                if verbose: print("fold ",fold+1,"/",kfold, "for variable" ,m["label"],"|   Elapsed time %.3f[s]"%(time.time()-start))

                if npts == 1:
                    steps = [len(train_index)]
                else:
                    steps = np.logspace(np.log10((2**Nb) * 20),np.log10(len(train_index)),npts,dtype=int)

                for j,single_model in enumerate(m["models"]):
                    for n,s in enumerate(steps):
                        pdf_esti = single_model["func"](t[train_index[:s]],
                                        l[train_index[:s]],
                                        single_model["arg"])
                        prs = pdf_esti.predict_proba(t[test_index])
                        prs[np.where(prs<1E-100)] = 1E-100
                        del pdf_esti

                        single_model["ntrain"][fold,n] = s
                        single_model["pi"][fold,n] = Nb + np.mean(np.log2(prs[np.arange(len(test_index)),l[test_index]]))
            del t,l
            for single_model in m["models"]: single_model.pop("func")

            if verbose:
                print("Max PI:")
                for single_model in m["models"]: print("    ",single_model["method_tag"]," -> %.5f"%(np.mean(single_model["pi"][:,-1])))

            np.savez(PI_PREFIX+"_"+m["label"]+".npz",label=m["label"],
                    poi=m["poi"],
                    models=m["models"],allow_pickle=True)
