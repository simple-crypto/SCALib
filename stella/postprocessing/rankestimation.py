import stella.lib.ranklib as ranklib

def rank_nbin(costs,key,nbins,method="hist"):
    return ranklib.rank_nbin(costs,key,nbins,2,method)

def rank_accuracy(costs,key,acc,method="hist"):
    return ranklib.rank_accuracy(costs,key,acc,2,method)
