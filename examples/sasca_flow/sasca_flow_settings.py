import os
import numpy as np
DIR_PROFILE_TRACES = "/tmp/stella_ex/profile/traces/"
DIR_ATTACK_TRACES = "/tmp/stella_ex/attack/traces/"
DIR_PROFILE_LABELS = "/tmp/stella_ex/profile/labels/"
DIR_SNR = "/tmp/stella_ex/profile/snr/"
DIR_POI = "/tmp/stella_ex/profile/poi/"
DIR_MODEL = "/tmp/stella_ex/profile/model/"
tag = "example"

PREFIX_PROFILE_TRACES = DIR_PROFILE_TRACES+tag+"_traces"
PREFIX_ATTACK_TRACES = DIR_ATTACK_TRACES+tag+"_traces"
PREFIX_PROFILE_LABELS = DIR_PROFILE_LABELS+tag+"_labels"
FILE_SNR = DIR_SNR+tag+"_snr.npz"
FILE_POI = DIR_POI+tag+"_poi.npz"
FILE_MODEL = DIR_MODEL+tag+"_model.npz"

nfile_profile = 50
nfile_attack = 10
ntraces = 100000
ntraces_attack = 2000

os.system("mkdir -p "+DIR_PROFILE_TRACES)
os.system("mkdir -p "+DIR_PROFILE_LABELS)
os.system("mkdir -p "+DIR_SNR)
os.system("mkdir -p "+DIR_POI)
os.system("mkdir -p "+DIR_MODEL)

os.system("mkdir -p "+DIR_ATTACK_TRACES)
noise = 1
Nk = 256
sbox = np.random.permutation(range(Nk)).astype(np.uint8)

