import numpy as np


def read_inlist(inlist_file):
    testarr = np.loadtxt(inlist_file, dtype=str)
    n_components = (testarr.shape[1]-1) // 4
    filenames = np.loadtxt(inlist_file, usecols=(0,), dtype=str)
    rv_sets = []
    weight_sets = []
    fitflag_sets = []
    subtractflag_sets = []
    for i in range(n_components):
        rv_sets.append(np.loadtxt(inlist_file, usecols=(1+i*4, )))

        weight_sets.append(np.loadtxt(inlist_file, usecols=(2+i*4, )))

        fitflag = np.loadtxt(inlist_file, usecols=(3+i*4, ), dtype=str)
        fitflagtrue = (fitflag == "true") | (fitflag == "True") | (fitflag == "1")
        fitflag_sets.append(fitflagtrue)

        subtract = np.loadtxt(inlist_file, usecols=(4+i*4, ), dtype=str)
        subtracttrue = (subtract == "true") | (subtract == "True") | (subtract == "1")
        subtractflag_sets.append(subtracttrue)
    return filenames, rv_sets, weight_sets, fitflag_sets, subtractflag_sets

