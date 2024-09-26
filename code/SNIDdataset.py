#from __future__ import division
import numpy as np
import SNIDsn as snid
import matplotlib.pyplot as plt
from collections import OrderedDict
import pickle

def savePickle(path, dataset, protocol=2):
    """
    Saves a SNIDdataset object using pickle.

    Parameters
    ----------
    path : string
    dataset : SNIDdataset object
    protocol : int
        Pickle protocol. Default protocol=2 compatible with Python 2.7 and 3.4

    Returns
    -------

    """
    f = open(path, 'wb')
    pickle.dump(dataset, f, protocol=2)
    f.close()
    return

def loadPickle(path):
    """
    Loads pickled SNIDdataset object

    Parameters
    ----------
    path : string

    Returns
    -------

    d : SNIDdataset object

    """
    f = open(path, 'rb')
    d = pickle.load(f)
    return d

def loadDataset(pathdir, snlist):
    """
    Creates a SNIDdataset object from a list of SNID templates.

    Parameters
    ----------
    pathdir : string
        Path to SNID template directory
    snlist : string
        Path to file with list of SNID templates to load.

    Returns
    -------
    dataset : SNIDdataset object.

    """
    dataset = OrderedDict()
    with open(snlist) as f:
        lines = f.readlines()
        f.close()
    for sn in lines:
        print(sn)
        filename = sn.strip()
        snname = sn.strip().split('.')[0]
        snidObj = snid.SNIDsn()
        snidObj.loadSNIDlnw(pathdir+filename)
        dataset[snname] = snidObj
    return dataset

def deleteSN(dataset, phasekey):
    """
    Deletes a SNIDsn object from the SNIDdataset dictionary.

    Parameters
    ----------
    dataset : SNIDdataset object.
    phasekey : string
        name of the SNIDsn object to delete.

    Returns
    -------

    """
    del dataset[phasekey]
    return

def subset(dataset, keys):
    """
    Creates a subset of an existing SNIDdataset object.

    Parameters
    ----------
    dataset : SNIDdataset object.
    keys : iterable
        Subset of keys to keep in the subset SNIDdataset.

    Returns
    -------
    subset : SNIDdataset object

    """
    subset = {key:dataset[key] for key in keys if key in dataset}
    return subset

def datasetTypeDict(dataset):
    """
    Returns a dictionary where the keys are the different SN types present in dataset
    and the values are lists of the names of the SNe of the key type.

    Parameters
    ----------
    dataset : SNIDdataset object

    Returns
    -------
    typeinfo : dict

    """
    typeinfo = dict()
    for sn in list(dataset.keys()):
        sntype = dataset[sn].type
        if sntype in typeinfo:
            typeinfo[sntype].append(sn)
        else:
            typeinfo[sntype] = [sn]
    for key in list(typeinfo.keys()):
        typeinfo[key] = np.array(typeinfo[key])
    return typeinfo

def datasetPhaseDict(dataset):
    """
    Returns a dictionary where the keys are the SN names in dataset
    and the values are a list of that SN's phases present in dataset.

    Parameters
    ----------
    dataset : SNIDdataset object

    Returns
    -------
    phaseinfo : dict

    """
    phaseinfo = dict()
    for snname in list(dataset.keys()):
        snobj = dataset[snname]
        phases = snobj.phases
        phaseinfo[snname]=phases
    return phaseinfo

def numSpec(dataset):
    """
    Returns the total number of spectra present in dataset.

    Parameters
    ----------
    dataset : SNIDdataset object

    Returns
    -------
    numSpec : int

    """
    numSpec = 0
    for snname in list(dataset.keys()):
        snobj = dataset[snname]
        numSpec = numSpec + len(snobj.getSNCols())
    return numSpec

def preprocess(dataset):
    """
    Applies SNIDsn preprocessing to every SN in dataset.

    Parameters
    ----------
    dataset : SNIDdataset object

    Returns
    -------

    """
    for snname in list(dataset.keys()):
        snobj = dataset[snname]
        colnames = snobj.getSNCols()
        for col in colnames:
            snobj.preprocess(col)
    return

def snidsetNAN(dataset):
    """
    Replaces all 0.0 values in all spectra in dataset with NaN.

    Parameters
    ----------
    dataset : SNIDdataset object

    Returns
    -------

    """
    for snname in list(dataset.keys()):
        snobj = dataset[snname]
        snobj.snidNAN()
    return

def interpGaps(dataset, minwvl, maxwvl, maxgapsize):
    """
    For each SNIDsn object in the dataset, this method removes phases where
    the spectrum has large gaps in the wavelength range of interest. All
    remaining spectra are linearly interpolated in the wavelength region
    of interest to remove NaN gaps.

    Parameters
    ----------
    dataset : SNIDdataset object
    minwvl : float
        minimum wavelength
    maxwvl : float
        maximum wavelength
    maxgapsize : float
        maximum gap size tolerable for interpolation (angstroms)

    Returns
    -------

    """
    for snname in list(dataset.keys()):
        snobj = dataset[snname]
        phases = snobj.phases
        colnames = snobj.getSNCols()
        for ph, col in zip(phases, colnames):
            gaps = snobj.findGaps(col)
            largeGapInRange = snid.largeGapsInRange(gaps, minwvl, maxwvl, maxgapsize)
            if largeGapInRange:
                snobj.removeSpecCol(col)
            else:
                wvlmsk = np.logical_and(snobj.wavelengths > minwvl, snobj.wavelengths < maxwvl)
                wvl = snobj.wavelengths[wvlmsk]
                wvlStart = wvl[0]
                wvlEnd = wvl[-1]
                interpWvlStart, interpWvlEnd = snobj.getInterpRange(wvlStart, wvlEnd, col)
                snobj.interp1dSpec(col, interpWvlStart, interpWvlEnd)
    return

def datasetWavelengthRange(dataset, minwvl, maxwvl):
    """
    For each SNIDsn object in the dataset, filters all spectra to the specified wvl range.

    Parameters
    ----------
    dataset : SNIDdataset object
    minwvl : float
        minimum wavelength
    maxwvl : float
        maximum wavelength

    Returns
    -------

    """
    for snname in list(dataset.keys()):
        snobj = dataset[snname]
        snobj.wavelengthFilter(minwvl, maxwvl)
    return

def smoothSpectra(dataset, velcut, velcutIcBL, plot=False):
    """
    For all SNIDsn objects in dataset, applies SNIDsn smoothing of all spectra.

    Parameters
    ----------
    dataset : SNIDdataset object
    velcut : float
        velocity cut for SN features of non broad line type spectra.
    velcutIcBL : float
        velocity cut for SN features of broad line Ic spectra.
    plot : Boolean
        Plots smoothed spectra if True.

    Returns
    -------

    """
    typedict = datasetTypeDict(dataset)
    nonBL = np.concatenate((typedict['IIb'], typedict['Ib'], typedict['Ic']))
    BL = typedict['IcBL']
    for snname in nonBL:
        snobj = dataset[snname]
        colnames = snobj.getSNCols()
        for col in colnames:
            snobj.smoothSpectrum(col, velcut, plot=plot)
    for snname in BL:
        snobj = dataset[snname]
        colnames = snobj.getSNCols()
        for col in colnames:
            snobj.smoothSpectrum(col, velcutIcBL, plot=plot)
    return

def plotDataset(dataset, figsize):
    """
    Plots all spectra in the dataset.

    Parameters
    ----------
    dataset : SNIDdataset object
    figsize : tuple
        matplotlib figure size.

    Returns
    -------

    """
    fig = plt.figure(figsize=figsize)
    count = 0
    for snname in list(dataset.keys()):
        snobj = dataset[snname]
        colnames = snobj.getSNCols()
        for col in colnames:
            plt.plot(snobj.wavelengths, snobj.data[col] + count)
            count = count + 1
    return fig

def choosePhaseType(dataset, phtype):
    """
    Filters out all SNIDsn objects from dataset that are
    not of the desired phase type from SNID templates.

    Parameters
    ----------
    dataset : SNIDdataset object
    phtype : int
        Phase type from SNID template

    Returns
    -------

    """
    for key in list(dataset.keys()):
        sn_obj = dataset[key]
        if sn_obj.phaseType != phtype:
            deleteSN(dataset, key)
    return

def removeSubType(dataset, subtypename):
    """
    Removes all SNIDsn objects of the specified subtype from dataset.

    Parameters
    ----------
    dataset : SNIDdataset
    subtypename : string
        Name of subtype to remove

    Returns
    -------

    """
    for key in list(dataset.keys()):
        sn_obj = dataset[key]
        if sn_obj.subtype == subtypename:
            deleteSN(dataset, key)
    return

def filterPhases(dataset, phaseRangeList, uniquePhaseFlag):
    """
    User can specify a list of phase ranges of the form [(minPhase1, maxPhase1), (minPhase2, maxPhase2), ...]
    and the dataset is filtered so that each SNIDsn object only has spectra observed at phases that
    can be found in one of the specified phase ranges. If uniquePhaseFlag is True, then only one phase
    for each SNIDsn object is chosen for each phase range. The phase that is chosen is the observed phase
    closest to the center of the phase range. If uniquePhaseFlag is False, then all phases that satisfy
    each phase range are included.

    Parameters
    ----------
    dataset : SNIDdataset object
    phaseRangeList : list
        list of (minPhase, maxPhase) tuples
    uniquePhaseFlag : Boolean
        only keeps phase closest to center of (minPhase, maxPhase) tuple
        if True. Otherwise keeps all phases in the phase range.

    Returns
    -------

    """
    for snname in list(dataset.keys()):
        snobj = dataset[snname]
        phases = snobj.phases
        phasekeys = snobj.getSNCols()
        savePhasekeys = []
        for phaseRange in phaseRangeList:
            minPh = phaseRange[0]
            maxPh = phaseRange[1]
            if uniquePhaseFlag:
                # only keep one phase per range.
                centerPh = (minPh + maxPh)/2.0
                closestInd = np.argmin(np.abs(phases - centerPh))
                phasekey = phasekeys[closestInd]
                savePhasekeys.append(phasekey)
            else:
                # keep all phases in each range.
                phmsk = np.logical_and(phases > minPh, phases < maxPh)
                for phk in np.array(phasekeys)[phmsk]:
                    savePhasekeys.append(phk)
        savePhasekeys = np.array(savePhasekeys)
        savePhasekeys = np.unique(savePhasekeys)
        for phk in phasekeys:
            if phk not in savePhasekeys:
                snobj.removeSpecCol(phk)
        if len(snobj.phases) == 0:
            deleteSN(dataset, snname)
    return


def getDiagnostics(dataset):
    """
    Returns some useful diagnostics of a SNIDdataset.

    Parameters
    ----------
    dataset : SNIDdataset

    Returns
    -------
    snnames : list
        names of SNe in dataset
    snphases : list
        phases of each SN present in dataset
    snid_type_pair : list
        SNID type and subtype ints
    snid_type_str : list
        SNID type and subtype strings
    snphasetype : list
        SNID phase types of SNe in dataset

    """
    snnames = []
    snphases = []
    snid_type_pair = []
    snid_type_str = []
    snphasetype = []

    for key in list(dataset.keys()):
        snobj = dataset[key]
        name = snobj.header['SN']
        phase = snobj.phases
        type_pair = (int(snobj.header['TypeInt']), int(snobj.header['SubTypeInt']))
        type_str = snobj.type + snobj.subtype
        phtype = snobj.phaseType

        snnames.append(name)
        snphases.append(phase)
        snid_type_pair.append(type_pair)
        snid_type_str.append(type_str)
        snphasetype.append(phtype)
 
    return snnames, snphases, snid_type_pair, snid_type_str, snphasetype
     
        

