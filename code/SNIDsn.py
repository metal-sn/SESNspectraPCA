from __future__ import division
import numpy as np
import pickle
import scipy
import scipy.stats as st
import scipy.optimize as opt
from scipy import interpolate
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_color_codes('colorblind')



def getType(tp, subtp):
    """
Convert tuple type designation from SNID to string.
    """
    if tp == 1:
        sntype = 'Ia'
        if subtp == 2: 
            snsubtype = 'norm'
        elif subtp == 3: 
            snsubtype = '91T'    
        elif subtp == 4: 
            snsubtype = '91bg'
        elif subtp == 5: 
            snsubtype = 'csm'
        elif subtp == 6: 
            snsubtype = 'pec'
        elif subtp == 7: 
            snsubtype = '99aa'
        elif subtp == 8: 
            snsubtype = '02cx'
        else: 
            snsubtype = ''
    elif tp == 2:
        sntype = 'Ib'
        if subtp == 2: 
            snsubtype = 'norm'
        elif subtp == 3: 
            snsubtype = 'pec'
        elif subtp == 4:
            sntype = 'IIb'
            snsubtype = ''
        elif subtp == 5: 
            snsubtype = 'Ibn'
        elif subtp == 6:
            snsubtype = 'Ca'
        else: 
            snsubtype = ''
    elif tp == 3:
        sntype = 'Ic'
        if subtp == 2: 
            snsubtype = 'norm'
        elif subtp == 3: 
            snsubtype = 'pec'
        elif subtp == 4:
            sntype = 'IcBL'
            snsubtype = ''
        elif subtp == 5:
            snsubtype = 'SL'
        else: 
            snsubtype = ''
    elif tp == 4:
        sntype = 'II'
        if subtp == 2: 
            snsubtype = 'P'
        elif subtp == 3: 
            snsubtype = 'pec'
        elif subtp == 4: 
            snsubtype = 'n'
        elif subtp == 5: 
            snsubtype = 'L'
        else: 
            snsubtype = ''
    elif tp == 5:
        snsubtype = ''
        if subtp == 1: 
            sntype = 'NotSN'
        elif subtp == 2: 
            sntype = 'AGN'
        elif subtp == 3: 
            sntype = 'Gal'
        elif subtp == 4: 
            sntype = 'LBV'
        elif subtp == 5: 
            sntype = 'M-star'
        elif subtp == 6: 
            sntype = 'QSO'
        elif subtp == 7: 
            sntype = 'C-star'
        else: 
            sntype = ''
    return sntype, snsubtype


def largeGapsInRange(gaps, minwvl, maxwvl, maxgapsize):
    """
Given a list of gaps, min and max wavelengths, and a maximum acceptable gap size,\
returns a boolean which is True if a gap larger than maximum acceptable size \
intersects the specified wavelength range. 
     """
    gapInRange = False
    for gap in gaps:
        gapStart = gap[0]
        gapEnd = gap[1]
        gapsize = gapEnd - gapStart
        if maxgapsize > gapsize:
            continue
        else:
            if gapStart < minwvl and gapEnd > maxwvl: gapInRange = True
            if gapStart > minwvl and gapStart < maxwvl: gapInRange = True
            if gapEnd > minwvl and gapEnd < maxwvl: gapInRange = True
    return gapInRange


# Binspec implemented in python.
def binspec(wvl, flux, wstart, wend, wbin):
    nlam = (wend - wstart) / wbin + 1
    nlam = int(np.ceil(nlam))
    outlam = np.arange(nlam) * wbin + wstart
    answer = np.zeros(nlam)
    interplam = np.unique(np.concatenate((wvl, outlam)))
    interpflux = np.interp(interplam, wvl, flux)

    for i in np.arange(0, nlam - 1):
        cond = np.logical_and(interplam >= outlam[i], interplam <= outlam[i+1])
        w = np.where(cond)
        if len(w) == 2:
            answer[i] = 0.5*(np.sum(interpflux[cond])*wbin)
        else:
            answer[i] = scipy.integrate.simps(interpflux[cond], interplam[cond])

    answer[nlam - 1] = answer[nlam - 2]
    cond = np.logical_or(outlam >= max(wvl), outlam < min(wvl))
    answer[cond] = 0
    return answer/wbin, outlam

#smooth spectrum using SNspecFFTsmooth procedure
def smooth(wvl, flux, cut_vel):
    c_kms = 299792.47 # speed of light in km/s
    vel_toolarge = 100000 # km/s

    wvl_ln = np.log(wvl)
    num = wvl_ln.shape[0]
    binsize = wvl_ln[-1] - wvl_ln[-2]
    f_bin, wln_bin = binspec(wvl_ln, flux, min(wvl_ln), max(wvl_ln), binsize)
    fbin_ft = np.fft.fft(f_bin)#*len(f_bin)
    freq = np.fft.fftfreq(num)
    num_upper = np.max(np.where(1.0/freq[1:] * c_kms * binsize > cut_vel))
    num_lower = np.max(np.where(1.0/freq[1:] * c_kms * binsize > vel_toolarge))
    mag_avg = np.mean(np.abs(fbin_ft[num_lower:num_upper+1]))
    powerlaw = lambda x, amp, exp: amp*x**exp
#    print num_upper
#    print nup
    num_bin = len(f_bin)
    xdat = freq[num_lower:num_upper]
#    print xdat
    ydat = np.abs(fbin_ft[num_lower:num_upper])
#    print ydat
    nonzero_mask = xdat!=0
    slope, intercept, _,_,_ = st.linregress(np.log(xdat[nonzero_mask]), np.log(ydat[nonzero_mask]))
    exp_guess = slope
    amp_guess = np.exp(intercept)
#    print slope, intercept

    #do powerlaw fit
    xdat = freq[num_lower:int(num_bin/2)]
    ydat = np.abs(fbin_ft[num_lower:int(num_bin/2)])
    #exclude data where x=0 because this can cause 1/0 errors if exp < 0
    finite_mask = np.logical_not(xdat==0)
    finite_mask = np.logical_and(finite_mask, np.isfinite(ydat))
    ampfit, expfit = opt.curve_fit(powerlaw, xdat[finite_mask], ydat[finite_mask], p0=[amp_guess, exp_guess])[0]

    #find intersection of average fbin_ft magnitude and powerlaw fit to calculate separation
    #velocity between signal and noise.
    intersect_x = np.power((mag_avg/ampfit), 1.0/expfit)
    sep_vel = 1.0/intersect_x * c_kms * binsize

    #filter out frequencies with velocities higher than sep_vel
    smooth_fbin_ft = np.array([fbin_ft[ind] if np.abs(freq[ind])<np.abs(intersect_x) else 0 \
                               for ind in range(len(freq))])#/len(f_bin)
    #inverse fft on smoothed fluxes
    smooth_fbin_ft_inv = np.real(np.fft.ifft(smooth_fbin_ft))

    #interpolate smoothed fluxes back onto original wavelengths
    w_smoothed = np.exp(wln_bin)
    f_smoothed = np.interp(wvl, w_smoothed, smooth_fbin_ft_inv)

    return w_smoothed, f_smoothed, sep_vel

def knot_meanflux_list(cont_header):
    knot_meanflux_pair_list = []
    for i in range(int(len(cont_header[1:])/2)):
        nknot = cont_header[1:][2*i]
        logfmean = cont_header[1:][2*i + 1]
        pair = (nknot, logfmean)
        knot_meanflux_pair_list.append(pair)
    return knot_meanflux_pair_list

def knot_dict(cont):
    d = dict()
    for row in cont:
        key = row[0]
        xyknot_list = []
        for i in range(int(len(row[1:])/2)):
            xknot = row[1:][2*i]
            yknot = row[1:][2*i + 1]
            pair = (xknot, yknot)
            xyknot_list.append(pair)
        d[int(key)] = xyknot_list
    return d

def snid_wvl_axis():
    nw = 1024
    w0 = 2500
    w1 = 10000
    dwlog = np.log(w1/w0)/nw
    wlog = w0*np.exp(np.arange(nw+1)*dwlog)
    dwbin = np.diff(wlog)
    new_wlog = []
    for i in range(nw):
        el = 0.5*(wlog[i]+wlog[i+1])
        new_wlog.append(el)
    new_wlog = np.array(new_wlog)
    return new_wlog, dwbin, dwlog

def convert_xknot_wvl(xknot, nw, wvl):
    pix = np.arange(nw)+1
    wave = np.interp(xknot,pix,wvl)
    return wave

class SNIDsn:
    def __init__(self):
        self.header = None
        self.continuum = None
        self.phases = None
        self.phaseType = None
        self.wavelengths = None
        self.data = None
        self.type = None
        self.subtype = None

        self.smoothinfo = dict()

        return

    def loadSNIDlnw(self, lnwfile):
        """
Loads a SNID .lnw file into a SNIDsn object. Header information from the .lnw file \
is stored in a header dictionary. Other data is stored in the following fields: continuum, phases,\
phaseType, wavelengths, data, type, subtype.
        """
        with open(lnwfile) as lnw:
            lines = lnw.readlines()
            lnw.close()
        header_line = lines[0].strip()
        header_items = header_line.split()
        header = dict()
        header['Nspec'] = int(header_items[0])
        header['Nbins'] = int(header_items[1])
        header['WvlStart'] = float(header_items[2])
        header['WvlEnd'] = float(header_items[3])
        header['SplineKnots'] = int(header_items[4])
        header['SN'] = header_items[5]
        header['dm15'] = float(header_items[6])
        header['TypeStr'] = header_items[7]
        header['TypeInt'] = int(header_items[8])
        header['SubTypeInt'] = int(header_items[9])
        self.header = header

        tp, subtp = getType(header['TypeInt'], header['SubTypeInt'])
        self.type = tp
        self.subtype = subtp

        phase_line_ind = len(lines) - self.header['Nbins'] - 1
        phase_items = lines[phase_line_ind].strip().split()
        self.phaseType = int(phase_items[0])
        phases = np.array([float(ph) for ph in phase_items[1:]])
        self.phases = phases

        wvl = np.loadtxt(lnwfile, skiprows=phase_line_ind + 1, usecols=0)
        self.wavelengths = wvl
        lnwdtype = []
        colnames = []
        for ph in self.phases:
            colname = 'Ph'+str(ph)
            if colname in colnames:
                colname = colname + 'v1'
            count = 2
            while(colname in colnames):
                colname = colname[0:-2] + 'v'+str(count)
                count = count + 1
            colnames.append(colname)
            dt = (colname, 'f4')
            lnwdtype.append(dt)
        #lnwdtype = [('Ph'+str(ph), 'f4') for ph in self.phases]
        #print lines[phase_line_ind+1]
        data = np.loadtxt(lnwfile, dtype=lnwdtype, skiprows=phase_line_ind + 1, usecols=range(1,len(self.phases) + 1))
        self.data = data

        continuumcols = len(lines[1].strip().split())
        continuum = np.ndarray((phase_line_ind - 1,continuumcols))
        for ind in np.arange(1,phase_line_ind - 0):
            cont_line = lines[ind].strip().split()
            #print cont_line
            continuum[ind - 1] = np.array([float(x) for x in cont_line])
        self.continuum = continuum
        return

    def preprocess(self, phasekey):
        """
Zeros the mean and scales std to 1 for the spectrum indicated.
        """
        specMean = np.mean(self.data[phasekey])
        specStd = np.std(self.data[phasekey])
        self.data[phasekey] = (self.data[phasekey] - specMean)/specStd
        return

    def restoreContinuum(self, verbose=False, spl_a_ind=0, spl_b_ind=-1):
        """
Restores the SNID continuum.
Arguments:
    knotmode -- ('wvl', 'pix') whether to compute knot values in wavelength or pixel units.
        """
        continuum_header = self.continuum[0]
        continuum = self.continuum[1:]
        if verbose: 
            print ("continuum lines")
            print (continuum)
        nknot_mean_list = knot_meanflux_list(continuum_header)
        if verbose: 
            print ("nknot mean list")
            print (nknot_mean_list)
        xy_knot_dict = knot_dict(continuum)
        if verbose: print (xy_knot_dict)

        wvl, dwbin, dwlog = snid_wvl_axis()
        data_unflat = []
        for nspec_ind in range(self.header['Nspec']):
            spline_x = []
            spline_y = []
            spline_deg = 3
            num_splines_spec = int(nknot_mean_list[nspec_ind][0])
            if verbose:
                print ("num splines for this spectrum")
                print (num_splines_spec)
            for spline_ind in np.array(xy_knot_dict.keys())[:num_splines_spec]:
                pair = xy_knot_dict[spline_ind][nspec_ind]
                if verbose: 
                    print ("knot pair")
                    print (pair)

                xknot = pair[0]
                yknot = pair[1]
                xknot = np.power(10,xknot)
                yknot = np.power(10,yknot)*np.power(10,nknot_mean_list[nspec_ind][1])
                spline_x.append(xknot)
                spline_y.append(yknot)
                if verbose:
                    print ("xknot, yknot") 
                    print (xknot, yknot)

            spline_x = np.array(spline_x)
            spline_x_wvl = np.array([convert_xknot_wvl(x,1024,wvl) for x in spline_x])
            spline_y = np.array(spline_y)
            if verbose:
                print ("splines")
                print (spline_x)
                print (spline_x_wvl)
                print (spline_y)
                print (spline_deg)

            msk = np.logical_and(wvl >= spline_x_wvl[spl_a_ind], wvl <= spline_x_wvl[spl_b_ind])
            cubicspline = CubicSpline(spline_x_wvl, np.log10(spline_y))
            y = cubicspline(wvl)
            if verbose:
                print ("spline eval") 
                print (y)
                print (np.power(10,y)[1])

            unflat = []
            for i in range(len(y)):
                phkey = self.data.dtype.names[nspec_ind]
                newf = (self.data[phkey][i]+1)*np.power(10,y[i])
                #newf = (lnw_dat[i,1]+1)*np.power(10,y[i])
                unflat.append(newf)
            if verbose:
                print ("unflat")
                print (unflat[0:10])

            unflat = np.array(unflat)
            zeromsk = np.logical_or(wvl < spline_x_wvl[spl_a_ind], wvl > spline_x_wvl[spl_b_ind])
            unflat[zeromsk] = 0.0
            unflat = unflat/dwbin/np.mean(unflat[msk]/dwbin[msk])
            data_unflat.append(unflat)
        self.data_unflat = np.array(data_unflat).T
        return

        

    def wavelengthFilter(self, wvlmin, wvlmax):
        """
Filters the wavelengths to a user specified range, and adjusts the spectra data\
array to reflect that wavelength range.
        """
        wvlfilter = np.logical_and(self.wavelengths < wvlmax, self.wavelengths > wvlmin)
        wvl = self.wavelengths[wvlfilter]
        self.data = self.data[wvlfilter]
        self.wavelengths = self.wavelengths[wvlfilter]
        return
       
    def removeSpecCol(self, colname):
        """
Removes the column named colname from the structured spectra matrix. Removes the \
phase from the list of phases.
        """
        newdtype = [(dt[0], dt[1]) for dt in self.data.dtype.descr if dt[0] != colname]
        newshape = (len(self.data),len(newdtype))
        ndarr = np.ndarray(newshape)
        for i in range(len(newdtype)):
            ndtype = newdtype[i]
            nm = ndtype[0]
            ndarr[:,i] = self.data[nm]
        newstructarr = np.array([tuple(row.tolist()) for row in ndarr], dtype=newdtype)
        rmInd = np.where(np.array(self.getSNCols()) == colname)[0][0]
        newphases = []
        for i in range(len(self.phases)):
            if i != rmInd:
                newphases.append(self.phases[i])
        self.phases = np.array(newphases)
        self.data = newstructarr
        return


    def snidNAN(self):
        """
SNID uses 0.0 as a placeholder value when observed data is not available for \
a certain wavelength. This method replaces all 0.0 values with NaN.
        """
        colnames = self.getSNCols()
        for col in colnames:
            self.data[col][self.data[col] == 0] = np.nan
        return 

    def getSNCols(self):
        """
Returns spectra structured array column names for the user.
        """
        return self.data.dtype.names


    def findGaps(self, phase):
        """
Returns a list of all the gaps present in the spectrum for the given phase.
        """
        spec = self.data[phase]
        wvl = self.wavelengths
        nanind = np.argwhere(np.isnan(spec)).flatten()
        nanwvl = wvl[np.isnan(spec)]
        gaps = []
        if len(nanind) == 0:
            return gaps
        gapStartInd = nanind[0]
        gapStartWvl = nanwvl[0]
        for i in range(0,len(nanind) - 0):
            ind = nanind[i]
            if ind == nanind[-1]:
                gap = (gapStartWvl, wvl[ind])
                gaps.append(gap)
                break
            nextInd = nanind[i + 1]
            if ind +1 != nextInd:
                gap = (gapStartWvl, wvl[ind])
                gaps.append(gap)
                gapStartInd = nextInd
                gapStartWvl = wvl[nextInd]
        return gaps
    
    def getInterpRange(self, minwvl, maxwvl, phase):
        """
Returns the wavelength range needed for interpolating out gaps. Returned \
range encloses the range specified by min and max wvl. Does not assume \
anything about size of the gaps inside the user specified wvl range. \
Exits with assert error if no finite values exist on one of the sides \
of the user specified wvl range.
        """
        wavelengths = self.wavelengths
        spec = self.data[phase]
        wv = wavelengths[np.logical_and(wavelengths > minwvl, wavelengths < maxwvl)]
        wvStart = wv[0]
        wvEnd = wv[-1]
        wvFinite = wavelengths[np.logical_not(np.isnan(spec))]
        startmsk = wvFinite < wvStart
        endmsk = wvFinite > wvEnd
        assert len(wvFinite[startmsk]) > 0, "no finite wvl values before %f"%(wvStart)
        assert len(wvFinite[endmsk]) > 0, "no finite wvl values after %f"%(wvEnd)
        wvStartFinite = wvFinite[startmsk][np.argmin(np.abs(wvFinite[startmsk] - wvStart))]
        wvEndFinite = wvFinite[endmsk][np.argmin(np.abs(wvFinite[endmsk] - wvEnd))]
        return wvStartFinite, wvEndFinite 





    def interp1dSpec(self, phase, minwvl, maxwvl, plot=False):
        """
Linearly interpolates any gaps in the spectrum specified by phase, in the wvl range (inclusive of endpoints) \
specified by the user. Default behavior is to not check for gaps. It is up to the user to determine \
whether large gaps exist using the module function SNIDsn.largeGapsInRange().
        """
        wvlmsk = np.logical_and(self.wavelengths >= minwvl, self.wavelengths <= maxwvl)
        specRange = self.data[phase][wvlmsk]
        wvlRange = self.wavelengths[wvlmsk]
        rangeFiniteMsk = np.isfinite(specRange)
        interp_fn = interp1d(wvlRange[rangeFiniteMsk], specRange[rangeFiniteMsk])
        interpSpecRange = interp_fn(wvlRange)

        self.data[phase][wvlmsk] = interpSpecRange

        if plot:
            fig = plt.figure(figsize=(15,5))
            plt.plot(self.wavelengths, self.data[phase], 'r')
            plt.plot(wvlRange, specRange)
            return fig
        return


    def smoothSpectrum(self, phase, velcut, plot=False):
        """
Uses the Modjaz et al method to smooth a spectrum.
        """
        spec = np.copy(self.data[phase])
        wsmooth, fsmooth, sepvel = smooth(self.wavelengths, spec, velcut)
        self.smoothinfo[phase] = sepvel
        self.data[phase] = fsmooth

        if plot:
            fig = plt.figure(figsize=(15,5))
            plt.plot(self.wavelengths, spec, 'r', label='pre-smooth')
            plt.plot(self.wavelengths, self.data[phase], 'k', label='smooth')
            plt.legend(title=self.header['SN'])
            return fig
        return




    def save(self, path='./'):
        """
Saves the SNIDsn object using pickle. Filename is automatically generated using \
the sn name stored in the SNIDsn.header['SN'] field.
        """
        filename = self.header['SN']
        f = open(path+filename+'.pickle', 'w')
        pickle.dump(self, f)
        f.close()
        return
