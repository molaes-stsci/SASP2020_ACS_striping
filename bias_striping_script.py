#Time Series Analysis of ACS/WFC Bias STRIPING
import os
import os.path
import shutil
import glob
import numpy as np
from astropy.io import fits
from astroquery.mast import Observations
from astropy.io import fits
from astropy.table import Table
from stwcs import updatewcs
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm
from astropy.time import Time
from astropy import units as u
from scipy.interpolate import UnivariateSpline
import pylab as pl
import time
from progress.bar import IncrementalBar

################

def header_info(fitsfiles, switch_on):
    '''
    setting values to "PERFORM" from "OMIT"
    '''
    for f in fitsfiles:
        for kwd in switch_on:
            fits.setval(f, kwd, value='PERFORM')
        command_line_input = F'crds bestrefs --files {f} --sync-references=1 --update-bestrefs'
        os.system(command_line_input)
    return

def get_striping_noise(fname):
    with fits.open(fname) as hdulist:
        long_string = hdulist[0].header['BIASFILE']
        wfc2_rawbias = hdulist[1].data
        wfc1_rawbias = hdulist[1].data
    fbias = long_string.split('$')[-1]
    superbias_path = os.environ['jref'] + fbias
    with fits.open(superbias_path) as hdulis:
        wfc2_superbias = hdulis[1].data
        wfc1_superbias = hdulis[4].data
    # Subtract superbias from raw bias
    wfc1_strip = wfc1_rawbias - wfc1_superbias
    wfc2_strip = wfc2_rawbias - wfc2_superbias
    # Approximating stripe intensity as the mean of the row
    intensities = [np.mean(row) for row in np.vstack((wfc1_strip, wfc2_strip))]
    # Sigma clipping intensities to remove extreme values
    clipped = sigma_clip(intensities)
    # Taking the median of the clipped values as the global median.
    # Subtracting that median from the stripe intensities to get the
    # bias striping noise in intensities

    return clipped - np.ma.median(clipped)

    ################
if __name__ == '__main__':

    fitsfiles = glob.glob('*raw.fits')

    #hdulist = [fits.open(f) for f in fitsfiles]
    switch_on = ['BLEVCORR', 'BIASCORR']

    #use CRDS to access calibration data system, call in terminal in astroconda environment
    #setting environment variables and access to CRDS
    os.environ['CRDS_SERVER_URL'] = 'https://hst-crds.stsci.edu'
    os.environ['CRDS_SERVER'] = 'https://hst-crds.stsci.edu'
    os.environ['CRDS_PATH'] = './crds_cache'
    os.environ['jref'] = './crds_cache/references/hst/acs/'

    header_info(fitsfiles, switch_on)

    # Initialize lists for skipped files
    skip_indexerror = []
    skip_othererror = []

    # Initialize list to collect stripe noise arrays
    stripenoise_arrays = []

    # Consolidate stripe noise arrays into one array
    if os.path.isfile('stripenoise_20090706.txt'):
        print("Stripe Noise Calculated")
    else:
        result = open("stripenoise_20090706.txt","w+")
        for f in fitsfiles:
            # Try to run our function on the fitsfile.
            try:
                stripenoise_arr = get_striping_noise(f)
            # If an IndexError is raised, do this.
            except IndexError:
                # Add this file to our skipped array. Continue to next iteration in loop
                skip_indexerror.append(f)
                continue
            # For all other errors, do this.
            except:
                skip_othererror.append(f)
                continue
            else:
                stripenoise_arrays.append(stripenoise_arr)
        all_data_stripenoise = np.ma.hstack(stripenoise_arrays)
        print(len(all_data_stripenoise))
        all_data_stripenoise_c = all_data_stripenoise.compressed()
        print(len(all_data_stripenoise_c))
        np.savetxt(result, all_data_stripenoise_c, fmt="%s")
        result.close()

    # Print error results
    print(F'# of IndexErrors {len(skip_indexerror)}')
    print(F'# of Other Errors {len(skip_othererror)}')
    np.savetxt('valueerror_ffiles.txt', X=skip_indexerror, fmt='%s')

    ################
    from histogauss import histogauss, gaussian_fitter
    histogauss(all_data_stripenoise_c)
