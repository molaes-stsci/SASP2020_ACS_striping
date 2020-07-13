#!/usr/bin/env python
# coding: utf-8

# # TIME SERIES ANALYSIS OF ACS/WFC BIAS STRIPING
# "Post-SM4 ACS/WFC Bias Striping : Characterization and Mitigation" (Grogin et al. 2011)

# In[ ]:


import os
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


# # DOWNLOAD IMAGES
# **astroquery.mast.query_criteria** [[documentation](https://astroquery.readthedocs.io/en/latest/mast/mast.html)]
#
# PARAMETERS :
# - instrument_name = 'ACS/WFC'
# - intentType = 'calibration' (or 'science')
# - target_name = 'BIAS'
# - calib_level = 1 (completely raw)
#
# **numpy.argwhere** [[documentation](https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html)]
#
# PARAMETERS :
# - a (array_like) = obs_table
# RETURNS :
# - ndarray (index_array) = *finds specified images in obs_table**
#
# **astroquery.mast.download_products** [[documentation](https://astroquery.readthedocs.io/en/latest/api/astroquery.mast.ObservationsClass.html#astroquery.mast.ObservationsClass.download_products)]
#
# PARAMETERS :
# - products
# - productSubGroupDescription = ['RAW']
# - mrp_only = False (*default False*)

# In[ ]:


obs_table = Observations.query_criteria(instrument_name='ACS/WFC', intentType='calibration', target_name='BIAS', calib_level=1)

#below are images listed in Table 01 - 07 from Grogin et al. 2011
#power-spectrum analysis [Table 05]
idx_ps = ['ja8wa2e8q', 'ja8wa3fpq', 'ja8wa6jmq', 'ja8wa7jqq', 'ja8waaloq', 'ja8wablsq', 'ja8waedtq', 'ja8wafe2q', 'ja8wain1q', 'ja8wajnmq', 'ja8wambyq', 'ja8wanc2q', 'ja8wb0n3q', 'ja8wb1ndq', 'ja8wb4izq', 'ja8wb5j6q']
#striping mitigation testing [Table 06]
idx_sm = ['ja8wc2gjq', 'ja8wc3gvq', 'ja8wc6xlq', 'ja8wc7yaq', 'ja8wcae8q', 'ja8wcbfbq', 'ja8wcetxq', 'ja8wcfugq', 'ja8wciipq', 'ja8wcjjdq', 'ja8wcmxiq', 'ja8wcnxmq', 'ja8wd0t3q', 'ja8wd1tnq', 'ja8wd4hhq', 'ja8wd5i8q', 'ja8wd8w9q', 'ja8wd9wgq', 'ja8wdcmgq', 'ja8wddmzq', 'ja8wdgxfq', 'ja8wdhbrq', 'ja8wdkjiq', 'ja8wdljsq', 'ja8wdockq', 'ja8wdpdlq', 'jbana0eqq', 'jbana1fsq', 'jbana2g2q', 'jbana3gtq']
#calibrated super biases [Table 07]
idx_sb = ['u7q18492j_bia', 'u7q18492j_bia']
#ngc 6217 images used as de-striping demonstration [Table 8]
idx_dd = ['ja7z03ujq', 'ja7za3uzq', 'ja7z03ulq', 'ja7za3v1q']

dl_table = Observations.download_products(oids,
                                          productSubGroupDescription=['RAW'],
                                          mrp_only=False)
#download one image
#idx = np.argwhere(ipst[0]==obs_table['obs_id']).flatten()[0]
#dl_table = Observations.download_products(obs_table[idx]['obsid'],productSubGroupDescription=['RAW'], mrp_only=False)

#download multiple images in a loop
idx01 = [np.argwhere(i==obs_table['obs_id']).flatten()[0] for i in idx_ps]
dl_table01 = Observations.download_products(obs_table[idx01]['obsid'], productSubGroupDescription=['RAW'], mrp_only=False)

print("idx_ps has finished downloading.")

idx02 = [np.argwhere(i==obs_table['obs_id']).flatten()[0] for i in idx_sm]
dl_table02 = Observations.download_products(obs_table[idx02]['obsid'], productSubGroupDescription=['RAW'], mrp_only=False)

print("idx_sm has finished downloading.")


# # Adjustments

# In[ ]:


#Delete mastDownload directory and all subdirectories it contains

for row in dl_table01:
    oldfname = row['Local Path']
    newfname = os.path.basename(oldfname)
    os.rename(oldfname, newfname)
shutil.rmtree('mastDownload')

for row in dl_table02:
    oldfname = row['Local Path']
    newfname = os.path.basename(oldfname)
    os.rename(oldfname, newfname)
shutil.rmtree('mastDownload')


# # CRDS

# In[ ]:


fitsfiles = glob.glob('*.fits')

hdulist = [fits.open(ff) for ff in fitsfiles]

switch_on = ['BLEVCORR', 'BIASCORR']

#use CRDS to access calibration data system, call in terminal in astroconda environment
#setting environment variables and access to CRDS
os.environ['CRDS_SERVER_URL'] = 'https://hst-crds.stsci.edu'
os.environ['CRDS_SERVER'] = 'https://hst-crds.stsci.edu'
os.environ['CRDS_PATH'] = './crds_cache'
os.environ['jref'] = './crds_cache/references/hst/acs/'

def header_info(fits_file):
    '''
    setting values to "PERFORM" from "OMIT"
    '''
    with fits.open(fits_file) as hdulist:
        wfc2_data = hdulist[1].data
        wfc1_data = hdulist[4].data
        for kwd in switch_on:
            fits.setval(f, kwd, value='PERFORM')
    command_line_input = F'crds bestrefs --files {fits_file} --sync-references=1 --update-bestrefs'
    os.system(command_line_input)
    return

#confirm switches are completed
hdulist = fits.open(f)
hdulist[0].header


# In[ ]:


SM4_end = Time('2009-05-25T00:00:00', format='isot')
SM4_plus1year = SM4_end + (1*u.year)
mjd_i = SM4_end.mjd.astype(float)
mjd_f = SM4_plus1year.mjd.astype(float)
is_1yrpostSM4 = [i for i,row in enumerate(obs_table)
                 if row['t_min'] > mjd_i
                 and row['t_min'] < mjd_f]
oids = [row['obsid'] for row in obs_table[is_1yrpostSM4]
        if row['obs_title']=='CCD Daily Monitor {Part 1}']


# # ANALYSIS

# In[ ]:


def get_striping_noise(fname):
    with fits.open(fname) as hdulist:
        print(fname)
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


# In[ ]:


_ = [header_info(f) for f in fitsfiles]
print("finished with super biases")


# In[ ]:


# Initialize lists for skipped files
skip_indexerror = []
skip_othererror = []

# Initialize list to collect stripe noise arrays
stripenoise_arrays = []

for f in fitsfiles:
    bar = 0
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


# Consolidate stripe noise arrays into one array
all_data_stripenoise = np.ma.hstack(stripenoise_arrays)

# Print error results
print(F'# of IndexErrors {len(skip_indexerror)}')
print(F'# of Other Errors {len(skip_othererror)}')

np.savetxt('valueerror_ffiles.txt', X=skip_indexerror, fmt='%s')


# In[ ]:


#running on masked arrays, every time it runs the function, it appends the output to one single array
all_data_stripnoise = np.ma.hstack([get_striping_noise(f) for f in fitsfiles])


# # Plotting

# In[ ]:


#n, bins, _ = plt.hist(all_data_stripenoise, bins=50, density=False)
with fits.open(f) as hdulist:
    long_string = hdulist[0].header['BIASFILE']
    wfc2_rawbias = hdulist[1].data
    wfc1_rawbias = hdulist[1].data

fbias = long_string.split('$')[-1]
superbias_path = os.environ['jref'] + fbias

with fits.open(superbias_path) as hdulis:
    wfc2_superbias = hdulis[1].data
    wfc1_superbias = hdulis[4].data

# Only doing analysis for WFC1 in this example.
# These lines were a "quick and dirty" way to get a quick first plot,
# and we should be changing them according to the method in the paper
# for "characterizing bias striping"
# Subtract superbias from raw bias
wfc1_strip = wfc1_rawbias - wfc1_superbias

# Approximating stripe intensity as the mean of the row
intensities = [np.mean(row) for row in wfc1_strip]

# Sigma clipping intensities to remove extreme values
clipped = sigma_clip(intensities)

# Taking the median of the clipped values as the global median.
# Subtracting that median from the stripe intensities to get the
# bias striping noise in intensities
cleaned = clipped - np.median(clipped)

num_bins = np.arange(400, 700, 100)
print(type(num_bins))

for n in num_bins:
    plt.figure(1)
    result = plt.hist(all_data_stripenoise, bins=n, density=False)
    plt.xlim(np.ma.min(all_data_stripenoise), np.ma.max(all_data_stripenoise))
    mean = np.ma.mean(all_data_stripenoise)
    variance = np.ma.var(all_data_stripenoise)
    sigma = np.ma.sqrt(variance)
    x = np.linspace(np.ma.min(all_data_stripenoise), np.ma.max(all_data_stripenoise), 2068)
    dx = result[1][1] - result[1][0]
    scale = np.ma.shape(all_data_stripenoise)[0] * dx
    plt.plot(x, norm.pdf(x, mean, sigma) * scale)
    plt.xlim(-3.0, 3.0)
    plt.title("Histogram of Stripe Values From 277 Bias Frames")
    plt.xlabel('Stripe Intensity')
    plt.ylabel('Count')
    plt.show()


# # FWHM

# In[ ]:


def fwhm(x,y):
    half_max = amax(y)/2.0
    spline = UnivariateSpline(x, y - (half_max/2), s=0)
    roots = spline.roots()
    return abs(roots[1] - roots[0])

import pylab as pl
pl.plot(x, y)
pl.axvspan(roots[0], roots[1], facecolor='g', alpha=0.5)
pl.show()


# In[ ]:


xmin = np.ma.min(all_data_stripenoise)
xmax = np.ma.max(all_data_stripenoise)


# - print standard deviation of full-fit gaussian to plot
# - write code to do partial-fit of gaussian
# - print standard deviation of partial-fit gaussian to plot

# In[ ]:
