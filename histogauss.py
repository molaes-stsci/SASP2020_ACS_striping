#!/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp_stats
from astropy.stats.biweight import biweight_location,biweight_scale
from astropy.visualization import hist
import mpfit

def histogauss(SAMPLE,domain=[0,0]):

#+
#NAME:
#       HISTOGAUSS
#
# PURPOSE:
#       Histograms data and overlays it with a fitted Gaussian.
#       Uses "MPFIT" to perform the fitting, which can be done on a restricted domain.
#
# CALLING SEQUENCE:
#       model_params = histogauss.histogauss(mydata)
#       model_params = histogauss.histogauss(mydata,[lower,upper])
#
# INPUT:
#       mydata = nparray of data values (near-normally distributed) to be histogrammed
#       domain = optional 2-valued list, specifying domain endpoints for Gaussian fit;
#                default is to fit Gaussian to entire domain of data values
#
# OUTPUT ARGUMENTS:
#       model_param = list of coefficients of the Gaussian fit:
#
#               model_param[0]= the normalization ("height") of the fitted Gaussian
#               model_param[1]= the mean of the fitted Gaussian
#               model_param[2]= the standard deviation of the fitted Gaussian
#               model_param[3]= the half-width of the 95% confidence interval
#                               of the biweight mean (not fitted)
#
# REVISION HISTORY:
#       Written, H. Freudenreich, STX, 12/89
#       More quantities returned in A, 2/94, HF
#       Added NOPLOT keyword and print if Gaussian, 3/94
#       Stopped printing confidence limits on normality 3/31/94 HF
#       Added CHARSIZE keyword, changed annotation format, 8/94 HF
#       Simplified calculation of Gaussian height, 5/95 HF
#       Convert to V5.0, use T_CVF instead of STUDENT_T, GAUSSFIT instead of
#           FITAGAUSS  W. Landsman April 2002
#       Correct call to T_CVF for calculation of A[3], 95% confidence interval
#                P. Broos/W. Landsman   July 2003
#       Allow FONT keyword to be passed.  T. Robishaw Apr. 2006
#       Use Coyote Graphics for plotting W.L. Mar 2011
#       Better formatting of text output W.L. May 2012
#-
#       (Crudely) converted to Python3 by N. Grogin July 2020
#
    DATA = SAMPLE.copy()
    N = len(DATA)

# Make sure that not everything is in the same bin. If most
# data = 0, reject zeroes. If they are all some other value,
#  complain and give up.
    DATA.sort()
    ### boundaries of first and third quartiles
    N3 = int(0.75*N)-1
    N1 = int(0.25*N)-1

    if (DATA[N3] == DATA[N1]):

        if (DATA[int(N/2)-1] == 0):

            if (sum(DATA!=0) > 15):
                print('Suppressing Zeroes')
                DATA = DATA[np.where(DATA!=0)]
                N = len(DATA)

            else:
                print('Too Few Non-0 Values!')
                return 0

        else:
            print('Too Many Identical Values: '+str(DATA[int(N/2)]))
            return 0

    # legacy structure from the IDL version; A[0] was an effective height
    A = np.zeros(4)

    NTOT = len(DATA)
    # Outlier-resistant estimator of sample "mean":
    A[1] = biweight_location(DATA)
    # Outlier-resistant estimator of sample "standard deviation":
    A[2] = biweight_scale(DATA)
    # Compute he 95% confidence half-interval on the above mean:
    M=0.7*(NTOT-1)  #appropriate for a biweighted mean
    CL = 0.95
    two_tail_area = 1 - CL
    A[3]=abs( sp_stats.t.ppf(1 - (two_tail_area)/2.0,M) ) * A[2] / np.sqrt(NTOT)

    ### clear the figure
    plt.clf()

    # Plot the histogram [important to have density=True for uniform normalization
    # Also determines 'optimal' bin sizing, based on Freedman algorithm
    histy,histx,ignored=hist(DATA, bins=200, histtype='stepfilled', density=True)

    # Compute the midpoints of the histogram bins
    xmid = np.zeros(len(histy))
    for i in range(len(histy)):
         xmid[i] = (histx[i]+histx[i+1])/2

    # trim the histogram-fitting region, if a domain is specified
    if (domain==[0,0]):
        fitx = xmid
        fity = histy
    else:
        fitx = xmid[(xmid>=domain[0]) & (xmid<=domain[1])]
        fity = histy[(xmid>=domain[0]) & (xmid<=domain[1])]

    # Array containing the initial guess of Gaussian params: normalization; mean; sigma
    # !!! POOR RESULTS IF INPUT DATA ARE HIGHLY NON-GAUSSIAN !!!
    p0 = [max(histy)*(A[2] * np.sqrt(2 * np.pi)),A[1],A[2]]

    # Uniform weighting of each histogram bin value, in Gaussian fit
    err = np.ones(len(fitx))

    # prepare the dictionary of histogram information, for the fitting
    fa = {'x':fitx, 'y':fity, 'err':err}

    # perform the Gaussian fit to the histogram
    m = mpfit.mpfit(gaussian_fitter, p0, functkw=fa)

    # post-fitting diagnostics
    print('mpfit status = ', m.status)
    if (m.status <= 0):
        print('mpfit error message = ', m.errmsg)
        return 0
    print('mpfit parameters = ', m.params)
    [norm, mu, sigma] = m.params

    ### plot the model-fit Gaussian at finely-spaced locations spanning the data bins
    ### color the model-fitted region green, and the ignored region(s) red
    finex = histx[0] + np.arange(1000) * (histx[-1]-histx[0])/1000
    if (domain==[0,0]):
        plt.plot(finex, norm/(sigma * np.sqrt(2 * np.pi)) *
                 np.exp( - (finex - mu)**2 / (2 * sigma**2) ),
                 linewidth=2, color='g')
    else:
        xsec = finex[(finex<domain[0])]
        if len(xsec) > 0:
            plt.plot(xsec, norm/(sigma * np.sqrt(2 * np.pi)) *
                     np.exp( - (xsec - mu)**2 / (2 * sigma**2) ),
                     linewidth=2, color='r')
        xsec = finex[(finex>=domain[0]) & (finex<=domain[1])]
        plt.plot(xsec, norm/(sigma * np.sqrt(2 * np.pi)) *
                 np.exp( - (xsec - mu)**2 / (2 * sigma**2) ),
                 linewidth=2, color='g')
        xsec = finex[(finex>domain[1])]
        if len(xsec) > 0:
            plt.plot(xsec, norm/(sigma * np.sqrt(2 * np.pi)) *
                     np.exp( - (xsec - mu)**2 / (2 * sigma**2) ),
                     linewidth=2, color='r')

    # NAG's devel environment does not easily allow plot-windowing,
    # so the 'plt.show' is commented out, in favor of file-dumping
    #plt.show()

    ### !!! BEWARE HARDCODED OUTPUT-FILENAME, BELOW !!!
    ### SUGGEST ADDING OUTPUT-FILENAME AS ADDITIONAL PARAMETER PASSED TO HISTOGAUSS
    plt.savefig('temp.png')

    return [norm, mu, sigma, A[3]]

def gaussian_fitter(p, fjac=None, x=None, y=None, err=None):
 # Gaussian fitting-function for MPFIT parameter optimization
 # Model parameter values are passed in "p"
 # If fjac==None, then partial derivatives should not be computed.
 # It will always be None if MPFIT is called with default flag.
 # Values for arrays "x","y","err" are passed externally, via MPFIT param "functkw"
    [norm, mu, sigma] = p
    model = norm/(sigma * np.sqrt(2 * np.pi)) * np.exp( -(x - mu)**2 / (2 * sigma**2) )
 # Non-negative status value means MPFIT should continue, negative means
 # stop the calculation.
    status = 0
    return([status, (y-model)/err])
