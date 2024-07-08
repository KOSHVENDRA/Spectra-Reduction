#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 21:12:39 2024

@author: kushu
"""

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits,ascii
from astropy.visualization import ImageNormalize, LinearStretch, SqrtStretch, LogStretch, ZScaleInterval,MinMaxInterval
from astropy.stats import sigma_clip, biweight_location
from astropy.table import Table

from scipy.ndimage import shift, rotate, percentile_filter
from scipy.signal import find_peaks
from scipy import interpolate

from lmfit.models import GaussianModel, ConstantModel, gaussian

#~~~~~~~~~~~~~~~~~ This code is to do the Spectra reduction from a clean image of spectra  ~~~~~~~~


#take a spectra for example

XZtau_hfosc = "/Users/kushu/Documents/PythonCodes/Spec-Reduction/XZtau_HFOSC.fits"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def construct_spectra_slitprofile(spec_image,dispaxis=1,threshold_peak=0.007,min_peaks=5):
    """
    This function constructs the slit profile from a lamp spectra and generates a plot of the profile too
    
    **The slit profiles on the rotated (scipy.ndimage.rotate) image is slightly different from that of the unrotated spectra**

    Parameters
    ----------
    spec_image : numpy 2D array
        Spectrum image for which the slit profile os generated.
    dispaxis : int, optional
        Dispersion axis. 0 for row and 1 for column. The default is 1.
    threshold_peak : float, optional
        Peaks to be detected above this value. **0 to 1**. The default is 0.0075.
    min_peaks : int, optional
        Minimum number of peaks to construct slit profile. The default is 5.

    Returns
    -------
    slit_profiles : (N,M) array
        an array of size (N,M) constructed for N peaks over M pixels.

    """
    
    img_size = spec_image.shape
    
    #create a spectra from the image
    spectrum = np.sum(spec_image,axis=dispaxis)
    
    # plt.figure()
    # plt.plot(spectrum,'ro',ls='-')
    
    #normalize the spectra
    spectrum_norm = spectrum / max(spectrum)
    
    #Now find peaks in the spectra
    peaks, _ = find_peaks(spectrum_norm,threshold=threshold_peak)
    
    # print(peaks)
    
    #x-axis array along which the slit profile is measured
    xarr = np.arange(0,img_size[dispaxis],1)
    
    #plot the slit profile also
    fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(7,12),height_ratios=(3,2),sharex=True)
    calar = plt.cm.rainbow(np.linspace(0,1,len(peaks)))
    plt.tight_layout(pad=2)
    
    #check if number of found peak is above min_peaks
    if len(peaks) >= min_peaks:
        
        #median slit profiles for each of the peaks
        slit_profiles = []
        
        #loop through each of the files
        for i in range(len(peaks)):
            
            if dispaxis ==1:
                image_cutout = spec_image[peaks[i]-20:peaks[i]+20,0:img_size[1]]
            else:
                image_cutout = spec_image[0:img_size[0],peaks[i]-20:peaks[i]+20]
            
            # now create the slit profiles
            sltprfl = np.argmax(image_cutout,axis=int(1-dispaxis))
            
            #there could be additional data points that are not removed by sigma_clipping and we will do it iteratively
            #fit a straight line and do sigma clipping on residual. Once outliers are removed, we can do 3rd order plynomial fit
            #on the slit profiles
            
            #3 sigma clipping
            sltprfl_clip = sigma_clip(sltprfl,sigma=3)
            
            #fit a straight line
            poly_line = np.polyfit(xarr, y=sltprfl_clip, deg=1)
            bestfit_line = np.poly1d(poly_line)(xarr)
            
            sltprfl_residue = sltprfl_clip - bestfit_line
            
            sltprfl_residue_clip = sigma_clip(sltprfl_residue,sigma=3)
            
            #bring the data back to the orginal state and fit a 3rd degree polynomial
            sltprfl_cliptwice = sltprfl_residue_clip + bestfit_line
            
            poly_3d = np.polyfit(xarr, sltprfl_cliptwice, deg=3)
            fit_slitprfl = np.poly1d(poly_3d)(xarr)
            
            #do median subtraction, so that shift are around 0, some beong negative and otehrs beong positive
            fit_slitprfl_medsub = fit_slitprfl - np.median(fit_slitprfl)
            
            #plot
            ax[0].plot(xarr,sltprfl,color=calar[i],ls='',marker='o',markersize=3,alpha=0.3)
            ax[0].plot(xarr,fit_slitprfl,color=calar[i],ls='-',label='peaks:'+str(peaks[i]))
            
            #save this median profile
            slit_profiles.append(fit_slitprfl_medsub)
            
            ax[1].plot(xarr,fit_slitprfl_medsub,color=calar[i],ls='-',alpha=0.25,lw=3)
            
            
        #plot the median profle
        Slit_profile_median = np.median(slit_profiles,axis=0)
        Slit_profile_biweight_loc = biweight_location(slit_profiles,axis=0)
        
        ax[1].plot(xarr,Slit_profile_median,'k',ls='dotted',lw=2.2,label='median')
        ax[1].plot(xarr,Slit_profile_biweight_loc,'k',ls=(0,(5,10)),lw=3.2,label='Biweight location')
        
        ax[0].legend()
        ax[1].legend()
        
        ax[0].set_title('Slit profiles',fontsize=14)
        ax[1].set_xlabel('Image Axis',fontsize=14)
        
        ax[0].set_ylabel('Slit profile',fontsize=14)
        ax[1].set_ylabel('Shifted slit profiles',fontsize=14)
        
        return slit_profiles
    else:
    
        return print('Less than 5 peaks detected, Decrease the "threashold_peak" to detect atleast 5 peaks')



def correct_slitprofile_spectImage(spec_image,slit_profile,dispaxis=1):
    """
    This function corrects a spectrum image for its slit profile, This is to work on a single order spectrum

    Parameters
    ----------
    spec_image : 2D numpy array
        The single order spectrum image.
    slit_profile : numpy 1D array
        Slit profile, an array of numbers by which each axis 
        along dispersion axis is to be shifted to correct the spectrum image for slit profile.
    dispaxis : int, optional
        Dispersion axis, '0' or '1' for spectrum dispersion along 'row' or 'column' respectively . The default is 1.

    Returns
    -------
    copy_spec_image : 2D numpy array
        Slit profile corrected the image.

    """
    
    
    # make a copy of the actualy image
    copy_spec_image = spec_image.copy()
    
    #get the shape of the spectra
    # spec_img_shape = spec_image.shape
    
    #now we shift the coloumns of the image
    for i in range(len(slit_profile)):
        
        if dispaxis==1:
            copy_spec_image[:,i] = shift(spec_image[:,i],shift=slit_profile[i])
        else:
            copy_spec_image[i,:] = shift(spec_image[i,:],shift=slit_profile[i])
            
            
    return copy_spec_image


#We need to get the spectral line center by fitting gaussians over each of the profiles in lamp spectra
def get_peakcenters(spectra,continuum_percentile=35,peak_threshold=0.00018,show_fig_peak=False):
    """
    This function re-estimates the spectral line centers by fitting gaussian functions over each of the lines
    **The spectra should be slit profile corrected and normalized by maximum flux**

    Parameters
    ----------
    spectra : numpy.array
        Slit profile corrected and normalized by maxima of the flux for the lamps.
    continuum_percentile : float, optional
        The percentile for continuum extraction of the spectra. The default is 35.
    peak_threshold : float
        The signal will be detected above 'peak_threshold*local_median'
    show_fig_peak : bool, optional
        Whether to plot the spectra with identified line marked or not. The default is False.

    Returns
    -------
    Table_peaks : astropy.table
        Table with all the information about the spectral lines, identified and fitted with gaussian.

    """
    
    pxl_arr = np.arange(0,len(spectra),1)
    
    #define an lmfit composit model, of gaussian+constant, that can be used to fit the spectral lines
    Gauss_model = GaussianModel() + ConstantModel()
    #all fitting parameters
    Fitparam_names = ['amplitude','sigma','center','c']
    #we will define the parameters separately for each of the peaks
    
    #let's create a continuum of the spectrum, then take 3*std of the continuum as peak lower threshold
    conti_spec = percentile_filter(spectra, percentile=continuum_percentile,size=300)
    med_spec, std_spec = np.median(conti_spec),np.std(conti_spec)
    
    #detection threshold
    height_threshold = conti_spec + peak_threshold*std_spec
    
    # peaks_spec, _ = find_peaks(spectra,threshold=height_threshold)
    peaks_spec, _ = find_peaks(spectra - conti_spec,threshold=peak_threshold)

    #get rough eastimate of the peak line height
    height_peak_spec = np.array([np.max(spectra[peak-3:peak+3]) for peak in peaks_spec])
    
    if show_fig_peak is not False:
        
        fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(17,5))
        
        ax.plot(spectra,'ko',markersize=3,ls='')
        ax.plot(spectra,'b',ls='-',lw=2.2)
        ax.vlines(peaks_spec,ymin=height_peak_spec+0.05,ymax=height_peak_spec+0.15,color='r')      #mark the identified lines in the plot
        ax.set_xlabel('Image pixel',fontsize=14)
        ax.set_ylabel('Normalize spectra',fontsize=14)
        ax.set_title('Detected peaks in the spectra',fontsize=14)
        plt.tick_params(axis='both',direction='in')
        plt.tight_layout(h_pad=0,w_pad=0)
        
    #we need to fit gaussian around each of the spectral line peaks
    
    #initiate a dictionary to save all the parameters
    saveFitparam_dict = {}      
    for fpn in Fitparam_names:
        saveFitparam_dict[str(fpn)] = []
        saveFitparam_dict[str(fpn)+'_err'] = []
    
    redchisqr_fit = []
    
    for p in range(len(peaks_spec)):
        
        # initiate parameters of this model
        Gauss_model.set_param_hint(name='amplitude', value=height_peak_spec[p],min=0,max=10)
        Gauss_model.set_param_hint(name='sigma', value=3,min=0.1,max=10)
        Gauss_model.set_param_hint(name='center', value=0,min=-2,max=+2)
        Gauss_model.set_param_hint(name='c', value=0,min=-2,max=2)
        
        Gauss_model_param = Gauss_model.make_params()
        
        #get the data to fit
        line_peak_pos = peaks_spec[p]               #line peak positions
        spec_cutout = spectra[line_peak_pos-8:line_peak_pos+8]
        pxl_arr_cutout = pxl_arr[line_peak_pos-8:line_peak_pos+8]
        pxl_arr_cutout -= line_peak_pos                  #shift the x-axis, pixel array
        
        #fit the data
        Fit_line = Gauss_model.fit(spec_cutout,params=Gauss_model_param,x=pxl_arr_cutout)
        
        #save the fitted params
        for fpnn in Fitparam_names:
            saveFitparam_dict[str(fpnn)].append(Fit_line.best_values[str(fpnn)])
        redchisqr_fit.append(Fit_line.redchi)
        
        
    #now convert this dictionary into an astropy Table
    Table_peaks = Table()
    Table_peaks['Line center init'] = np.array(peaks_spec)
    for fpnn in Fitparam_names:
        Table_peaks[str(fpnn)] = np.array(saveFitparam_dict[str(fpnn)])
    Table_peaks['redchi'] = np.array(redchisqr_fit)
    
    #the fitted line center and add to the table
    line_center_fit = np.array(saveFitparam_dict['center']) + np.array(peaks_spec)
    Table_peaks['Line center fit'] = np.array(line_center_fit)
    
    if show_fig_peak is not False:
        ax.vlines(line_center_fit,ymin=height_peak_spec+0.05,ymax=height_peak_spec+0.15,color='g')
    
    #add some comment to the table
    Table_peaks.meta['comments'] = 'For fitting, pixel array was shifted to center at 0 \n So, fitted line center in column #2 is "Line center init" + "center" \n\n'.splitlines()
    
    if show_fig_peak is not False:
        return Table_peaks, fig
    else:
        return Table_peaks

# a function to return values from a table those are closest to the values in the inpu table
def getvalues_fromFile(file_arr, yourarr):
    
    file_arr.astype(float)
    
    fine_values = []
    
    for num in yourarr:
        
        indx = np.argmin(np.abs(file_arr - num))
        fine_values.append(file_arr[indx])

    return np.array(fine_values)


# function to plot the spectral image
def read_and_plot_specImage(spec_data, title='Figure'):
    
    
    #read the file
    # spec_data = fits.open(spec_file)[0].data
    
    #Now plot the spectra image
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(2,10))
    
    #The spectra  a 2D numpy array, so we need imshow
    ax.imshow(spec_data,cmap='gray',norm =ImageNormalize(spec_data,interval=ZScaleInterval(),stretch=LinearStretch()))
    ax.set_title(str(title),fontsize=13,family='sans-serif')
    
    plt.tight_layout()
    
    return fig


#Now we start with the spectrum

#First Job os to find where does this spectra lies in direction perpendicular to dispersion axis of the spectra
def get_aperture_location(spec_data,dispersion_axis=0):
    
    #First get at what row/column does image maximizes when seen along column/row 
    max_dispers_loc = np.argmax(spec_data,axis=dispersion_axis)
    
    #look up at the max_dispers_loc and remove the deviated data points
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(9,4))
    ax.plot(max_dispers_loc,color='b',marker='x',markersize=6,linestyle='',alpha=0.3)
    ax.set_title('Argmax along the dispersion axis',fontsize=13,fontfamily='Monospace')
    ax.grid(True)
    
    return max_dispers_loc, fig

def remove_outliers(spec_data,disp_axis_min, disp_axis_max):
    
    outlier = None
    
    
    return None







#~~~~~~~~~~~~~~~~ See the plotting function
#read the spectral data
# spec_data =  fits.open(XZtau_hfosc)[0].data
# disp_mean_spec = get_aperture_location(spec_data,1)
# print(np.shape(spec_data))

# fig = read_and_plot_specImage(spec_data,title='HFOSC- XZ-TAU')
# plt.figure(figsize=(9,4))
# plt.plot(disp_mean_spec,color='b',marker='x',markersize=6,linestyle='',alpha=0.3)
# plt.title('Argmax along the dispersion axis',fontsize=13,fontfamily='Monospace')
# plt.ylim(np.median(disp_mean_spec)-40,max(disp_mean_spec)+40)



#~~~~~~~~~~~~~~~ This function has everything that calibrates pixels against wavelength for FeNe Gr8 
# address = "/Users/kushu/Downloads/checkTable.dat"
# table_peak = ascii.read(address,format='fixed_width')
# peak_init = np.array(table_peak['Line center init'])
# peak_fit  = np.array(table_peak['Line center fit'])





# # array of coordinates and array of wavelengths
# pixl_peak =  [358,556,639,694,1292,1330,1480,1535,1647,1734,1900,1931,1997,2073,2160,2176,2218,2243,2275,2317,2363,2380,2421,2440,2478,2552,2634]
# pixl_gauss = [358,556,639,694,1292,1330,1480,1535,1647,1734,1900,1931,1997,2073,2160,2176,2218,2243,2275,2317,2363,2380,2421,2440,2478,2552,2634]

# wave_peak =  [8780.622,8495.36,8377.367,8300.326,7488.872,7438.899,7245.167,7173.939,7032.413,6929.488,6717.043,6678.276,
#               6598.95,6506.57,6402.246,6382.991,6334.428,6304.79,6266.445,6217.2,6163.19,6143.062,6096.16,6074.33,6029.993,5944.830,5852.489]


# wave_gauss = [8780.622,8495.36,8377.367,8300.326,7488.872,7438.899,7245.167,7173.939,7032.413,6929.488,6717.043,6678.276,
#               6598.95,6506.57,6402.246,6382.991,6334.428,6304.79,6266.445,6217.2,6163.19,6143.062,6096.16,6074.33,6029.993,5944.830,5852.489]
# #write a function to retrive values exactly from the orignal file

# # @keep wavelengths in the increasing order and so all others
# #same pixels and waveengths are identified
# pixels_revers = pixl_gauss[::-1]
# waveln_revers = wave_gauss[::-1]




# #read feNe file for Gr7
# fene = "/Users/kushu/Downloads/fene.dat"
# FeNe_dat = ascii.read(fene)['4790.218']

# def getvalues_fromFile(file_arr, yourarr):
    
#     file_arr.astype(float)
    
#     fine_values = []
    
#     for num in yourarr:
        
#         indx = np.argmin(np.abs(file_arr - num))
#         fine_values.append(file_arr[indx])

#     return np.array(fine_values)

# wave_clean_peak = getvalues_fromFile(file_arr=FeNe_dat, yourarr=np.array(waveln_revers))
# wave_clean_gaus = getvalues_fromFile(file_arr=FeNe_dat, yourarr=np.array(waveln_revers))

# pxl_clean_peak = getvalues_fromFile(file_arr=peak_init, yourarr=np.array(pixels_revers))
# pxl_clean_gaus = getvalues_fromFile(file_arr=peak_fit, yourarr=np.array(pixels_revers))


# #now let's plot the fitted pixels
# fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(16,7))


# cs_gaus = interpolate.CubicSpline(x=wave_clean_gaus, y=pxl_clean_gaus)
# # cs_peak = interpolate.CubicSpline(x=wave_clean_peak[1:], y=pxl_clean_peak[1:])
# cs_peak = interpolate.PchipInterpolator(x=wave_clean_peak, y=pxl_clean_peak)
# # cs_peak = interpolate.Akima1DInterpolator(x=wave_clean_peak, y=pxl_clean_peak)
# # cs_peak = interpolate.KroghInterpolator(x=wave_clean_peak, y=pxl_clean_peak)

# ax[0].plot(FeNe_dat,cs_gaus(FeNe_dat),ls='',color='b',marker='o',alpha=0.3,markersize=3)
# ax[1].plot(FeNe_dat,cs_peak(FeNe_dat),ls='',color='r',marker='o',alpha=0.3,markersize=7)

# ax[0].plot(wave_clean_gaus,pxl_clean_gaus,'yo',markersize=4,alpha=0.4,ls='',fillstyle='none',markeredgecolor='k',markeredgewidth=1.8)
# ax[1].plot(wave_clean_peak,pxl_clean_peak,'go',markersize=4,alpha=0.4,ls='',fillstyle='none',markeredgecolor='g',markeredgewidth=1.8)
# ax[0].set_title('Line centers from Gauss fitting of lines',fontsize=14)
# ax[1].set_title('Line centers from Peak positions',fontsize=14)



#~~~~~~~wavelength Dispersion relation for Gr7 FeAr lamp of HFOSC
# FeAr_gr7_HFOSC = "/Users/kushu/Downloads/nFeAr_gr7_2.fits"

# #read the spectra
# spec = fits.open(FeAr_gr7_HFOSC)[0].data

# #Now get the slit profile
# FeNe_slitprfl = Spec.construct_spectra_slitprofile(spec)
# slit_prfl = biweight_location(FeNe_slitprfl,axis=0)

# FeNe_sltprfl_corr = Spec.correct_slitprofile_spectImage(spec_image=spec, slit_profile=slit_prfl*-1)

# #now identify lines and plot them, but before that make spectra from the image
# spectra_FeNe = np.sum(spec,axis=1)
# spectra_FeNe /= max(spectra_FeNe)
# conti_spec = ndimage.percentile_filter(spectra_FeNe, percentile=20,size=300)

# # peak_centers = Spec.get_peakcenters(spectra_FeNe,show_fig_peak=False,peak_threshold=0.000105,continuum_percentile=20)
# #save this table
# loc = "/Users/kushu/Downloads/Gr7_FeAr_peakTable.dat"
# # ascii.write(peak_centers,loc,format='fixed_width',overwrite=True)
# peak_centers = ascii.read(loc,format='fixed_width')

# #let's draw the gaussinas
# gauss_centr = peak_centers['Line center fit']
# gauss_sigma = peak_centers['sigma']
# gauss_amplt = peak_centers['amplitude']
# xarr = np.arange(0,max(np.shape(spec)),0.2)

# line_center_init = peak_centers['Line center init']
# heights = np.array([np.max(spectra_FeNe[peak-3:peak+3]) for peak in line_center_init])

# Gauss_spectra = []
# for i in range(len(gauss_amplt)):
    
#     gauss_line = gaussian(xarr,amplitude=gauss_amplt[i],center=gauss_centr[i],sigma=gauss_sigma[i])
#     Gauss_spectra.append(gauss_line)
    
# Final_gauss_model_spec = np.sum(Gauss_spectra,axis=0)

# #Now get the lines from the plot and pre-identified lines
# line_peak = [185,262,332,398,473,521,583,640,713,760,922,1076,1113,1158,1242,1595,1624,1664,1692,1731,1925,1976,2005,
#              2079,2112,2157,2181,2202,2297,2348,2453,2525,2603,2654]
# line_gaus = [185,262,332,398,475,521,583,640,760,922,1076,1113,1242,1595,1624,1664,1692,1731,1925,1976,2005,
#              2079,2112,2157,2181,2202,2297,2348,2453,2525,2603,2654]

# wave_peak = [7635.106,7503.86,7383.98,7272.9,7147.04,7067.2,6965.4,6871.28,6752.83,6677.28,6416.30,6172.27,6114.92,6043.22,
#               5912.08,5371.48,5328.03,5269.53,5227.16,5167.48,4879.86,4806.02,4764.86,4657.9,4609.56,
#               4545.05,4510.73,4481.81,4348.06,4277.5,4131.72,4033.80,3930,3859.9]
# wave_gaus = [7635.106,7503.86,7383.98,7272.9,7147.04,7067.2,6965.4,6871.28,6677.28,6416.30,6172.27,6114.92,
#               5912.08,5371.48,5328.03,5269.53,5227.16,5167.48,4879.86,4806.02,4764.86,4657.9,4609.56,
#               4545.05,4510.73,4481.81,4348.06,4277.5,4131.72,4033.80,3930,3859.9]


# #Now we need to fit a cubic spline over this data 
# #retrieve data from FeAr file 
# fear = "/Users/kushu/Downloads/FeAr.dat"
# fear_data = ascii.read(fear)
# fear_wave = fear_data['col1']

# #get the wavelength in increasing order and accordingly sort the pixel arrays
# sort_wave_gaus = wave_gaus[::-1]
# sort_wave_peak = wave_peak[::-1]

# sort_pixl_gaus = line_gaus[::-1]
# sort_pixl_peak = line_peak[::-1]

# #get the best values from the data arrays
# good_wave_gaus = Spec.getvalues_fromFile(fear_wave, sort_wave_gaus)
# good_wave_peak = Spec.getvalues_fromFile(fear_wave, sort_wave_peak)

# good_pixl_gaus = Spec.getvalues_fromFile(gauss_centr, sort_pixl_gaus)
# good_pixl_peak = Spec.getvalues_fromFile(line_center_init, sort_pixl_peak)


# #Now fit CublicSpline and plot to see 
# fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(16,7))


# cs_gaus = interpolate.PchipInterpolator(x=good_wave_gaus, y=good_pixl_gaus)
# cs_gaus2 = interpolate.CubicSpline(x=good_wave_gaus, y=good_pixl_gaus)
# # cs_peak = interpolate.CubicSpline(x=wave_clean_peak[1:], y=pxl_clean_peak[1:])
# cs_peak = interpolate.PchipInterpolator(x=good_wave_peak, y=good_pixl_peak)
# cs_peak2 = interpolate.CubicSpline(x=good_wave_peak, y=good_pixl_peak)
# # cs_peak = interpolate.Akima1DInterpolator(x=wave_clean_peak, y=pxl_clean_peak)
# # cs_peak = interpolate.KroghInterpolator(x=wave_clean_peak, y=pxl_clean_peak)

# ax[0].plot(fear_wave,cs_gaus(fear_wave),ls='',color='b',marker='o',alpha=0.3,markersize=3,label='PchiInterpolator')
# ax[1].plot(fear_wave,cs_peak(fear_wave),ls='',color='b',marker='o',alpha=0.3,markersize=3,label='PchiInterpolator')
# ax[0].plot(fear_wave,cs_gaus2(fear_wave),ls='',color='r',marker='o',alpha=0.3,markersize=3,label='CublicSpline')
# ax[1].plot(fear_wave,cs_peak2(fear_wave),ls='',color='r',marker='o',alpha=0.3,markersize=3,label='CublicSpline')

# ax[0].plot(good_wave_gaus,good_pixl_gaus,'go',markersize=6,alpha=0.8,ls='',fillstyle='none',markeredgecolor='k',markeredgewidth=1.8)
# ax[1].plot(good_wave_peak,good_pixl_peak,'go',markersize=6,alpha=0.8,ls='',fillstyle='none',markeredgecolor='g',markeredgewidth=1.8)
# ax[0].set_title('FeAr Gr7, Line centers from Gauss fitting of lines',fontsize=14)
# ax[1].set_title('FeAr Gr7, Line centers from Peak positions',fontsize=14)
# ax[0].set_ylim(0,3500)
# ax[1].set_ylim(0,3500)
# # ax[0].text(6000,2000,'"PchiInterpolator"',fontsize=14)
# # ax[1].text(6000,2000,'"PchiInterpolator"',fontsize=14)
# ax[0].legend()
# ax[1].legend()









