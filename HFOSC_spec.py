#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 16:26:28 2024

@author: kushu
"""

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import ascii, fits
from astropy.table import Table

from scipy.interpolate import CubicSpline

from lmfit.models import GaussianModel,gaussian, ConstantModel

import glob
import os


#~~~~~~~~~~~~~~~~~ This code will serve as a stand point for the spectrum of the 2MASS J16133650-2503473 soruce
file_HFOSC = "/Users/kushu/Documents/Projects/ASASSN_V/data/HFOSC/wlspec_crnASASSN_V.0001.fits"

#read the file and plot the spectra
# HFOSC_specData = fits.open(file_HFOSC)[0]
# HFOSC_head = HFOSC_specData.header
# HFOSC_flux = HFOSC_specData.data[2][0]
# HFOSC_flux /= max(HFOSC_flux)

# wavelength = np.arange(0,len(HFOSC_flux),1)*HFOSC_head['CD1_1'] + HFOSC_head['CRVAL1']



##plots
# fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(17,6))
# plt.tight_layout(pad=2)
# ax.plot(wavelength,HFOSC_specData.data[0][0],color='b')
# ax.set_xlabel(r'$\AA$',fontsize=14,fontfamily='serif')
# ax.set_ylabel('Normalized Flux',fontsize=14,fontfamily='serif')
# ax.set_title('HFOSC spectra of 2MASS J16133650-2503473 soruce (2024-07-09)',fontsize=14,fontfamily='serif')
# ax.tick_params(axis='both',direction='in',length=8,labelsize=11,labelfontfamily='serif')



#~~~~~~~~~~~ Wavelength correction in the HFOSC spectra ~~~~~~~~~~~
Gauss_model = GaussianModel() + ConstantModel()

gues_wavelength = [5571,6294,7912,8342,8825]
atmos_wavelength = [5577.338,6300.304,7913.708,8344.602,8827.096]
sl = [20,20,20,20,20]
sr = [20,20,20,20,20]

#~~~~~~~~~~~~~~~~~~~write a function to do perform wavelength shift correction of the HFOSC spectrum following OI atmospheric lines

def AtmosphericLines_in_HFOSCspectrum(specFile,atmos_linecen_SpecWavelength,atmos_linecen_ActWavelength, atmos_linewindow_left, atmos_linewindow_right,overwrite_bool=False, Gauss_model = GaussianModel() + ConstantModel()):
    """
    This functuon takes the IRAF reduced spectrum file, retrives atmospgeric spectral lines whose wavelengths are known, 
    #and estimates line center by fitting a gaussain over the spectral line. It produces one .dat file which saves all the fitting params and
    anotehr plot which shows the fitted lines.
    **The spectral line should cover the important wavelength range to be used in scientific analysis**

    Parameters
    ----------
    specFile : str
        Path to location of the spectral file.
    atmos_linecen_SpecWavelength : list 
        line centers of the visually identified atmospheric line.
    atmos_linecen_ActWavelength : list
        Actual wavelength of the atmospheric lines.
    atmos_linewindow_left : list
        left window around the line center to make a line cutout.
    atmos_linewindow_right : list
        left window around the line center to make a line cutout.
    overwrite_bool : bool, optional
        Whether to overwrite the fitted. The default is False.
    Gauss_model : lmfot.model, optional
        LMFIT model to be used for atmospheric line fitting. The default is GaussianModel() + ConstantModel().
        **It model is changed, mind to change the parameetrs of the model accordingly**

    Returns
    -------
    atmos_lineFit_table : astropy.table
        Fitted parameters of the spectral lines.

    """
    
    #retrive the atmospheric spectra from the spectra file
    #read the specfile and retrieve the atmospheric spectra
    spec_data_hdu = fits.open(specFile)[0]
    spec_hdr = spec_data_hdu.header
    atmosphere_spec = spec_data_hdu.data[2][0]
    atmosphere_spec /= max(atmosphere_spec)           #normalize the spectrum
    wavelength_spec = np.arange(0,len(atmosphere_spec),1)*spec_hdr['CD1_1'] + spec_hdr['CRVAL1']
    
    #make a plot to plot the spectral line and corresponding fitting
    fig, axs = plt.subplots(nrows=1,ncols=int(len(atmos_linecen_SpecWavelength)),figsize=(17,6))
    ax = axs.ravel()
    plt.tight_layout(pad=1.7)
    
    #center axis
    cen_axs = int(len(ax)/2)
    ax[cen_axs].set_xlabel(r'Wavelength ($\AA$)',fontsize=13,fontfamily='serif')
    ax[0].set_ylabel('Scaled Count',fontsize=13,fontfamily='serif')
    ax[cen_axs].set_title('Atmospheric OI lines for wavelength calibration -->'+str((os.path.basename(specFile)).split('.fits')[0]),fontsize=13,fontfamily='serif')
    
    #dictionary to save all the fitted parameters
    param_names = ['center','amplitude','sigma','c']
    fit_dic = {}
    for n in param_names:
        fit_dic[str(n)] = []
        

    # Now let's loop through each of the atmospheric lines
    for at in range(len(atmos_linecen_SpecWavelength)):
        
        #we nned to get the spectral cutout
        indx_specCutout = np.argmin(np.abs(wavelength_spec - atmos_linecen_SpecWavelength[at]))
        wave_cut = wavelength_spec[indx_specCutout - atmos_linewindow_left[at] : indx_specCutout + atmos_linewindow_right[at]]
        spec_cut= atmosphere_spec[indx_specCutout - atmos_linewindow_left[at] : indx_specCutout + atmos_linewindow_right[at]]
        
        #bring the cutout to zero state
        spec_cut -= min(spec_cut)
        
        #lets amke some guess for the gaussian fitting parameters
        spec_amp = max(spec_cut)
        
        #now fit a gaussian, define parameters
        Gauss_model.set_param_hint(name='amplitude', value=spec_amp) #,min=amp_init-0.3,max=amp_init + 0.9)
        Gauss_model.set_param_hint(name='sigma', value=4,min=1,max=12)
        Gauss_model.set_param_hint(name='center', value=atmos_linecen_SpecWavelength[at],min=atmos_linecen_SpecWavelength[at]-4,max=atmos_linecen_SpecWavelength[at]+4)
        Gauss_model.set_param_hint(name='c', value=0,min=-0.5,max=0.05)
        Gauss_model_param = Gauss_model.make_params()
    
        fits_gauss = Gauss_model.fit(data=spec_cut,params=Gauss_model_param,x=wave_cut)
        
        #retriev the fitted values and make a gaussian and over plot on the real data
        amp_fit, sigma_fit, cen_fit, cons_fit = fits_gauss.best_values['amplitude'],fits_gauss.best_values['sigma'],fits_gauss.best_values['center'],fits_gauss.best_values['c']
        
        #retrieve the fitted gaussian
        xarr = np.linspace(wave_cut[0],wave_cut[-1],200)
        gaussfitted = gaussian(xarr,amp_fit,cen_fit,sigma_fit) + cons_fit
        #put these figures in the plot
        ax[at].plot(wave_cut,spec_cut,'b',alpha=0.6,ls='',marker='o',markersize=4)
        ax[at].plot(wave_cut,spec_cut,'b',ls='--')
        ax[at].plot(xarr, gaussfitted,'k',alpha=0.5)
        
        axs[at].tick_params(axis='both',direction='in',length=5,labelsize=11,labelfontfamily='serif',labelcolor='k',labelrotation=0)
        axs[at].axvline(cen_fit,ymin=0,ymax=1,color='k',ls='--',alpha=0.88,lw=1.2)
        axs[at].text(cen_fit-4,0, str(np.round(cen_fit,3)),rotation=90,fontsize=10,fontfamily='serif',color='k',alpha=0.7)
           
        #actual wavelength
        axs[at].axvline(atmos_linecen_ActWavelength[at],ymin=0,ymax=1,color='g',ls='--',alpha=0.99,lw=1.2)
        axs[at].text(atmos_linecen_ActWavelength[at]+1,0,str(atmos_linecen_ActWavelength[at]),fontsize=10,fontfamily='serif',rotation=90,color='g')
            
        for nn in param_names:
            fit_dic[str(nn)].append(fits_gauss.best_values[str(nn)])
            
    #save in a table
    atmos_lineFit_table = Table()
    for nnn in param_names:
        atmos_lineFit_table[str(nnn)] = np.array(fit_dic[str(nnn)])
        
    #add the actual wavelength of the spectral lines to the table also
    atmos_lineFit_table.add_column(np.array(atmos_linecen_ActWavelength),index=0,name='Actual Wavelength (A)')
        
    #save this plot 
    locs_plot = os.path.dirname(specFile) + '/' + os.path.basename(specFile).split('.fits')[0] + "_AtmosLineFit.pdf"
    fig.savefig(locs_plot)
    
    #save this table at the same location of the orignal fits file
    locs_table = os.path.dirname(specFile) + '/' + os.path.basename(specFile).split('.fits')[0] + "_AtmosLineFit.dat"
    if not os.path.exists(locs_table):
        ascii.write(atmos_lineFit_table,locs_table,format='fixed_width',overwrite=overwrite_bool)
    elif os.path.exists(locs_table) and overwrite_bool==True:
        ascii.write(atmos_lineFit_table,locs_table,format='fixed_width',overwrite=overwrite_bool)
    else: 
        pass
        
    
    return atmos_lineFit_table


#Now we can use the output table of the above fucntion to get the actual and corresponding wavelength (in the spectra) of the atmospheric lines
def getAndCorrect_wavelengthIN_HFOSCspectrum(specFile,atmos_lineFit_table_loc,overwrite_bool=False):
    """
    Fits a Cubic spline over the calibrated and actual wavelength of the atmospheric lines, found in 'AtmosphericLines_in_HFOSCspectrum' function.
    #then using the Cubic spline fit, it corrects the wavelength association

    Parameters
    ----------
    specFile : str
        Path to the lcoation of the original spectral file.
    atmos_lineFit_table_loc : atr
        path to the location of the table with atmoshperic lines actual and calibrated (from lamp) wavelength.
    overwrite_bool : bool, optional
        Wheter to save the dat file with wavelength corrected spectra or not. The default is False.

    Returns
    -------
    wavel_corr_specTable : astropy.table
        The table contains the corrected wavelengtha long with the flux and flux-err.

    """
    
    #read the spectrum file and retrieve the data
    spec_data_hdu = fits.open(specFile)[0]
    spec_hdr = spec_data_hdu.header
    spec_dat = spec_data_hdu.data
    spectrum = spec_dat[0][0]
    spectrum_err = spec_dat[3][0]
    wavelength = np.arange(0,len(spectrum),1)*spec_hdr['CD1_1'] + spec_hdr['CRVAL1']
    
    #read the table and do a cubic spline fit
    atmos_lineFit_table = ascii.read(atmos_lineFit_table_loc,format='fixed_width')
    atmos_wavel_gausFit = np.array(atmos_lineFit_table['center'])
    atmos_wavel_actual  = np.array(atmos_lineFit_table['Actual Wavelength (A)'])
    
    #shifted wavelengths (actual wavelengths - calibrated)
    shifted_atmos_wavel = atmos_wavel_actual - atmos_wavel_gausFit
    
    #fit cubic spline over the atmospheric lines calibrated and actual wavelengths
    CubicSpline_fit = CubicSpline(atmos_wavel_gausFit, shifted_atmos_wavel)
    
    #estimate the cubic spline fitting
    fitted_cubicspline = CubicSpline_fit(wavelength)
    
    #adding this 'fitted_cubicspline' to actual wavelength will give us teh corrected wavelength
    corrected_wavelength = wavelength + fitted_cubicspline
    
    #Now save this corrected wavelength along with the flux and respective error in a dat file
    wavel_corr_specTable = Table()
    wavel_corr_specTable['Wavelength (A)'] = np.array(corrected_wavelength)
    wavel_corr_specTable['Flux'] = np.array(spectrum)
    wavel_corr_specTable['Flux-Err'] = np.array(spectrum_err)
    wavel_corr_specTable.meta['comments'] = "This file has wavelengths corrected using the best centroid of the atmospheric lines, \n  the lines are fitted with Gaussian \n These calibrated wavelngths are Cubic spline fitted agains the actual wavelength to correct the wavelength shift \n ".splitlines()
    
    #locatio to save this dat file and the plot below
    loc_dat = os.path.dirname(specFile) + '/' + os.path.basename(specFile) + '_wavelCorrected_spectrum.dat'
    loc_plt = os.path.dirname(specFile) + '/' + os.path.basename(specFile) + '_wavelCorrected_spectrum.pdf'
    
    #save the spectrum table
    if not os.path.exists(loc_dat):
        ascii.write(wavel_corr_specTable,loc_dat,format='fixed_width',overwrite=overwrite_bool)
    elif os.path.exists(loc_dat) and overwrite_bool == True:
        ascii.write(wavel_corr_specTable,loc_dat,format='fixed_width',overwrite=overwrite_bool)
        
    
    #we need to plot and ssave the cubic spline fitting
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(8,5))
    ax.plot(atmos_wavel_gausFit,shifted_atmos_wavel,'ko',markersize=7,alpha=0.4,ls='',label='Atmospheric lines')
    ax.plot(wavelength, fitted_cubicspline,color='r',ls='-',label='Fitted Cubic Spline')
    ax.set_xlabel(r'Wavelength from Lamp ($\AA$)',fontsize=13,fontfamily='serif')
    ax.set_ylabel('Shift (= Actual - Lamp) wavelength',fontsize=13,fontfamily='serif')
    ax.set_title('Wavelength Shift Profile using Atmosphreic lines',fontsize=13,fontfamily='serif')
    plt.tight_layout(pad=1)
    
    #saving this plot
    fig.savefig(loc_plt)
    
    return wavel_corr_specTable


Atmos_linefit_Table = AtmosphericLines_in_HFOSCspectrum(file_HFOSC,gues_wavelength,atmos_wavelength,sl,sr,overwrite_bool=True)

atmos_line_loc = os.path.dirname(file_HFOSC) + '/' + os.path.basename(file_HFOSC).split('.fits')[0] + "_AtmosLineFit.dat"
wavel_corr_specTable = getAndCorrect_wavelengthIN_HFOSCspectrum(file_HFOSC, atmos_lineFit_table_loc=atmos_line_loc,overwrite_bool=True)










