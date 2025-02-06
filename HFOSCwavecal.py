#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:06:34 2024

@author: koshvendra
"""

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import ascii, fits
from astropy.table import Table
import tkinter as tk
import tkinter.filedialog as fd

from scipy.signal import savgol_filter
from scipy.ndimage import percentile_filter

from scipy.interpolate import CubicSpline, UnivariateSpline

import lmfit
from lmfit.models import GaussianModel,gaussian, ConstantModel

import shutil
import pickle
import glob
import os
import sys
sys.path.append("/home/koshvendra/Documents/SpectroscopyCodes/WavelengthCalibrationTool")
import recalibrate as recal

sys.path.append("/home/koshvendra/Documents/XZTau/codes")
import HFOSC_spec as HFOSC


#take the referece lamp spectra along with its wavelength
# and use thewm to calibrate the new spectra.

ref_gr8 = "/home/koshvendra/Documents/MFES/HCT/Year_2023/2023-11-15KS/Reduction/spec_nFeNe_gr8.0001.fits"
ref_gr8_wave = ""


AllSpec_time = ['2023-11-15T16:32:43.272815',]




def wavelength_recalibrate_HFOSC(overwrightStarfile=False):
    """
    This function will ask for reference lamp spectra with calibrated (ascii file with first col: wavelength, second col: flux)
    Then it asks for lamp spectra file of HFOSC. It correlates the lamp file with reference file and them create a wavelength solution for this lamp
    Then it asks for all the stars' spectra files and use this wavelength solution to write a '.txt' file for each star' wavelength calibrtaed spectra

    Parameters
    ----------
    overwrightStarfile : bool, optional
        DESCRIPTION. Whether to overwrite the wavelength calibrated '.txt' file of each star. The default is False.

    Returns
    -------
    wavelength_solusn : array
        The wavelength solution array for the current LAMP/STAR etc.

    """
    
    in_root = tk.Tk()
    in_root.geometry("700x600")
    in_root.withdraw()
    loc_ref_file = fd.askopenfilename(title="Select a file with reference spectra, Col1: wave, Col2: flux")
    
    #select the other file 
    in_root2 = tk.Tk()
    in_root2.geometry("700x600")
    in_root2.withdraw()
    loc_object_file = fd.askopenfilename(title="Select the lamp HFOSC spectra (*.fits) to be wavelength calibrated")
    
    #Now we need to read the reference spectrum and use that to calibrate the input spectrum
    ref_spec_data = ascii.read(loc_ref_file,format='fixed_width')
    n=700
    ref_flux = np.array(ref_spec_data['flux'])       #This [0:2500] is for gr14 only. It need to be removed for othergrs
    #ref_flux = np.array(ref_flux)[n:3500]  #only for gr14, else comment it out
    ref_flux /= max(ref_flux[800:2000])
    
    #reference wavelength 
    ref_wave = np.array(ref_spec_data['wavelength'])
    # ref_wave = np.array(ref_wave)[n:3500]    #only for gr14, else comment it out
    print('refwave>>>',len(ref_wave))
    # spec_arange = np.arange(0,len(ref_flux),1)
    
    #what we will do is correlate this arange with wavelength and use that interpolation to tranfer from wavelength solution (in pixel) to wavelength units
    # Cs_pixel2wave = CubicSpline(spec_arange, ref_wave)
    
    ref_spec = np.column_stack((ref_wave,ref_flux))
    
    #now get the object lamp spectra
    ObjectLamp = fits.open(loc_object_file)[0].data[0][0][::-1]    
    ObjectLamp = np.array(ObjectLamp)[700:700+len(ref_flux)]       #only for gr14, else comment it out
    ObjectLamp /= max(ObjectLamp[800:2000])
    date_obs = fits.open(loc_object_file)[0].header['DATE-OBS']
    
    wavelength_solusn, pcov = recal.ReCalibrateDispersionSolution(ObjectLamp, RefSpectrum=ref_spec)
    
    #now use the wavelength_solusn to retrieve wavelength from pixel in A units
    # wavelength_solusn = Cs_pixel2wave(wavelength_solusn)
    
    #Now once I have the wavelength solution , I cn simply import star's files and
    #put this wavelength solution into the files and that's it
    #select the other file 
    in_root3 = tk.Tk()
    in_root3.geometry("700x600")
    in_root3.withdraw()
    loc_stars_files = fd.askopenfilenames(title="Select the stars' (multiple) HFOSC spectra (*.fits) to be wavelength calibrated")
    
    #go through each of the files and read them
    colnames = ['flux','flux-err','atmosphere']
    data_col_inFits = [0,3,2] 
    
    for f in loc_stars_files:
        #initiate a Table to save the data
        star_specTable = Table()
        
        #read the stars' spectrum fits files
        star_data = fits.open(f)[0].data
        Stardate_obs = fits.open(f)[0].header['DATE-OBS']
        
        #now save columns in the table
        star_specTable['wavelength'] = np.array(wavelength_solusn)

        for c in range(len(colnames)):
            
            count_arr = np.array(star_data[data_col_inFits[c]][0][::-1])
            count_arr = count_arr[0+700:700+len(ref_flux)]                     #only for gr14, else comment it out
            print('Countarr>>>',len(count_arr))
            star_specTable[str(colnames[c])] = np.array(count_arr)
            
        #write some meta information
        infos = "The date of observation of this source is " + str(Stardate_obs) + ' \n \n'
        star_specTable.meta['comments'] = str(infos).splitlines()
            
        #location to save the file
        loc_star_wlc_spec = os.path.dirname(f) + '/wlc' + str(os.path.basename(f).split('.fits')[0]) + '.txt'
        
        #Now write the file
        ascii.write(star_specTable,loc_star_wlc_spec,format='fixed_width',overwrite=overwrightStarfile)
    
    
    #make a plot of wavelength solution and save the plot
    fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(18,12),sharex=True,sharey=True)
    plt.tight_layout(pad=3.5)
    
    #location for saving the plot
    filename = os.path.basename(loc_object_file).split('.fits')[0]
    saveplotloc = loc_object_file.split('.fits')[0] + '_WavelengthSolution.pdf'
    saveplotlocPKL = loc_object_file.split('.fits')[0] + '_WavelengthSolution.pkl'
    
    ax[0].plot(ref_spec[:,0],ref_spec[:,1],'b',label='reference')
    ax[0].plot(ref_spec[:,0],ObjectLamp,'r',label='object')
    ax[0].legend(loc='upper left')
    ax[0].set_title(r'Lamp spectra ${\bf before}$ calibration -> '+str(filename) + ' ('+ str(date_obs)+')',fontsize=14,fontfamily='serif')
    
    ax[1].plot(ref_spec[:,0],ref_spec[:,1],'b')
    ax[1].plot(wavelength_solusn,ObjectLamp,'r')
    ax[1].set_title(r'Lamp spectra ${\bf after}$ calibration',fontsize=14,fontfamily='serif')
    
    ax[0].tick_params(axis='both',direction='in',length=7,labelsize=13)
    ax[1].tick_params(axis='both',direction='in',length=7,labelsize=13)
    ax[1].set_xlabel(r'Wavelength ($\AA$)',fontsize=14,fontfamily='serif')
    
    fig.savefig(saveplotloc)
    with open(saveplotlocPKL,'wb') as pklfile:
        pickle.dump(fig,pklfile)
    
    return wavelength_solusn


def collect_files_inOnelocation():
    
    in_root3 = tk.Tk()
    in_root3.geometry("700x600")
    in_root3.withdraw()
    loc_stars_files = fd.askopenfilenames(title="Select the files to be copied to another location")
    
    target_dir = "/home/koshvendra/Documents/XZTau/HFOSC_1dcadence/"
    
    for srcfile in loc_stars_files:
        
        filename = os.path.basename(srcfile).split('.txt')[0]
        date = os.path.dirname(srcfile).split('/')[-2]

        targetFilename = target_dir + filename + '_' + str(date) + '.txt'
        
        shutil.copyfile(src=srcfile, dst=targetFilename)
    
    return 


Gauss_model = GaussianModel() + ConstantModel()

gues_wavelength = [5572.5] #,6294#,7912,8342,8825]
atmos_wavelength = [5577.338]  #,6300.304,7913.708,8344.602,8827.096]
sl = [20]  #,20,20,20,20]
sr = [20]  #,20,20,20,20]

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
    # spec_data_hdu = fits.open(specFile)[0]
    # spec_hdr = spec_data_hdu.header
    # atmosphere_spec = spec_data_hdu.data[2][0]
    # atmosphere_spec /= max(atmosphere_spec)           #normalize the spectrum
    # wavelength_spec = np.arange(0,len(atmosphere_spec),1)*spec_hdr['CD1_1'] + spec_hdr['CRVAL1']
    
    spec_data = ascii.read(specFile,format='fixed_width')
    atmosphere_spec = spec_data['atmosphere']
    # atmosphere_spec = spec_data['flux']
    atmosphere_spec /= max(atmosphere_spec)           #normalize the spectrum
    wavelength_spec = spec_data['wavelength']
    
    #make a plot to plot the spectral line and corresponding fitting
    fig, ax = plt.subplots(nrows=1,ncols=int(len(atmos_linecen_SpecWavelength))+1,figsize=(17,6))
    axs = ax.ravel()
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
        Gauss_model.set_param_hint(name='center', value=atmos_linecen_SpecWavelength[at],min=atmos_linecen_SpecWavelength[at]-5,max=atmos_linecen_SpecWavelength[at]+5)
        Gauss_model.set_param_hint(name='c', value=0,min=-0.5,max=0.05)
        Gauss_model_param = Gauss_model.make_params()
    
        fits_gauss = Gauss_model.fit(data=spec_cut,params=Gauss_model_param,x=wave_cut)
        
        #retriev the fitted values and make a gaussian and over plot on the real data
        amp_fit, sigma_fit, cen_fit, cons_fit = fits_gauss.best_values['amplitude'],fits_gauss.best_values['sigma'],fits_gauss.best_values['center'],fits_gauss.best_values['c']
        
        #retrieve the fitted gaussian
        xarr = np.linspace(wave_cut[0],wave_cut[-1],200)
        gaussfitted = gaussian(xarr,amp_fit,cen_fit,sigma_fit) + cons_fit
        #put these figures in the plot
        axs[at].plot(wave_cut,spec_cut,'b',alpha=0.6,ls='',marker='o',markersize=4)
        axs[at].plot(wave_cut,spec_cut,'b',ls='--')
        axs[at].plot(xarr, gaussfitted,'k',alpha=0.5)
        
        axs[at].tick_params(axis='both',direction='in',length=5,labelsize=11,labelcolor='k',labelrotation=0)
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
    locs_plot = os.path.dirname(specFile) + '/AtmosLineFit_' + os.path.basename(specFile).split('.txt')[0] + ".pdf"
    fig.savefig(locs_plot)
    
    #save this table at the same location of the orignal fits file
    locs_table = os.path.dirname(specFile) + '/AtmosLineFit_' + os.path.basename(specFile)#.split('.fits')[0] + "_AtmosLineFit.txt"
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
    # spec_data_hdu = fits.open(specFile)[0]
    # spec_hdr = spec_data_hdu.header
    # spec_dat = spec_data_hdu.data
    # spectrum = spec_dat[0][0]
    # spectrum_err = spec_dat[3][0]
    # wavelength = np.arange(0,len(spectrum),1)*spec_hdr['CD1_1'] + spec_hdr['CRVAL1']
    
    #read the spectrum
    spec_data = ascii.read(specFile,format='fixed_width')
    wavelength = spec_data['wavelength']
    
    
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
    wavel_corr_specTable['wavelength'] = np.array(corrected_wavelength)
    wavel_corr_specTable['flux'] = np.array(spec_data['flux'])
    wavel_corr_specTable['flux-err'] = np.array(spec_data['flux-err'])
    wavel_corr_specTable['atmosphere'] = np.array(spec_data['atmosphere'])
    wavel_corr_specTable.meta['comments'] = "This file has wavelengths corrected using the best centroid of the atmospheric lines, \n  the lines are fitted with Gaussian \n These calibrated wavelngths are Cubic spline fitted agains the actual wavelength to correct the wavelength shift \n ".splitlines()
    
    #locatio to save this dat file and the plot below
    loc_dat = os.path.dirname(specFile) + "/wavelengthCorr" + '/waveCorr_' + os.path.basename(specFile)   #.split('.fits')[0] + '_wavelCorrected_spectrum.dat'
    loc_pdf = os.path.dirname(specFile) + "/wavelengthCorr" + '/waveCorr_' + os.path.basename(specFile).split('.txt')[0] + '.pdf'
    
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
    fig.savefig(loc_pdf)
    
    return wavel_corr_specTable





#building instrument response function from the standard star spectra and CALSPEC spectra
def build_instrumental_response(waveCorr_Standard_specFile, calspec_File,overwrite_bool=False):
    """
    This function builds an Instrument Response Function (IRF) by dividing the instrument taken standard source spectrum by the respective 
    CALSPEC spectrum

    Parameters
    ----------
    waveCorr_Standard_specFile : str
        Path to the location of the wavelength corrected standard source spectrum.
    calspec_File : str
        Path to the location of the CALSPEC file.
    overwrite_bool : bool, optional
        Whether to overwrite the IRF table in the given location. The default is False.

    Returns
    -------
    IRF_table : astropy.table
        It has Instrument Response Function and corresponding error for the wavelength.

    """
    
    #read and retrive the standard source spectrum
    waveCorr_specData = ascii.read(waveCorr_Standard_specFile,format='fixed_width')
    instrm_wavelength = waveCorr_specData['wavelength']
    instrm_flux = waveCorr_specData['flux']
    instrm_fluxerr = waveCorr_specData['flux-err']
    
    #Now read and retrieve the calspec spectrum
    calspec_data = fits.open(calspec_File)[1].data
    calspec_wavelength = calspec_data['WAVELENGTH']
    calspec_flux = calspec_data['FLUX']
    calspec_fluxerr = calspec_data['STATERROR']

    #Now we need to interpolate the calspec spectrum at the lcoation of the inetrument wavelengths
    #given the higher resolutuon of the 
    calspecFLUX_interpOn_intrumWavelength = np.interp(x=instrm_wavelength, xp=calspec_wavelength, fp=calspec_flux)
    calspecFLUX_ERR_interpOn_intrumWavelength = np.interp(x=instrm_wavelength, xp=calspec_wavelength, fp=calspec_fluxerr)
    
    #let's scale down the spectrum for the sake of calculational simplicity
    median_calspec_flux = np.median(calspecFLUX_interpOn_intrumWavelength)
    
    calspecFLUX_interpOn_intrumWavelength /= median_calspec_flux
    calspecFLUX_ERR_interpOn_intrumWavelength /= median_calspec_flux
    
    #Now since the CALSPEC and instrumental flux are written on the same wavelength, we can divide one by the other 
    #to retrieve the instrument response function (IRF)
    instru_response_function = np.divide(instrm_flux,calspecFLUX_interpOn_intrumWavelength)
    
    #We need to error propagate the flux error; errorIRF/IRF
    errorIRF_IRF = np.divide(instrm_fluxerr,instrm_flux) + np.divide(calspecFLUX_ERR_interpOn_intrumWavelength,calspecFLUX_interpOn_intrumWavelength)
    errorIRF = errorIRF_IRF * instru_response_function
    
    #Now this can be median normalized to get a instrument response function
    IRF = instru_response_function / np.median(instru_response_function)
    IRF_err = errorIRF / np.median(instru_response_function)
    
    #Now this file can be used to write a dat file
    IRF_table = Table()
    IRF_table['Wavelength (A)'] = np.array(instrm_wavelength)
    IRF_table['Flux'] = np.array(IRF)
    IRF_table['Flux-Err'] = np.array(IRF_err)
    
    IRF_table.meta['comments'] = "It contains median normalized instrument response function (IRF) for a given night \n IRF is calculated by dividing the standard source spectrum by the CALSPEC spectrum \n error propagation has been taken care of \n".splitlines()
    
    locs_table = os.path.dirname(waveCorr_Standard_specFile) + '/IRF_' + os.path.basename(waveCorr_Standard_specFile)
    
    #save the IRF table
    if not os.path.exists(locs_table):
        ascii.write(IRF_table,locs_table,format='fixed_width',overwrite=overwrite_bool)
    elif os.path.exists(locs_table) and overwrite_bool == True:
        ascii.write(IRF_table,locs_table,format='fixed_width',overwrite=overwrite_bool)
        
    return IRF_table


#we will need to build continuum level of teh spectrum very often and that. 
def build_IRFContinuum(spectrum, wavelength, deg_polyfit=9, mask_regions=None):
    """
    This function make spectrum continuum, I would suggest to use this function only when broad features are available as 
    it fits polynomial instead of percentile filter

    Parameters
    ----------
    spectrum : array
        Flux array of whcih the continuum is to be developed.
    wavelength : array
        wavelength array corresponding to the flux array .
    deg_polyfit : int, optional
        Degree of polynomial fit to the spectrum to retrieve a continuum. The default is 9.
    mask_regions : list, optional
        list of tuples where each tuple mark the starting and ending of a region to be masked. The default is None.

    Returns
    -------
    Continuum_level : array
        The normalized continuum level of the Instrumental Response Fucntion.

    """
    
    fluxArr = np.array(spectrum)
    wavelengthArr = np.array(wavelength)
    
    if mask_regions is not None:
        
        #then before we buld the continuum level, we need to mask the regions in the spectrum.
        mask = np.ones(wavelengthArr.shape,dtype=bool)
        for start, end in mask_regions:
            mask &= ~((wavelengthArr >= start) & (wavelengthArr <= end))
        
        wavelengthArr_modi = wavelengthArr[mask]
        fluxArr_modi = fluxArr[mask]
        
    else:
        wavelengthArr_modi = wavelengthArr.copy()
        fluxArr_modi = fluxArr.copy() 
        
    #Now we can simply fit a savitzky golay filter, we can alternatively fit a spline also on this masked spectrum
    cubicSplineFit = UnivariateSpline(wavelengthArr_modi, fluxArr_modi,s=0.3)
    Continuum_level_CS = cubicSplineFit(wavelengthArr)
    
    #polynomial fit 
    polyfit = np.polyfit(wavelengthArr_modi, fluxArr_modi, deg=deg_polyfit)
    polycoeff = np.poly1d(polyfit)
    #now estimate the flux over the entire wavelength range and that should be the continuum 
    Continuum_level_poly = polycoeff(wavelengthArr)
    
    # #let's use savitzky-Golay filter and get a continuum, it doesnot produce good result
    # percent_filter_continuum = percentile_filter(fluxArr_modi,percentile=30,size=int(len(fluxArr)/25))
    # Continuum_level_sg = savgol_filter(percent_filter_continuum, window_length=int(len(fluxArr)/100), polyorder=3)
    
    return Continuum_level_CS, Continuum_level_poly #, Continuum_level_sg
    


#make the spectrum IRF corrected
#correct the spectrum for the instrumental response function (IRF), using the normalized continuum of the IRF
def correct_instrumental_response(spectrum_file, IRF_file):
    """
    This function corrects the source spectrum for the Instrumental Response Function (IRF).

    Parameters
    ----------
    spec_wave : 1D array
        Wavelength array of the source spectrum.
    spec_flux : 1D array
        Flux array of the source spectrum.
    spec_fluxErr : 1D array
        Flux error array of the source spectrum.
    wave_IRF : 1D array
        Wavelength array of the IRF .
    conti_IRF : 1D array
        Normalized continuum array of the IRF.
    contiErr_IRF : 1D array
        Error associated with Normalized continuum array of the IRF.

    Returns
    -------
    IRF_corrected_sourceSpectrum : 1D array
        IRF corrected spectrum of the source.
    errorFlux_IRFcorrected_spectrum : 1D array
        error associated with the IRF_corrected_sourceSpectrum

    """
    #read the spectrum_file
    spectrum_data = ascii.read(spectrum_file,format='fixed_width')
    source_wavelengthArr = np.array(spectrum_data['wavelength'])
    source_fluxArr = np.array(spectrum_data['flux'])
    source_fluxArrErr = np.array(spectrum_data['flux-err'])
    
    #read the IRF file
    IRF_data = ascii.read(IRF_file,format='fixed_width')
    IRF_waveArr = np.array(IRF_data['wavelength'])
    IRF_contiArr = np.array(IRF_data['Median Continuum IRF'])
    IRF_contiArrErr = np.array(IRF_data['Median Continuum IRF error'])
    
    
    # #convert them into array format
    # source_wavelengthArr = np.asarray(spec_wave)
    # source_fluxArr = np.asarray(spec_flux)
    # source_fluxArrErr = np.asarray(spec_fluxErr)
    
    # #read the IRF_file and retrieve the normalized continuum of the Instrumental ersponse function and corresponding wavelength
    # IRF_waveArr = np.asarray(wave_IRF)
    # IRF_contiArr = np.asarray(conti_IRF)
    # IRF_contiArrErr = np.asarray(contiErr_IRF)
    
    #Now we need to divide the source spectrum by Insrtumental Response Function, estimated at same wavelenegth
    #In case, there is wavelength array mismatch between the two, let's interpolate the IRF_contiArr on the wavelength
    #of the source spectrum
    
    IRF_interpOn_sourceWave = np.interp(source_wavelengthArr, IRF_waveArr, IRF_contiArr)
    # IRFerr_interpOn_sourceWave = np.interp(source_wavelengthArr, IRF_waveArr, IRF_contiArr)
    
    #Now that IRF function is interpolated over the source spectrum wavelength, we can simply divide the fluxes
    IRF_corrected_sourceSpectrum = np.divide(source_fluxArr,IRF_interpOn_sourceWave)
    
    #perform error propagation
    errorFlux_Flux = np.divide(source_fluxArrErr, source_fluxArr) + np.divide(IRF_contiArrErr,IRF_contiArr)
    errorFlux_IRFcorrected_spectrum = errorFlux_Flux * IRF_corrected_sourceSpectrum

    #due to IRF correction of the spectrum, the flux could have gained some negative values thereby leading to negative error
    #And thus, errors needs to be made positive
    errorFlux_IRFcorrected_spectrum = np.abs(errorFlux_IRFcorrected_spectrum)

    return IRF_corrected_sourceSpectrum, errorFlux_IRFcorrected_spectrum


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Wavelength calibrate the ASAS-SN specta ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

wavelength_solusn = wavelength_recalibrate_HFOSC(overwrightStarfile=True)
















#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Correct the wavelength of the spectral line~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# specs = ['/home/koshvendra/Documents/XZTau/HFOSC_1dcadence_reatmos/XZTau_gr8/wlcspec_crnXZTau_gr8.0001_2023-11-15KS.txt',
#           '/home/koshvendra/Documents/XZTau/HFOSC_1dcadence_reatmos/XZTau_gr8/wlcspec_crnbXZTau_gr8.0001_2023-11-16K.txt',
#           '/home/koshvendra/Documents/XZTau/HFOSC_1dcadence_reatmos/XZTau_gr8/wlcspec_crnXZTau_gr8.0001_2023-11-17K.txt',
#           '/home/koshvendra/Documents/XZTau/HFOSC_1dcadence_reatmos/XZTau_gr8/wlcspec_crnXZTau_gr8.0001_2023-11-18KS.txt',
#           '/home/koshvendra/Documents/XZTau/HFOSC_1dcadence_reatmos/XZTau_gr8/wlcspec_crnXZTau_gr8.0001_2023-11-19KS.txt',
#           '/home/koshvendra/Documents/XZTau/HFOSC_1dcadence_reatmos/XZTau_gr8/wlcspec_crnXZTau_gr8_120.0001_2023-11-20.txt',
#           '/home/koshvendra/Documents/XZTau/HFOSC_1dcadence_reatmos/XZTau_gr8/wlcspec_crnXZTau_gr8_120.0001_2023-11-21KS.txt',
#           '/home/koshvendra/Documents/XZTau/HFOSC_1dcadence_reatmos/XZTau_gr8/wlcspec_crnXZTau_gr8_120.0001_2023-11-22KS.txt',
#           '/home/koshvendra/Documents/XZTau/HFOSC_1dcadence_reatmos/XZTau_gr8/wlcspec_crnXZTau_gr8_120.0001_2023-11-23k.txt',
#           '/home/koshvendra/Documents/XZTau/HFOSC_1dcadence_reatmos/XZTau_gr8/wlcspec_crnXZTau_gr8_120.0001_2023-11-24k.txt',
#           '/home/koshvendra/Documents/XZTau/HFOSC_1dcadence_reatmos/XZTau_gr8/wlcspec_crnXZTau_gr8_900.0001_2023-11-21KS.txt',
#           '/home/koshvendra/Documents/XZTau/HFOSC_1dcadence_reatmos/XZTau_gr8/wlcspec_crnXZTau_gr8_1200.0001_2023-11-20.txt',
#           '/home/koshvendra/Documents/XZTau/HFOSC_1dcadence_reatmos/XZTau_gr8/wlcspec_crnXZTau_gr8_1200.0001_2023-11-22KS.txt',
#           '/home/koshvendra/Documents/XZTau/HFOSC_1dcadence_reatmos/XZTau_gr8/wlcspec_crnXZTau_gr8_1200.0001_2023-11-23k.txt',
#           '/home/koshvendra/Documents/XZTau/HFOSC_1dcadence_reatmos/XZTau_gr8/wlcspec_crnXZTau_gr8_1200.0001_2023-11-24k.txt']

# for f in specs:
    
#       atmos_f = os.path.dirname(f) + '/AtmosLineFit_' + os.path.basename(f)
     
#       wavetable = getAndCorrect_wavelengthIN_HFOSCspectrum(specFile=f, atmos_lineFit_table_loc=atmos_f,overwrite_bool=True)
    
    

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Get the spectral lines center wavelength of Ha, Ca II ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# all_XZTau_files = glob.glob("/home/koshvendra/Documents/XZTau/HFOSC_1dcadence_reatmos/XZTau_gr8/wlcspec_*XZTau_gr8*.txt")

# # wavelen_lines = [6560,8496,8540,8660]
# # wavelen_actual= [6563,8498.02,8542.09,8662.14]

# # wavelen_lines = [4858,6560]
# # wavelen_actual = [4861.35,6562.8]
# atmos_lenwindo = [20,10,10,10,10,10]#,20,20]
# atmos_ritwindo = [20,10,10,10,10,10]#,20,20]

# wavelen_lines  = np.array([5571,6294,7910,8342,8825])-2# [5571,6296]
# wavelen_actual = [5577.338,6300.304,7913.708,8344.602,8827.096] #[5577.338,6300.304]

# i = 0
# for f in all_XZTau_files:
    
#     linelist = AtmosphericLines_in_HFOSCspectrum(f,wavelen_lines,wavelen_actual,atmos_lenwindo,atmos_ritwindo,overwrite_bool=True)
#     print('Done for ',i+1)
    
#     i += 1


#~~~~~~~~~~~~~~~~~~~~~~~~~~~ Plot the delta wavelength ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# all_fits = glob.glob("/home/koshvendra/Documents/XZTau/HFOSC_1dcadence_reatmos/XZTau_gr8/AtmosLineFit_*pec_*XZTau_gr8.*txt")

# fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(16,6))

# for f in range(len(all_fits)):
    
#     #read the files
#     file_data = ascii.read(all_fits[f],format='fixed_width')
    
#     ActualWave = np.array(file_data['Actual Wavelength (A)'])
#     center  = np.array(file_data['center'])
    
#     #difference
#     diff_wave = ActualWave - center
    
#     #plot them
#     ax.plot(ActualWave,diff_wave,marker='o',markersize=7,ls='')
    
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Plot all the spectra ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# # all_XZTau_files = glob.glob("/home/koshvendra/Documents/XZTau/HFOSC_1dcadence_reatmos/XZTau_gr8/wlcspec_*XZTau*txt")
# all_XZTau_files = glob.glob("/home/koshvendra/Documents/XZTau/HFOSC_1dcadence_reatmos/XZTau_gr8/wavelengthCorr/waveCorr_wlcspec_crn*XZTau_gr8*.txt")
# # all_XZTau_files = ["/home/koshvendra/Documents/XZTau/HFOSC_1dcadence_reatmos/XZTau_gr8/wlcspec_crnXZTau_gr8_1200.0001_2023-11-20.txt",
#                     # '/home/koshvendra/Documents/XZTau/HFOSC_1dcadence_reatmos/XZTau_gr8/wlcspec_crnXZTau_gr8.0001_2023-11-19KS.txt',
#                    # '/home/koshvendra/Documents/XZTau/HFOSC_1dcadence_reatmos/XZTau_gr8/wlcspec_crnbXZTau_gr8.0001_2023-11-16K.txt']

# all_XZTau_files = glob.glob("/home/koshvendra/Documents/XZTau/HFOSC_1dcadence_reatmos/XZTau_gr8/wlcspec_c*Feige34_gr8*.txt")

# fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(20,14),height_ratios=(5,2),sharex=True)
# plt.tight_layout(pad=3.2)
# calar = plt.cm.rainbow(np.linspace(0,1,len(all_XZTau_files)))


# i=0

# ds = []

# for file in all_XZTau_files:
    
#     #read the files
#     file_data = ascii.read(file,format='fixed_width')
    
#     wavelength = np.array(file_data['wavelength'])
#     flux = np.array(file_data['flux'])
#     fluxErr = np.array(file_data['flux-err'])
#     atmosSpec = np.array(file_data['atmosphere'])
    
#     # flux /= max(flux)
#     #flux += 1000
    
#     #let's normalize them
#     flux /= np.median(flux)
#     fluxErr /= np.median(flux)
#     atmosSpec /= np.median(atmosSpec)
    
#     #dateof observation
#     date = os.path.basename(file).split('.txt')[0].split('_')[-1][0:10]
    
#     if date in ds:
#         ax[0].plot(wavelength,flux,color=calar[i],alpha=0.6)
#         print('>>>>',len(flux))
#     else:
#         ax[0].plot(wavelength,flux,color=calar[i],alpha=0.6,label=str(date))
#         ds.append(date)
#         print('>>>>',len(flux))
    
        
#     # ax[0].errorbar(wavelength,flux,yerr=fluxErr,color=calar[i],alpha=0.3)
    
#     ax[1].plot(wavelength,atmosSpec,color=calar[i],alpha=0.6)
#     print('-->>>',len(atmosSpec))
#     i+=1

# ax[1].set_xlabel(r'Wavelength ($\AA$)',fontsize=14,fontfamily='serif')
# ax[1].set_ylabel('Flux',fontsize=14,fontfamily='serif')
# ax[1].set_ylabel('flux',fontsize=14,fontfamily='serif')
# ax[0].set_title('XZ Tau spectra from gr7 HFOSC',fontsize=14,fontfamily='serif')
# ax[1].set_title('Corresponding Atmospheric spectra from HFOSC',fontsize=14,fontfamily='serif')
# ax[0].tick_params(axis='both',direction='in',length=7,labelsize=14)
# ax[1].tick_params(axis='both',direction='in',length=7,labelsize=14)
# ax[0].legend(loc='upper left',fontsize=9,ncols=5)


    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#run it
# collect_files_inOnelocation()
    
# wavelength_recalibrate_HFOSC(overwrightStarfile=False)

#~~~~~~~~~~~~~~~~~~~~~~~~~ This function with build instrumental response function for gr7 and gr8 ~~~~~~~~~~~~~~~~~~~~~~~~~
# calspec = ['/home/koshvendra/Documents/XZTau/calspec/feige34_stis_006.fits','/home/koshvendra/Documents/XZTau/calspec/feige110_stisnic_008.fits']

# gr7_1 = "/home/koshvendra/Documents/XZTau/HFOSC_1dcadence/XZTau_gr7/wavelengthCorrected/WLCorr_wlcspec_crnFeige34_gr7.0001_2023-11-17K.txt"
# gr7_2 = "/home/koshvendra/Documents/XZTau/HFOSC_1dcadence/XZTau_gr7/wavelengthCorrected/WLCorr_wlcspec_crnFeige34_gr7.0001_2023-11-19KS.txt"
# gr7_3 = "/home/koshvendra/Documents/XZTau/HFOSC_1dcadence/XZTau_gr7/wavelengthCorrected/WLCorr_wlcspec_crnFeige34_gr7.0001_2023-11-23k.txt"
# gr7_4 = "/home/koshvendra/Documents/XZTau/HFOSC_1dcadence/XZTau_gr7/wavelengthCorrected/WLCorr_wlcspec_crnFeige34_gr7.0001_2023-11-24k.txt"
# gr7_5 = "/home/koshvendra/Documents/XZTau/HFOSC_1dcadence/XZTau_gr7/wavelengthCorrected/WLCorr_wlcspec_crnFeige110_gr7.0001_2023-11-15KS.txt"
# gr7_6 = "/home/koshvendra/Documents/XZTau/HFOSC_1dcadence/XZTau_gr7/wavelengthCorrected/WLCorr_wlcspec_crnFeige110_gr7.0001_2023-11-22KS.txt"

# gr7_standard = [gr7_1,gr7_2,gr7_3,gr7_4,gr7_5,gr7_6]
# gr7_num = [0,0,0,0,1,1]

# gr8_1 = "/home/koshvendra/Documents/XZTau/HFOSC_1dcadence/XZTau_gr8/wavelengthCorrected/WLCorr_wlcspec_crnFeige34_gr8.0001_2023-11-17K.txt"
# gr8_2 = "/home/koshvendra/Documents/XZTau/HFOSC_1dcadence/XZTau_gr8/wavelengthCorrected/WLCorr_wlcspec_crnFeige34_gr8.0001_2023-11-19KS.txt"
# gr8_3 = "/home/koshvendra/Documents/XZTau/HFOSC_1dcadence/XZTau_gr8/wavelengthCorrected/WLCorr_wlcspec_crnFeige34_gr8.0001_2023-11-23k.txt"
# gr8_4 = "/home/koshvendra/Documents/XZTau/HFOSC_1dcadence/XZTau_gr8/wavelengthCorrected/WLCorr_wlcspec_crnFeige110_gr8.0001_2023-11-15KS.txt"

# gr8_standard = [gr8_1,gr8_2,gr8_3,gr8_4]
# gr8_num = [0,0,0,1]

# #now use these files to develop an instrumental response function
# k = 0
# for f in gr8_standard:
    
#     IRFtable = build_instrumental_response(waveCorr_Standard_specFile=f, calspec_File=calspec[gr8_num[k]])
#     print('Stnadar source done >>',k,'/',len(gr8_standard))
#     k +=1

# ~~~~~~~~~~~~~~~~~~~~~~~~ Let's plot the Instrument response function spectra

# IRF_gr7 = glob.glob("/home/koshvendra/Documents/XZTau/HFOSC_1dcadence/XZTau_gr7/wavelengthCorrected/IRF_WLCorr_wlcspec_crnFeige*.txt")
# IRF_gr8 = glob.glob("/home/koshvendra/Documents/XZTau/HFOSC_1dcadence/XZTau_gr8/wavelengthCorrected/IRF_WLCorr_wlcspec_crnFeige*.txt")

# fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(18,5))

# calar7 = plt.cm.rainbow(np.linspace(0, 1,len(IRF_gr7)))
# k=0
# for f7 in IRF_gr7:
    
#     spec7 = ascii.read(f7,format='fixed_width')
#     wave7 = spec7['Wavelength (A)']
#     flux7 = spec7['Flux']
#     fluxerr7 = spec7['Flux-Err']
    
#     date = os.path.basename(f7).split('.txt')[0].split('_')[-1][:10]
#     ax.plot(wave7,flux7,color=calar7[k],label='gr7 '+str(date))
#     ax.errorbar(wave7,flux7,yerr=fluxerr7,color='k',alpha=0.05)

#     k +=1
    
# calar8 = plt.cm.rainbow(np.linspace(0, 1,len(IRF_gr8)))
# k=0
# for f8 in IRF_gr8:
    
#     spec8 = ascii.read(f8,format='fixed_width')
#     wave8 = spec8['Wavelength (A)']
#     flux8 = spec8['Flux']
#     fluxerr8 = spec8['Flux-Err']
    
#     date = os.path.basename(f8).split('.txt')[0].split('_')[-1][:10]
#     ax.plot(wave8,flux8,color=calar8[k],label='gr8 ' +str(date))
#     ax.errorbar(wave8,flux8,yerr=fluxerr8,color='k',alpha=0.05)

#     k +=1
    

# ax.set_xlabel(r'Wavelength $\AA$',fontsize=14,fontfamily='serif')
# ax.set_ylabel('Flux',fontsize=14,fontfamily='serif')
# ax.set_title('Normalized Instrumental response function for HFOSC Gr7 and Gr8',fontsize=14,fontfamily='serif')
# ax.legend()
# plt.tight_layout(pad=0)

# ~~~~~~~~~~~~~~~~~~~~~~~~ Now we need to develop IRF cintinuum by fitting a polynomial  ~~~~~~~~~~~~~~~~~~~~~~~~~

# IRF_gr7 = glob.glob("/home/koshvendra/Documents/XZTau/HFOSC_1dcadence/XZTau_gr7/wavelengthCorrected/IRF_WLCorr_wlcspec_crnFeige*.txt")
# IRF_gr8 = glob.glob("/home/koshvendra/Documents/XZTau/HFOSC_1dcadence/XZTau_gr8/wavelengthCorrected/IRF_WLCorr_wlcspec_crnFeige*.txt")


# #wavelength region to be masked
# gr7_wavemask = [(3946,3997),(4081,4136),(4311,4386),(4662.7,4716),(4833,4906),(5394,5435),(6250,6342),(6498,6630),(6844,6978),(7580,7676)]
# gr8_wavemask = [(5394,5435),(6250,6342),(6518,6615),(6844,6978),(7570,7721)]

# #let's build IRF continuum
# fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(18,10))

# calar7 = plt.cm.rainbow(np.linspace(0, 1,len(IRF_gr7)))
# k=0

# conti_fluxes_7 = []
# conti_fluxesErr7 = []

# for l in range(len(IRF_gr7)):     #len(IRF_gr7)
#     f7 = IRF_gr7[l]
#     spec7 = ascii.read(f7,format='fixed_width')
#     wave7 = spec7['Wavelength (A)']
#     flux7 = spec7['Flux']
#     fluxerr7 = spec7['Flux-Err']
    
#     if l ==0:
#         wave_reference = wave7
#         flux_reference = flux7
#         fluxerr_refern = fluxerr7
#         conti_fluxes_7.append(flux_reference)
#         conti_fluxesErr7.append(fluxerr_refern)
    
#     # flux7 += 0.1
    
#     conti_cs, conti_pl = build_IRFContinuum(flux7, wave7,mask_regions=gr7_wavemask)
    
#     #let's interpolate the conti_cs to the wave_reference
#     if l != 0:
#         interp_contiflux = np.interp(wave_reference,wave7,conti_cs)
#         conti_fluxes_7.append(interp_contiflux)
#         conti_fluxesErr7.append(fluxerr7)
    
#     #let's write this IRF's continuum into a table
#     file_loc = os.path.dirname(f7) + '/Continuum' + os.path.basename(f7)
#     IRFconti_table = Table()
#     IRFconti_table['wavelength'] = np.array(wave7)
#     IRFconti_table['continuum IRF'] = np.array(conti_cs)
#     # ascii.write(IRFconti_table,file_loc,format='fixed_width')
    
#     date = os.path.basename(f7).split('.txt')[0].split('_')[-1][:10]
#     ax.plot(wave7,flux7,color=calar7[k],label='gr7 '+str(date))
#     # ax.errorbar(wave7,flux7,yerr=fluxerr7,color='k',alpha=0.05)
    
#     ax.plot(wave7,conti_cs,color=calar7[k],ls="dashdot")
#     # ax.plot(wave7,conti_pl,color=calar7[k],ls="--")
#     # ax.plot(wave7,conti_sg,color=calar7[k],ls="dashdot")
    
#     k +=1
    
# #let's get the median of the IRF continuum
# median_IRF_continuum_7 = np.median(conti_fluxes_7,axis=0)

# ax.plot(wave_reference,median_IRF_continuum_7,color='k')
    
# #write these normalized continuum IRF of teh Gr7 and Gr8
# Gr7_IRFnorm_table = Table()
# Gr7_IRFnorm_table['wavelength'] = np.array(wave_reference)
# Gr7_IRFnorm_table['Median Continuum IRF'] = np.array(median_IRF_continuum_7)
# Gr7_IRFnorm_table['Median Continuum IRF error'] = np.array(np.median(conti_fluxesErr7,axis=0))
# loc_gr7 = "/home/koshvendra/Documents/XZTau/HFOSC_1dcadence/IRF/Gr7_MedianContinuumIRF.txt"
# ##ascii.write(Gr7_IRFnorm_table,loc_gr7,format='fixed_width')#,overwrite=True)

# conti_fluxes_8 = []
# conti_fluxesErr8 = []
    
# calar8 = plt.cm.rainbow(np.linspace(0, 1,len(IRF_gr8)))
# k=0
# for p in range(len(IRF_gr8)):
#     f8 = IRF_gr8[p]
#     spec8 = ascii.read(f8,format='fixed_width')
#     wave8 = spec8['Wavelength (A)']
#     flux8 = spec8['Flux']
#     fluxerr8 = spec8['Flux-Err']
    
#     # flux8 += 0.1
    
#     if p ==0:
#         wave_reference = wave8
#         flux_reference = flux8
#         fluxerr_refern = fluxerr8
#         conti_fluxes_8.append(flux_reference)
#         conti_fluxesErr8.append(fluxerr_refern)
    
#     conti_cs, conti_pl = build_IRFContinuum(flux8, wave8,mask_regions=gr8_wavemask)
    
#     #let's interpolate the conti_cs to the wave_reference
#     if p != 0:
#         interp_contiflux8 = np.interp(wave_reference,wave8,conti_cs)
#         conti_fluxes_8.append(interp_contiflux8)
#         conti_fluxesErr8.append(fluxerr8)
        
        
#     date = os.path.basename(f8).split('.txt')[0].split('_')[-1][:10]
#     ax.plot(wave8,flux8,color=calar8[k],label='gr8 ' +str(date))
#     ax.errorbar(wave8,flux8,yerr=fluxerr8,color='k',alpha=0.05)
    
#     file_loc = os.path.dirname(f8) + '/Continuum' + os.path.basename(f8)
#     IRFconti_table = Table()
#     IRFconti_table['wavelength'] = np.array(wave8)
#     IRFconti_table['continuum IRF'] = np.array(conti_cs)
#     # ascii.write(IRFconti_table,file_loc,format='fixed_width')
    
#     ax.plot(wave8,conti_cs,color=calar8[k],ls="--")
#     # ax.plot(wave8,conti_pl,color=calar8[k],ls="--")
#     # ax.plot(wave8,conti_sg,color=calar8[k],ls="dashdot")

#     k +=1
    

# ax.set_xlabel(r'Wavelength $\AA$',fontsize=14,fontfamily='serif')
# ax.set_ylabel('Flux',fontsize=14,fontfamily='serif')
# ax.set_title('Normalized Instrumental response function for HFOSC Gr7 and Gr8',fontsize=14,fontfamily='serif')
# ax.legend()
# plt.tight_layout(pad=0)

# #let's get the median of the IRF continuum
# median_IRF_continuum_8 = np.median(conti_fluxes_8,axis=0)

# ax.plot(wave_reference,median_IRF_continuum_8,color='k')


# #write these normalized continuum IRF of teh Gr7 and Gr8
# Gr8_IRFnorm_table = Table()
# Gr8_IRFnorm_table['wavelength'] = np.array(wave_reference)
# Gr8_IRFnorm_table['Median Continuum IRF'] = np.array(median_IRF_continuum_8)
# Gr8_IRFnorm_table['Median Continuum IRF error'] = np.array(np.median(conti_fluxesErr8,axis=0))
# loc_gr8 = "/home/koshvendra/Documents/XZTau/HFOSC_1dcadence/IRF/Gr8_MedianContinuumIRF.txt"
# ##ascii.write(Gr8_IRFnorm_table,loc_gr8,format='fixed_width')#,overwrite=True)



# ~~~~~~~~~~~~~~~~~~~~ Correct the source spectrum for the wavelength response ~~~~~~~~~~~~~~~~~~~~~~

# IRF_7 = "/home/koshvendra/Documents/XZTau/HFOSC_1dcadence/IRF/Gr7_MedianContinuumIRF.txt"
# IRF_8 = "/home/koshvendra/Documents/XZTau/HFOSC_1dcadence/IRF/Gr8_MedianContinuumIRF.txt"

# Gr7_XZTau = glob.glob("/home/koshvendra/Documents/XZTau/HFOSC_1dcadence/XZTau_gr7/wavelengthCorrected/*XZTau*txt")
# Gr8_XZTau = glob.glob("/home/koshvendra/Documents/XZTau/HFOSC_1dcadence/XZTau_gr8/wavelengthCorrected/*XZTau*.txt")

# parentDir_gr8 = "/home/koshvendra/Documents/XZTau/HFOSC_1dcadence/XZTau_gr8/IRF_corrected"
# parentDir_gr7 = "/home/koshvendra/Documents/XZTau/HFOSC_1dcadence/XZTau_gr7/IRF_corrected"

# i = 0
# for f in Gr8_XZTau:
    
#     #read the data file
#     SpecData = ascii.read(f,format='fixed_width')
#     SpecData_copy = SpecData.copy()
    
#     IRF_corr_flux, IRF_corr_fluxerr = correct_instrumental_response(f, IRF_8)

#     #modify the table
#     SpecData_copy['flux'] = np.array(IRF_corr_flux)
#     SpecData_copy['flux-err'] = np.array(IRF_corr_fluxerr)
    
#     #now this table spectrum can be written
#     locs = parentDir_gr8 + '/IRFcorr_' + os.path.basename(f)
#     ascii.write(SpecData_copy,locs,format='fixed_width')

#     print('Corrected for >>', i)
    
#     i+=1







#~~~~~~~~~~~~~~~~~~~~~~~~~ 
# all_XZTau_files = glob.glob("/home/koshvendra/Documents/XZTau/HFOSC_1dcadence/XZTau_gr7/wlc*XZTau*.txt")

# i = 1
# for f in all_XZTau_files:
    
#     atmos_file = AtmosphericLines_in_HFOSCspectrum(f, atmos_linecen_SpecWavelength=gues_wavelength,
#                                                     atmos_linecen_ActWavelength=atmos_wavelength, atmos_linewindow_left=sl, atmos_linewindow_right=sr)
    
#     print('Done for file number >>>>',i)
#     i+=1
    
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Let's correct the wavelength by reading the spectra and atmospheric files. This will just shift the spectra
# atmospheric_file = glob.glob("/home/koshvendra/Documents/XZTau/HFOSC_1dcadence/XZTau_gr8/Atmos*wlc*XZTau*.txt")
# spectra = glob.glob("/home/koshvendra/Documents/XZTau/HFOSC_1dcadence/XZTau_gr8/wlc*Feige*.txt")

# # The  way is to choose a spectra, then get the corresponding atmospheric file by looking at the date
# m=0
# for s in spectra:
    
#     #read the spectrum
#     spectrumData = ascii.read(s,format='fixed_width')
#     s_date = os.path.basename(s).split('.txt')[0].split('_')[-1]
    
   
#     #new location to save the spectra
#     new_loc = os.path.dirname(s) + '/wavelengthCorrected/WLCorr_' + os.path.basename(s)
#     # print(new_loc)
#     p=0
#     for at in atmospheric_file:
        
#         at_date = os.path.basename(at).split('.txt')[0].split('_')[-1]
        
#         if s_date == at_date:
#             if p==0:
#                 #read teh file
#                 atmos_data = ascii.read(at,format='fixed_width')
#                 # Then estimate the shift in wavelength
#                 waveshift = np.array(atmos_data['Actual Wavelength (A)']) - np.array(atmos_data['center'])
                
#                 #apply the wavelength shift to the spectra na dsave the new file at a new location
#                 spectrumData['wavelength'] = np.array(spectrumData['wavelength']) + waveshift
                
#                 #write the new spectra
#                 ascii.write(spectrumData,new_loc,format='fixed_width')
#                 p+=1
#             else:
#                 pass
#     m+=1
#     print('Wavelength Corrected >>>',m,'/',len(spectra))
    


# ~~~~~~~~~~~~~~~~~~~~~~~~ let's plot the XZ Tau's spectra
'''
import pickle

all_XZTau_files = glob.glob("/home/koshvendra/Documents/XZTau/HFOSC_1dcadence_Re/XZTau_gr8/wlcspec_*XZTau_gr8*.txt")
# all_XZTau_files = glob.glob("/home/koshvendra/Documents/XZTau/HFOSC_1dcadence/wlcspec_crnXZTau_gr7.0001_2023-11-15KS.txt")

fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(20,14),height_ratios=(5,2),sharex=True)
plt.tight_layout(pad=3.2)
calar = plt.cm.rainbow(np.linspace(0,1,len(all_XZTau_files)))


i=0

ds = []

for file in all_XZTau_files:
    
    #read the files
    file_data = ascii.read(file,format='fixed_width')
    
    wavelength = np.array(file_data['wavelength'])
    flux = np.array(file_data['flux'])
    fluxErr = np.array(file_data['flux-err'])
    atmosSpec = np.array(file_data['atmosphere'])
    
    # flux /= max(flux)
    #flux += 1000
    
    #let's normalize them
    flux /= max(flux)
    fluxErr /= max(flux)
    atmosSpec /= max(atmosSpec)
    
    #dateof observation
    date = os.path.basename(file).split('.txt')[0].split('_')[-1][0:10]
    
    if date in ds:
        ax[0].plot(wavelength,flux,color=calar[i],alpha=0.6)
        
    else:
        ax[0].plot(wavelength,flux,color=calar[i],alpha=0.6,label=str(date))
        ds.append(date)
    
        
    # ax[0].errorbar(wavelength,flux,yerr=fluxErr,color=calar[i],alpha=0.3)
    
    ax[1].plot(wavelength,atmosSpec,color=calar[i],alpha=0.6)
    
    i+=1

ax[1].set_xlabel(r'Wavelength ($\AA$)',fontsize=14,fontfamily='serif')
ax[1].set_ylabel('Flux',fontsize=14,fontfamily='serif')
ax[1].set_ylabel('flux',fontsize=14,fontfamily='serif')
ax[0].set_title('XZ Tau spectra from gr8 HFOSC',fontsize=14,fontfamily='serif')
ax[1].set_title('Corresponding Atmospheric spectra from HFOSC',fontsize=14,fontfamily='serif')
ax[0].tick_params(axis='both',direction='in',length=7,labelsize=14)
ax[1].tick_params(axis='both',direction='in',length=7,labelsize=14)
ax[0].legend(loc='upper left',fontsize=9,ncols=5)


#pickle.dump(fig, file('/home/koshvendra/Documents/XZTau/plots/IRFcorr__HFOSC.GR7.pkl', 'wb'))  

'''










