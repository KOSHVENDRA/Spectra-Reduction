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

import pickle
import os
import sys
sys.path.append("/home/koshvendra/Documents/SpectroscopyCodes/WavelengthCalibrationTool")
import recalibrate as recal

#take the referece lamp spectra along with its wavelength
# and use thewm to calibrate the new spectra.

ref_gr8 = "/home/koshvendra/Documents/MFES/HCT/Year_2023/2023-11-15KS/Reduction/spec_nFeNe_gr8.0001.fits"
ref_gr8_wave = ""



def wavelength_recalibrate_HFOSC(overwrightStarfile=False):
    
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
    ref_flux = np.array(ref_spec_data['flux'])
    ref_flux /= max(ref_flux)
    ref_spec = np.column_stack((np.array(ref_spec_data['wavelength']),ref_flux))
    
    ObjectLamp = fits.open(loc_object_file)[0].data[0][0][::-1]
    ObjectLamp /= max(ObjectLamp)
    date_obs = fits.open(loc_object_file)[0].header['DATE-OBS']
    
    wavelength_solusn, pcov = recal.ReCalibrateDispersionSolution(ObjectLamp, RefSpectrum=ref_spec)
    
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
            star_specTable[str(colnames[c])] = np.array(star_data[data_col_inFits[c]][0])
            
        #write some meta information
        infos = "The dat eof observation of this source is " + str(Stardate_obs) + ' \n \n'
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

wavelength_recalibrate_HFOSC(overwrightStarfile=True)