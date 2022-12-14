import numpy as np
from models import *
import json
from astropy import units as u
import matplotlib.pyplot as plt

import stsynphot as stsyn
from synphot import Observation,SpectralElement, units
from specutils import Spectrum1D
import csv
from scipy.interpolate import interp1d


#To find the filter curves
instrument_directories= {'nircam':"JWST_coronagraphy/",
                         'miri':"JWST_coronagraphy/MIRI/MIRI-",
                         'nirc2':'MKO/MKO-',
                         'niriss':'JWST_coronagraphy/NIRISS/NIRISS-',
                         'nircam_phot':'JWST_photometry/'}

#To find the models (in your ATMO2020 Directory)
model_instrument_directories= {'nircam':"JWST_coronagraphy/",
                         'miri':"JWST_coronagraphy/JWST_coron_MIRI/",
                         'nirc2':'MKO_WISE_IRAC/',
                         'niriss':'JWST_coronagraphy/NIRISS/NIRISS-',
                         'nircam_phot':'JWST_photometry/'}

#photometry modes are included in mask directory
nircam_mask_directories = {'mask210r':'NIRCAM_MASK210R/NIRCAM-',
                    'mask335r':'NIRCAM_MASK335R/NIRCAM-',
                    'mask430r':'NIRCAM_MASK430R/NIRCAM-',
                    'masklwb':'NIRCAM_MASKLWD/NIRCAM-',
                    'maskswb':'NIRCAM_MASKSWB/NIRCAM-',
                    'moda':'NIRCAM_modA/NIRCAM-','modb':'NIRCAM_modB/NIRCAM-',
                    'modab':'NIRCAM_modAB_mean/NIRCAM-'}

#model directory structure different from filter directories

model_nircam_mask_directories = {'mask210r':'JWST_coron_NIRCAM_MASK210R/',
                    'mask335r':'JWST_coron_NIRCAM_MASK335R/',
                    'mask430r':'JWST_coron_NIRCAM_MASK430R/',
                    'masklwb':'JWST_coron_NIRCAM_MASKLWD/',
                    'maskswb':'JWST_coron_NIRCAM_MASKSWB/',
                    }

#Stolen from poppy
lookuptable = {
            "O3V":   (50000, 0.0, 5.0),
            "O5V":   (45000, 0.0, 5.0),
            "O6V":   (40000, 0.0, 4.5),
            "O8V":   (35000, 0.0, 4.0),
            "O5I":   (40000, 0.0, 4.5),
            "O6I":   (40000, 0.0, 4.5),
            "O8I":   (34000, 0.0, 4.0),
            "B0V":   (30000, 0.0, 4.0),
            "B3V":   (19000, 0.0, 4.0),
            "B5V":   (15000, 0.0, 4.0),
            "B8V":   (12000, 0.0, 4.0),
            "B0III": (29000, 0.0, 3.5),
            "B5III": (15000, 0.0, 3.5),
            "B0I":   (26000, 0.0, 3.0),
            "B5I":   (14000, 0.0, 2.5),
            "A0V":   (9500, 0.0, 4.0),
            "A5V":   (8250, 0.0, 4.5),
            "A0I":   (9750, 0.0, 2.0),
            "A5I":   (8500, 0.0, 2.0),
            "F0V":   (7250, 0.0, 4.5),
            "F5V":   (6500, 0.0, 4.5),
            "F0I":   (7750, 0.0, 2.0),
            "F5I":   (7000, 0.0, 1.5),
            "G0V":   (6000, 0.0, 4.5),
            "G5V":   (5750, 0.0, 4.5),
            "G0III": (5750, 0.0, 3.0),
            "G5III": (5250, 0.0, 2.5),
            "G0I":   (5500, 0.0, 1.5),
            "G5I":   (4750, 0.0, 1.0),
            "K0V":   (5250, 0.0, 4.5),
            "K5V":   (4250, 0.0, 4.5),
            "K0III": (4750, 0.0, 2.0),
            "K5III": (4000, 0.0, 1.5),
            "K0I":   (4500, 0.0, 1.0),
            "K5I":   (3750, 0.0, 0.5),
            "M0V":   (3750, 0.0, 4.5),
            "M2V":   (3500, 0.0, 4.5),
            "M5V":   (3500, 0.0, 5.0),
            "M0III": (3750, 0.0, 1.5),
            "M0I":   (3750, 0.0, 0.0),
            "M2I":   (3500, 0.0, 0.0)}

def get_jwst_mag(spt,kmag,instrument,jwst_filt,filter_dir="./",jwst_mask=None,plot=False,
                norm_filter='bessel_k'):
    '''
    Input:
    spec_type: 'str', spectral type of your desired object, should match a model in the pickles catalog
    jwst_filt: 'str', desired jwst filter
    
    Output:
    mag: the calculated magnitude for input JWST filter
    
    '''
    
    src = get_pickles_spectrum(spt)

    #Get Castelli-Kuru
    teff, mh,logg = lookuptable[spt]
    src = stsyn.grid_to_spec('ck04models', teff, mh, logg)  

    #Renormalize to some input filter
    kband = SpectralElement.from_filter(norm_filter)
    src = src.normalize(kmag*units.VEGAMAG,band=kband, vegaspec=SourceSpectrum.from_vega()) 

    jwst_filt = get_filter_profiles_normalized(instrument,[jwst_filt],jwst_mask,plot=plot)[0]

    # return s.rc, jwst_filt
    obs = Observation(src, jwst_filt,force='taper')

    return obs.effstim(units.VEGAMAG,vegaspec=SourceSpectrum.from_vega())

def get_filter_profiles_normalized(instrument,filter_names, mask = None,
                        filter_dir = "../filters/",plot=True):
    
    profiles=[]
    
    filter_dir+=instrument_directories.get(instrument.lower())
    if mask is not None:
        filter_dir+=nircam_mask_directories.get(mask.lower())
        
    
    if plot==True:
        plt.figure(figsize=(10,7))
    
    for filter_name in filter_names:
        
        filt = filter_name.upper()+'.txt';file = filter_dir + filt
        f = open(file,'r')
        lines=f.readlines()[2:]
        wvs=[];transmission=[];
        for line in lines:
            wvs.append(float(line.split()[0]))
            transmission.append(float(line.split()[1]))
        transmission/=np.max(transmission)
        if plot==True:
            plt.plot(wvs,transmission,label=filter_name.upper())
            plt.fill_between(wvs,0,transmission,alpha=0.5)
            plt.ylabel("Transmission Efficiency"); plt.xlabel("Wavelength (Micron)");

        #might as well wave the transmission profiles as synphot Spectral Elements
        spec=Spectrum1D(spectral_axis = wvs*u.micron,flux=transmission*u.dimensionless_unscaled)
        bp = SpectralElement.from_spectrum1d(spec)
        profiles.append(bp)
        
    if plot==True:
        plt.legend()
        
        
    return profiles

def companion_detection_limit(host_mag,jwst_filter,contrast_curves,plot=True):
    '''
    Inputs:
    host_mag: 'float', magnitude of your host star in desired jwst filter
    jwst_filter: 'str', filter name
    curves: 'array', [separations, constrast] array of contrast curve if known
    
    Output:
    Separation, 'array'
    Constrast, 'array'
    '''
    
    separation,contrast = contrast_curves.get(jwst_filter[0:5].lower())
    
    companion_mag_limit = host_mag - 2.5*np.log10(contrast)
    
    if plot==True:
        plt.figure(figsize=(10,5))
        plt.plot(separation,companion_mag_limit)
        plt.ylabel('Minimum Companion Mag');plt.xlabel('Separation (")')
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        
    return [separation,companion_mag_limit];
    
def read_contrast_curves():
    '''
    Read in the contrast curves that we have for JWST
    '''
    curve_dir = "../curves/"
    av_filt=['f250m.csv','f300m.csv','f356w.csv','f410m.csv','f444w.csv','f1140c.csv','f1550c.csv']
    
    contrast_curves={}

    for filt_file in av_filt:
        separation=[];contrast=[]
        with open(curve_dir+filt_file,'r') as f:
            reader=csv.reader(f)
            for row in reader:
                separation.append(float(row[0]))
                contrast.append(float(row[1]))
            f.close()
        separation=np.array(separation);contrast=np.array(contrast)
        sort=np.argsort(separation)
        separation=separation[sort];contrast=contrast[sort]
        
        interpd = interp1d(separation,contrast)
        
        contrast_curves[filt_file[0:5]]=[separation,interpd(separation)]
    
    return contrast_curves

def generate_mass_curve(age,distance,companion_mags,jwst_filt,separation,model_dir,
                        bex_model_filename=None,plot=True,verbose=False):
    '''
    A function to extract a mass interpolazation function from a grid 

    Inputs: 
    age - age of your system (in Myr)
    distance - distance to the system (in pc)
    companion_mags - companion magnitude detection limit, array of size n
    jwst_filt - JWST Filter name 
    separation - Separation  output from read_contrast_curves(), array of size n
    model_dir - directory pointing to ATMO_CEQ models


    Returns: 
    mass_limits - minimum detectible companion masses
    '''

    available_filters = get_available_atmo_filters(model_dir)
    if jwst_filt.lower() not in available_filters:
        print("The chosen filter,{} , is not available in the ATMO2020 grid in this instrument configuration".format(jwst_filt.lower()))
        print("Please choose from:")
        print(available_filters)

    atmo_masses, atmo_ages, atmo_mags = read_atmo_track_for_filter(model_dir,jwst_filt);

    mass_func=get_mass_func_from_mag_atmo(age,distance,atmo_masses,atmo_ages,atmo_mags,kind='linear')
    mass_limits = mass_func(companion_mags)

    if bex_model_filename is not None:         
        if jwst_filt.lower() != 'f444w':
            print("Right now until we fix it BEX will only take the f444w filter")
            print("Retuning just the ATMO mass limits")
            return mass_limits

        with open(bex_model_filename, 'r') as f:
            bex_data = json.load(f)
        bex_ages = 10**np.array(bex_data['log(age/yr)']) / 1e6     # In Myr
        bex_masses = np.array(bex_data['mass/mearth']) / 317.8       #In MJup
        bex_abs_mags = np.array(bex_data[jwst_filt])

        if verbose: 
            print("Getting the bex function")
        bex_mass_func = get_mass_func_from_mag_bex(age,distance,bex_masses,bex_ages,bex_abs_mags)
        
        if plot==True: 
            plt.semilogy(separation,bex_mass_func(companion_mags))
            plt.semilogy(separation,mass_func(companion_mags),'--')
            plt.savefig("tmp.png",dpi=200)
            plt.close()

        if verbose: 
            print("Combining the functions")
        mass_limits = get_atmo_bex_masses(bex_mass_func,mass_func,companion_mags)


    # if plot==True:
    #     plt.semilogy(separation,mass_limits)
    #     plt.ylabel('Mass ($M_{Jup}$)');plt.xlabel('Separation (")');

    return mass_limits


def detect_companion(seps,mass_limits,comp_sep,comp_mass):
    '''
    Return a boolean whether the companion is detected or not at its given separation. 
    '''

    mass_func = interp1d(seps,mass_limits,bounds_error=False,fill_value='extrapolate')
    limit = mass_func(comp_sep)

    return (comp_mass >= limit)
    