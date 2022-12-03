import glob
import numpy as np
import os
from astropy.io import fits
from scipy.interpolate import interp2d, interp1d
from synphot import SourceSpectrum


def get_available_filters(model_dir):
    '''
    In a given ATMO2020 model directory, read the header 
    from the first file and figure out the available filters
    '''
    #Grab any file from this directory and read the top row
    any_file = glob.glob(model_dir+"*.txt")[0]
    with open(any_file, 'r') as f:
        header = f.readlines()[0].replace('#','').lower().split()
    
    #assume the first six columns are: mass, age teff, luminosity, radius and gravity. The rest are filters. 
    header = header[6:]
    #filter names are separated by either "-" for JWST or "_" for IRAC/MKO
    filters = [filtername.split("-")[-1].split("_")[0] for filtername in header] 

    return filters

def read_track_for_filter(model_dir, filter_name):
    '''
    Read in the 3D evolutionary track for a given filter: mass, age and magnitude
    '''
    filelist = glob.glob(model_dir+"*.txt")

    masses = []
    age_list = []
    mag_list = []

    for filename in filelist:
        
        mass,ages,mags = read_one_atmo_file(filename,filter_name)
        
        masses.append(mass)
        age_list.append(ages)
        mag_list.append(mags)
        
    sorted_by_mass = np.argsort(masses)

    return np.array(masses)[sorted_by_mass], np.array(age_list)[sorted_by_mass], np.array(mag_list)[sorted_by_mass]

def read_one_atmo_file(filename,filter_name):
    '''
    Read in one file and return the mass, the ages and the mags from a filter
    '''

    #Read in this file
    atmo_data = np.genfromtxt(filename).transpose()
    #Column names
    with open(filename, 'r') as f:
        header = f.readlines()[0].replace('#','').lower().split()
    
    #Get this file's mass
    mass = atmo_data[header.index('mass')][0]/0.0009543 #Convert to M_Jup0.000954588
    #Get the ages
    ages = atmo_data[header.index('age')]*1e3  #Convert to Myr
    
    #Figure out which index for this filter
    for i,entry in enumerate(header):
        if filter_name.lower() in entry.lower():
            # print("Found filter {} in index {} in file {}".format(filter_name,i,filename))
            filter_index = i
            continue

    #Magnitudes
    mags = atmo_data[filter_index] 

    return mass, ages, mags

def get_mass_func_from_mag(age,distance,masses,ages,mags,kind='cubic',age_interp='linear'):
    '''
    A function to extract a mass interpolazation function from a grid 

    Inputs: 
    age - age of your system (in Myr)
    distance - distance to the system (in pc)
    masses - your model grid masses of dimension n
    ages - your model grid ages (Myr), a list of length n, 
            each element a list itself of length m
    mags - your model grid magnitudes (apparent @ 10pc), a list of length n, 
            each element a list itself of length m

    Returns: 
    mass_func - a function that returns the masses for input apparent magnitudes at the 
                correponding system age and distance
    '''

    ###The next bit interpolates things to the correct age:
    model_mag_interp = []
    model_mass_interp = []

    ## At the end we should have two lists corresponding 
    ## to the mags and magnitudes at a given age
    for i,mass in enumerate(masses):

        if age_interp == 'linear':
            interp_func = interp1d(ages[i], mags[i], kind=kind,bounds_error=False)

            #Make sure we didn't go down to below 0
            if interp_func(age) > 0:
                model_mag_interp.append(interp_func(age))
                model_mass_interp.append(mass)
                
        elif age_interp == 'log':
            interp_func = interp1d(np.log10(ages[i]), mags[i], kind=kind,bounds_error=False)

            #Make sure we didn't go down to below 0
            if interp_func(np.log10(age)) > 0:
                model_mag_interp.append(interp_func(np.log10(age)))
                model_mass_interp.append(mass)
            
    #Ok great, now let's adjust for distance: 
    model_mag_interp +=  5*np.log10(distance) - 5

    #Create the interpolation function that we want to return
    mass_func = interp1d(model_mag_interp,model_mass_interp, kind =kind,
                        bounds_error=False,fill_value='extrapolate')

    return mass_func

def get_pickles_spectrum(spt,verbose=True):
    '''
    A function that retuns a pysynphot pickles spectrum for a given spectral type
    '''
    psyn_cdbs_dir = os.environ['PYSYN_CDBS']
    #Read in the pickles master list.
    pickles_dir =  psyn_cdbs_dir+"grid/pickles/dat_uvk/"
    # pickles_dir = "/Users/connorvancil/Desktop/AstroResearch/Data/trds/grid/pickles/dat_uvk/"
    pickles_filename = pickles_dir+"pickles_uk.fits"
    pickles_table = np.array(fits.open(pickles_filename)[1].data)
    pickles_filenames = [x[0].decode().replace(" ","") for x in pickles_table]
    pickles_spts = [x[1].decode().replace(" ","") for x in pickles_table]
    
    #The spectral types output by EXOSIMS are sometimes annoying
    spt=spt.upper()
    spt = spt.replace(" ","").split("/")[-1]

    #Sometimes there are fractional spectral types. Rounding to nearest integer
    spt_split = spt.split(".")
    if np.size(spt_split) > 1: 
        spt = spt_split[0] + spt_split[1][1:]

    #Get the index of the relevant pickles spectrum filename
    try: 
        ind = pickles_spts.index(spt)
    except: 
        if verbose:
            print("Couldn't match spectral type {} to the pickles library".format(spt))
            print("Assuming 'G0V'")
        ind = pickles_spts.index('G0V')

    sp = SourceSpectrum.from_file(pickles_dir+pickles_filenames[ind]+".fits",flux_unit='flam')

    return sp