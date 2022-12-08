import glob
import numpy as np
import os
from astropy.io import fits
from scipy.interpolate import interp2d, interp1d
from synphot import SourceSpectrum

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

def get_available_atmo_filters(model_dir):
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

def read_atmo_track_for_filter(model_dir, filter_name):
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

def get_mass_func_from_mag_atmo(age,distance,masses,ages,mags,kind='linear',age_interp='linear'):
    '''
    A function to extract a mass interpolazation 
    function from a grid using the formatting of how I read in the ATMO2020 models

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
                        bounds_error=False,fill_value=np.nan)

    return mass_func

def get_mass_func_from_mag_bex(input_age,input_distance,bex_masses,bex_ages,bex_abs_mags):
    '''
    A function to extract a mass interpolazation 
    function from a grid using the formatting of how I read in the BEX2020 models

    All of this is repurposed from a function written by Aarynn Carter 'mag_to_mass_func'

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

    bex_mags = bex_abs_mags + 5*np.log10(input_distance/10)

    bex_ages_nodups = list(dict.fromkeys(bex_ages))
    bex_ages_diff = [abs(x-input_age) for x in bex_ages_nodups]
    bex_masses_nodups =  list(dict.fromkeys(bex_masses))

    if min(bex_ages_diff) == 0:
        #Exact age in grid, less interpolation needed.
        closest_age = bex_ages_nodups[bex_ages_diff.index(np.min(bex_ages_diff))]
        closest_ages = [closest_age, closest_age]
    else:
        #Need two points of interpolation
        min_diffs_index = [bex_ages_diff.index(x) for x in np.partition(bex_ages_diff, 1)[0:2]]
        closest_ages = [bex_ages_nodups[x] for x in min_diffs_index]

    common_masses = set(bex_masses[np.where(bex_ages == closest_ages[0])]).intersection(bex_masses[np.where(bex_ages == closest_ages[1])])
    common_masses = list(sorted(common_masses))   #The ones we want to keep

    for bex_age in bex_ages:
        row_indexes = np.where(bex_ages==bex_age)
        age_masses = bex_masses[row_indexes]
        if len(age_masses) < len(common_masses): #Number of masses too small, cut this age
            bex_ages = np.delete(bex_ages, row_indexes)
            bex_masses = np.delete(bex_masses, row_indexes)
            bex_mags = np.delete(bex_mags, row_indexes)	
        elif len(age_masses) == len(common_masses): 
            if not np.array_equal(age_masses,common_masses): #Number of masses correct, but does not match common_masses, so not enough
                bex_ages = np.delete(bex_ages, row_indexes)
                bex_masses = np.delete(bex_masses, row_indexes)
                bex_mags = np.delete(bex_mags, row_indexes)
            else: #Number of masses must be correct and match
                continue
        else: #Number of masses too large, must trim
            for age_mass in age_masses:
                row_index = row_indexes[0][np.where(age_masses==age_mass)[0][0]]   #Indexing quite specific here as using np arrays
                if age_mass not in common_masses:
                    bex_ages = np.delete(bex_ages, row_index)
                    bex_masses = np.delete(bex_masses, row_index)
                    bex_mags = np.delete(bex_mags, row_index)

    # Remake non-duplicate arrays with new trimmed grid
    bex_ages_nodups = list(dict.fromkeys(bex_ages))
    bex_masses_nodups =  list(dict.fromkeys(bex_masses))

    #Make magnitude array 2D with dimensions len(age), len(masses)
    bex_mags = np.reshape(bex_mags, (len(bex_ages_nodups), len(bex_masses_nodups))).transpose()

    x = np.array(bex_ages_nodups)
    y = np.array(bex_masses_nodups)
    z = np.array(bex_mags)

    agemass_interp = interp2d(x, y, z, kind='linear')

    mag_interp = agemass_interp([input_age]*len(bex_masses_nodups), bex_masses_nodups)
    mag_interp = [x[0] for x in mag_interp]

    mass_interp = interp1d(mag_interp, bex_masses_nodups, kind='slinear', bounds_error=False, fill_value=np.nan)

    return mass_interp

def get_mag_func_from_mass_atmo(age,distance,masses,ages,mags,kind='linear',age_interp='linear'):
    '''
    A function to extract a magnitude interpolation function from a grid 

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
    mass_func = interp1d(model_mass_interp, model_mag_interp, kind =kind,
                        bounds_error=False,fill_value='extrapolate')

    return mass_func

def get_atmo_bex_masses(bex_mass_func, atmo_mass_func,magnitudes):
    '''
    Combine two functions that interpolate masses based on magnitudes for two different model grids. 

    Copy and pasted from Aarynn Carter's code

    Inputs: 

    bex_mass_interp - A function that given apparent magnitudes returns masses based on the BEX models
    atmo_mass_interp - A function that given apparent magnitudes returns masses based on the ATMO2020 models
    magnitudes - A list of apparent magnitudes
    '''

    bex_mass_interp = bex_mass_func(magnitudes)
    atmo_mass_interp = atmo_mass_func(magnitudes)

    masses = np.nanmean(np.vstack([atmo_mass_interp,bex_mass_interp]),axis=0)

    # crossover_index = np.argwhere(np.logical_and(np.isfinite(atmo_mass_interp), np.isfinite(bex_mass_interp)))

    # atmo_masses = atmo_mass_interp[np.argwhere(np.isnan(bex_mass_interp))]
    # bex_masses = bex_mass_interp[np.argwhere(np.isnan(atmo_mass_interp))]

    # if crossover_index.size == 0:
    #     #There is no crossover, variety of situations
    #     if not np.isnan(atmo_mass_interp).any():
    #         #Can all be done by ATMO
    #         masses =  atmo_mass_interp
    #     elif np.isnan(bex_mass_interp).any():
    #         #Can all be done by BEX
    #         masses = bex_mass_interp
    #     else:
    #         #There is no overlap
    #         pre_cross = atmo_masses[np.where(atmo_masses > bex_masses[0])[0][-1]:]
    #         post_cross = bex_masses[:np.where(bex_masses < atmo_masses[-1])[0][1]]
    #         atmo_fill = np.linspace(atmo_masses[-1], post_cross[-1], len(post_cross)+1)[1:-1]
    #         bex_fill = np.linspace(pre_cross[0], bex_masses[0], len(pre_cross)+1)[1:-1]
    #         atmo_evo = np.concatenate([pre_cross, atmo_fill, [post_cross[-1]]])
    #         bex_evo = np.concatenate([[pre_cross[0]], bex_fill, post_cross])

    #         fill = np.mean([atmo_evo, bex_evo], axis=0)
    #         atmo_pre = atmo_masses[:np.where(atmo_masses == fill[0])[0][0]]
    #         bex_post = bex_masses[1+np.where(bex_masses == fill[-1])[0][0]:]
    #         masses = np.concatenate([atmo_pre, fill, bex_post])
    #         masses = [i[0] for i in masses]		
    # else:
    #     #There is a crossover
    #     atmo_cross = atmo_mass_interp[crossover_index]
    #     bex_cross = bex_mass_interp[crossover_index]

    #     if len(atmo_masses) == 0:
    #         #There are no "bad values" for BEX, no pre-cross
    #         cross = np.mean([atmo_cross, bex_cross], axis=0)
    #         masses = np.concatenate([cross, bex_masses])
    #     elif len(bex_masses) == 0:
    #         #There are no "bad values" for ATMO
    #         cross = np.mean([atmo_cross, bex_cross], axis=0)
    #         pre_cross = np.where(atmo_masses >= np.nanmax(bex_cross))[0]
    #         atmo_masses[pre_cross] = np.linspace(atmo_masses[np.nanmin(pre_cross)-1], np.nanmax(cross), num=2+len(pre_cross))[1:-1]
    #         masses = np.concatenate([atmo_masses, cross])
    #     else:
    #         #Pre Cross
    #         pass_flag, c = False, 0
    #         while pass_flag == False:
    #             try:
    #                 if c < 10:
    #                     pre_cross = atmo_masses[np.where(atmo_masses > bex_cross[c])[0][-1]:]
    #                     pass_flag = True
    #                 else:
    #                     pre_cross = atmo_masses[np.where(atmo_masses > bex_masses[0])[0][-1]:]
    #                     pass_flag = True
    #             except:
    #                 c += 1
            
    #         #Post Cross
    #         pass_flag, c = False, -1
    #         while pass_flag == False:
    #             try:
    #                 if c > -10:
    #                     post_cross = bex_masses[:np.where(bex_masses < atmo_cross[c])[0][1]]
    #                     pass_flag = True
    #                     funky_flag = False
    #                 else:
    #                     # print(len(np.where(bex_masses < atmo_masses[-1])[0][1]))
    #                     if np.size(np.where(bex_masses < atmo_masses[-1])[0]) > 0:
    #                         post_cross = bex_masses[:np.where(bex_masses < atmo_masses[-1])[0][1]]
    #                         pass_flag = True
    #                         funky_flag = False
    #                     else: 
    #                         post_cross = np.array([bex_masses[-1]])
    #                         pass_flag = True
    #                         funky_flag = True
    #             except Exception as e:
    #                 c -= 1
            
    #         atmo_fill = np.linspace(atmo_cross[-1], post_cross[-1], len(post_cross)+1)[1:-1]
    #         bex_fill = np.linspace(bex_cross[0], pre_cross[0], len(pre_cross)+1)[1:-1]
    #         atmo_evo = np.concatenate([pre_cross, atmo_mass_interp[crossover_index], atmo_fill,[post_cross[-1]]])
    #         bex_evo = np.concatenate([[pre_cross[0]], bex_fill, bex_mass_interp[crossover_index], post_cross])

    #         fill = np.nanmean([atmo_evo, bex_evo], axis=0)
    #         atmo_pre = atmo_masses[:np.where(atmo_masses == fill[0])[0][0]]

    #         #Maybe there's no post-crossover
    #         if funky_flag:
    #             masses = np.concatenate([atmo_pre, fill[:-1]])
    #             masses = masses[:np.size(magnitudes)]
    #         else:
    #             bex_post = bex_masses[1+np.where(bex_masses == fill[-1])[0][0]:]
    #             masses = np.concatenate([atmo_pre, fill, bex_post])
    #             masses = masses[:np.size(magnitudes)]
        
    #     masses = [i[0] for i in masses]

    return np.array(masses)

def match_spt(spt,lookuptable=lookuptable):
    available_keys=list(lookuptable.keys())
    matches=[];diffs=[]
    for key in available_keys:
        if spt[0]==key[0]:
            matches.append(key)
    for match in matches:
        diffs.append(np.abs(int(spt[1])-int(match[1])))
    forced_spt= spt[0] + str(matches[np.where(diffs==np.min(diffs))[0][0]])[1]+'V'
    return forced_spt

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