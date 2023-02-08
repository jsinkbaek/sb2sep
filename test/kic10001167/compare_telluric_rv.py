import sys

import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation
import os
from barycorrpy import get_BC_vel, utc_tdb
import scipy.constants as scc
import matplotlib.pyplot as plt
import matplotlib
from PyAstronomy.pyasl.asl.rotBroad import rotBroad

try:
    from src.sb2sep import spectrum_processing_functions as spf
    from src.sb2sep.storage_classes import RadialVelocityOptions, SeparateComponentsOptions, RoutineOptions, load_configuration_files
    import src.sb2sep.spectral_separation_routine as ssr
    from src.sb2sep.linear_limbd_coeff_estimate import estimate_linear_limbd
    import src.sb2sep.calculate_radial_velocities as cRV
except ModuleNotFoundError:
    print('Using installed package sb2sep.')
    from sb2sep import spectrum_processing_functions as spf
    from sb2sep.storage_classes import RadialVelocityOptions, SeparateComponentsOptions, RoutineOptions, load_configuration_files
    import sb2sep.spectral_separation_routine as ssr
    from sb2sep.linear_limbd_coeff_estimate import estimate_linear_limbd
    import sb2sep.calculate_radial_velocities as cRV
import warnings

matplotlib.rcParams.update({'font.size': 10.3})


##################### VARIABLES AND OPTIONS FOR SCRIPT ##############################3
warnings.filterwarnings("ignore", category=UserWarning)
# plt.ion()
plt.ioff()
files_science = [
    'FIBi300035_step011_merge.fits',
    'FIDg060105_step011_merge.fits', 'FIBj040096_step011_merge.fits', 'FIBl010111_step011_merge.fits',
    'FIDh200065_step011_merge.fits', 'FIEh190105_step011_merge.fits', 'FIDh150097_step011_merge.fits',
    'FIBl050130_step011_merge.fits', 'FIEh020096_step012_merge.fits', 'FIBk050060_step011_merge.fits',
    'FIDg070066_step011_merge.fits', 'FIDg080034_step011_merge.fits', 'FIBl080066_step011_merge.fits',
    'FIDg050102_step011_merge.fits', 'FIDh170096_step011_merge.fits', 'FIDh160097_step011_merge.fits',
    'FIBi240077_step011_merge.fits', 'FIBi230039_step011_merge.fits', 'FIBi240074_step011_merge.fits',
    'FIBk230065_step011_merge.fits', 'FIBk060008_step011_merge.fits', 'FIFj100096_step011_merge.fits',
    'FIEh060100_step012_merge.fits', 'FIBj010039_step011_merge.fits', 'FIBk030040_step011_merge.fits',
    'FIBk140072_step011_merge.fits', 'FIDh100076_step011_merge.fits', 'FIBk040034_step011_merge.fits',
    'FIEf140066_step011_merge.fits', 'FIBj150077_step011_merge.fits', 'FIDg160034_step011_merge.fits'
]
folder_science = '/home/sinkbaek/Data/KIC10001167/'
stellar_target = "kic10001167"
wavelength_normalization_limit = (4315, 7000)   # Ångström, limit to data before performing continuum normalization
wavelength_buffer_size = 4.0                     # Ångström, padding included at ends of spectra. Useful when doing
                                                # wavelength shifts with np.roll()
delta_v = 1.0         # interpolation sampling resolution for spectrum in km/s


# # Template Spectra # #
template_spectrum_path_A = 'Data/template_spectra/4750_25_m05p00.ms.fits'
template_spectrum_path_B = 'Data/template_spectra/6250_45_m05p00.ms.fits'

load_previous = True

# # Alternative generation of options objects above through configuration files:
routine_options, sep_comp_options, rv_options = load_configuration_files(
    'routine_config.txt', 'sep_config.txt', 'rv_config.txt'
)


##################### PREPARATION BEFORE SEPARATION ROUTINE CALLS #############################
# # Prepare collection lists and arrays # #
flux_collection_list = []
wavelength_collection_list = []
date_array = np.array([])
RA_array = np.array([])
DEC_array = np.array([])
spectral_separation_array_A = np.array([])
spectral_separation_array_B = np.array([])
expt_array = np.array([])


# # # Load fits files, collect and normalize data # # #
i = 0

for filename in files_science:
    # Load observation
    wavelength, flux, date, ra, dec, exptime = spf.load_program_spectrum(folder_science + filename)
    date_array = np.append(date_array, date)
    RA_array = np.append(RA_array, ra * 15.0)  # converts unit
    DEC_array = np.append(DEC_array, dec)
    expt_array = np.append(expt_array, exptime)

    # Prepare for continuum fit
    selection_mask = (wavelength > wavelength_normalization_limit[0]) & \
                     (wavelength < wavelength_normalization_limit[1])
    wavelength = wavelength[selection_mask]
    flux = flux[selection_mask]

    # Remove values under 0
    selection_mask = (flux >= 0.0)
    flux = flux[selection_mask]
    wavelength = wavelength[selection_mask]

    # Performs continuum fit and reduces emission lines (by removing above 2.5 std from fitted continuum)
    if load_previous is False:
        wavelength, flux = spf.simple_normalizer(
            wavelength, flux, reduce_em_lines=True, plot=False, poly1deg=5, poly2deg=6
        )
        np.save(f'temp/wavelength_{i}.npy', wavelength)
        np.save(f'temp/flux_{i}.npy', flux)
    else:
        wavelength = np.load(f'temp/wavelength_{i}.npy')
        flux = np.load(f'temp/flux_{i}.npy')

    # Append to collection
    wavelength_collection_list.append(wavelength)
    flux_collection_list.append(flux)
    i += 1

asort = np.argsort(files_science)

wavelength, (flux_collection) = spf.resample_multiple_spectra(
    delta_v, (wavelength_collection_list, flux_collection_list), wavelength_a=6866, wavelength_b=6925
)
flux_collection = flux_collection[0]
print(flux_collection.shape)

# flux_collection = flux_collection[:, asort]
# files_science = np.array(files_science)[asort]

mean_flux = np.sum(flux_collection, axis=1) / flux_collection.shape[1]

from lyskryds.krydsorden import getCCF, getRV

plt.figure()
rvs = []
errs = []
for i in range(flux_collection.shape[1]):
    vel, ccf = getCCF(1-flux_collection[:, i], 1-mean_flux, rvr=71)
    rv, err = getRV(vel, ccf, poly=False, new=True, zucker=False)
    label = f'{files_science[i][:6]}  {rv:.3f} km/s'
    plt.plot(vel, ccf/np.max(ccf)-i, label=label)
    plt.plot([rv, rv], [-i, 1-i], 'k--')
    rvs.append(rv)
    errs.append(err)
plt.xlabel('Velocity [km/s]')
plt.yticks([])
plt.legend()
print(np.array([rvs, errs]).T)
# np.savetxt('tell_rv.txt', rvs)
plt.show()