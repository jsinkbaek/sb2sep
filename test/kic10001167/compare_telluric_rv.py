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
    'FIEf140066_step011_merge.fits', 'FIBj150077_step011_merge.fits', 'FIDg160034_step011_merge.fits',
    # NEW SPECTRA BELOW
    'FIGb130102_step011_merge.fits', 'FIGb200113_step011_merge.fits', 'FIGb260120_step011_merge.fits',
    'FIGc030078_step011_merge.fits', 'FIGc110124_step011_merge.fits', 'FIGc170105_step011_merge.fits',
    'FIGc280075_step011_merge.fits', 'FIGc290066_step011_merge.fits', 'FIGc290075_step011_merge.fits',
    'FIGd010114_step011_merge.fits', 'FIGd070138_step011_merge.fits', 'FIGd120038_step011_merge.fits',
    'FIGd260101_step011_merge.fits', 'FIGe040084_step011_merge.fits'
]
folder_science = '/home/sinkbaek/Data/KIC10001167/'
stellar_target = "kic10001167"
wavelength_normalization_limit = (4315, 7200)   # Ångström, limit to data before performing continuum normalization
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

    # Append to collection
    wavelength_collection_list.append(wavelength)
    flux_collection_list.append(flux)
    i += 1

sep_comp_options.use_for_spectral_separation_A = spectral_separation_array_A        # see notes earlier
sep_comp_options.use_for_spectral_separation_B = spectral_separation_array_B

if load_previous:
    wavelength = np.load('temp/wavelength.npy')
    flux_collection = np.load('temp/flux_normalized.npy')
else:
    wavelength, flux_collection = spf.resample_and_normalize_all_spectra(
        wavelength_collection_list, flux_collection_list, delta_v, plot=True,
        wavelength_limits=wavelength_normalization_limit
    )
    np.save('temp/wavelength.npy', wavelength)
    np.save('temp/flux_normalized.npy', flux_collection)

mean_flux = np.sum(flux_collection, axis=1) / flux_collection.shape[1]
mask = (wavelength > 6866) & (wavelength < 6925)
flux_collection = flux_collection[mask, :]
mean_flux = mean_flux[mask]

from lyskryds.krydsorden import getCCF, getRV
rvs_combined = np.zeros(flux_collection.shape[1])
for k in range(3):
    plt.figure()
    if k != 0:
        for i in range(flux_collection.shape[1]):
            flux_collection[:, i] = ssr.shift_spectrum(flux_collection[:, i], -rvs[i], delta_v)
    mean_flux = np.sum(flux_collection, axis=1) / flux_collection.shape[1]
    rvs = []
    errs = []
    for i in range(flux_collection.shape[1]):

        vel, ccf = getCCF(1 - flux_collection[:, i], 1 - mean_flux, rvr=71)
        vel = vel * delta_v
        rv, err = getRV(vel, ccf, poly=False, new=True, zucker=False)
        label = f'{files_science[i][:6]}  {rv*1000:.2f} m/s'
        plt.plot(vel*1000, ccf / np.max(ccf) - i, label=label)
        plt.plot([rv*1000, rv*1000], [-i, 1 - i], 'k--')
        rvs.append(rv)
        errs.append(err)
    rvs_combined += rvs
    plt.xlabel('Velocity [m/s]')
    plt.yticks([])
    plt.legend(fontsize=7)
    print(np.array([rvs*1000, errs*1000]).T)
    np.savetxt(f'tell_rv_{k}.txt', rvs)
    np.savetxt(f'tell_rv_tot.txt', rvs_combined)

# Second pass, do a bootstrap to remove lines and estimate uncertainty
plt.figure()
plt.plot(mean_flux)
plt.show()
line_markers = [
    (0, 75), (75, 150), (150, 191), (191, 250), (250, 315), (315, 357), (357, 401),
    (401, 438), (438, 488), (488, 535), (535, 701), (757, 810), (823, 950), (964, 1094),
    (1096, 1227), (1268, 1401), (1445, 1562), (1605, 1762), (1824, 1948), (1948, 2168),
    (2220, 2371), (2381, 2554)
]
from numpy.random import default_rng
rng = default_rng()
n_runs = 1000
results = np.empty((len(rvs_combined), n_runs))
print(len(line_markers)//3)
for i in range(n_runs):
    if np.mod(i, 10) == 0.0:
        print(f'{i} of {n_runs}')
    temp_mean = np.copy(mean_flux)
    temp_flux = np.copy(flux_collection)
    sample = rng.choice(len(line_markers), size=rng.integers(2, len(line_markers)//3, 1), replace=False)
    for k in range(len(sample)):
        low, high = line_markers[sample[k]]
        temp_mean[low:high+1] = 1.0
        temp_flux[low:high+1, :] = 1.0
    for j in range(flux_collection.shape[1]):
        vel, ccf = getCCF(1 - temp_flux[:, j], 1 - temp_mean, rvr=71)
        vel = vel * delta_v
        rv, err = getRV(vel, ccf, poly=False, new=True, zucker=False)
        results[j, i] = rv
print('Cross validation')
print(np.std(results, axis=1)*1000)
np.savetxt('telluric_cross_validation_std.dat', np.std(results, axis=1))
plt.figure()
for i in range(results.shape[0]):
    plt.plot([np.std(results[i, 0:x]) for x in range(50, results.shape[1])])
plt.show()


plt.figure()
plt.plot(rvs_combined, '.')
plt.show()

thar = np.loadtxt('thar_drifts.dat')
star = np.loadtxt('results_backup0227/prepared/rvA_extra_points.dat')
asort = np.argsort(files_science)
plt.figure()
plt.plot((thar-rvs_combined[asort])*1000, 'D')

thar_uncertainties_sorted = np.empty(thar.size)

print('Error estimates from ThAr')
print(' < 58750d')
selection = thar[star[asort, 0] < 58750]
print(np.std(selection)*1000)
print(np.abs(selection-np.mean(selection))*1000)
thar_uncertainties_sorted[star[asort, 0] < 58750] = np.std(selection)
print(' 58750d < x < 59250')
selection = thar[(star[asort, 0] < 59250) & (star[asort, 0] > 58750)]
print(np.std(selection)*1000)
print(np.abs(selection-np.mean(selection))*1000)
thar_uncertainties_sorted[(star[asort, 0] < 59250) & (star[asort, 0] > 58750)] = np.std(selection)
thar_uncertainties_sorted[(star[asort, 0] > 59250) & (star[asort, 0] < 59750)] = np.std(selection)


print(' 59250 < x < 59750')
selection = thar[(star[asort, 0] > 59250) & (star[asort, 0] < 59750)]
print(np.std(selection)*1000)
print(np.abs(selection-np.mean(selection))*1000)
print(' 59750 < x < 60250')
selection = thar[(star[asort, 0] > 59750) & (star[asort, 0] < 60250)]
print(np.std(selection)*1000)
print(np.abs(selection-np.mean(selection))*1000)
thar_uncertainties_sorted[(star[asort, 0] > 59750) & (star[asort, 0] < 60250)] = np.std(selection)

np.savetxt('thar_uncertainties_sorted.dat', thar_uncertainties_sorted)


plt.figure()
plt.plot(star[asort, 0], thar*1000, 'D', label='ThAr drift')
plt.plot(star[asort, 0], rvs_combined[asort]*1000, 's', label='Telluric RV')
plt.legend()
plt.show()
