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

matplotlib.rcParams.update({'font.size': 25})


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
    'FIGd260101_step011_merge.fits'
]
files_science = sorted(files_science)

folder_science = '/home/sinkbaek/Data/KIC10001167/'

use_for_spectral_separation = np.array([
    'FIBi230039_step011_merge.fits', 'FIBi240074_step011_merge.fits', 'FIBi240077_step011_merge.fits',
    'FIBj150077_step011_merge.fits', 'FIBk030040_step011_merge.fits', 'FIBk040034_step011_merge.fits',
    'FIBk050060_step011_merge.fits', 'FIBk060008_step011_merge.fits', 'FIBk140072_step011_merge.fits',
    'FIDg050102_step011_merge.fits', 'FIDg060105_step011_merge.fits', 'FIDg070066_step011_merge.fits',
    'FIDg080034_step011_merge.fits', 'FIDh100076_step011_merge.fits', 'FIDh150097_step011_merge.fits',
    'FIDh160097_step011_merge.fits', 'FIDh170096_step011_merge.fits', 'FIDh200065_step011_merge.fits',
    'FIEf140066_step011_merge.fits', 'FIEh020096_step012_merge.fits', 'FIEh060100_step012_merge.fits',
    'FIEh190105_step011_merge.fits', 'FIFj100096_step011_merge.fits', 'FIGb130102_step011_merge.fits',
    'FIGb200113_step011_merge.fits', 'FIGb260120_step011_merge.fits', 'FIGc030078_step011_merge.fits',
    'FIGc170105_step011_merge.fits', 'FIGd010114_step011_merge.fits', 'FIGd070138_step011_merge.fits',
    'FIGd120038_step011_merge.fits', 'FIGd260101_step011_merge.fits'
])
observatory_location = EarthLocation.of_site("Roque de los Muchachos")
observatory_name = "Roque de los Muchachos"
stellar_target = "kic10001167"
wavelength_normalization_limit = (4315, 7000)   # Ångström, limit to data before performing continuum normalization
wavelength_RV_limit = (4315, 7000)              # Ångström, the area used after normalization
wavelength_buffer_size = 8.0                     # Ångström, padding included at ends of spectra. Useful when doing
                                                # wavelength shifts with np.roll()
wavelength_intervals_full = [(4400, 5825)]      # Ångström, the actual interval used.
wavelength_intervals = [                        # Intervals used for error calculation
    (4400, 4690), (4690, 4980), (4980, 5270), (5270, 5560), (5560, 5850),
    (5985, 6275), (6565, 6840)
]
delta_v = 1.0         # interpolation sampling resolution for spectrum in km/s

system_RV_estimate = -103.40          # to subtract before routine
orbital_period_estimate = 120.3900601075     # for ignoring component B during eclipse

# # Template Spectra # #
template_spectrum_path_A = 'Data/template_spectra/4750_25_m05p04.ms.fits'
template_spectrum_path_B = 'Data/template_spectra/6250_45_m05p04.ms.fits'
# convolve_templates_to_res = 80000

load_previous = True

# # Alternative generation of options objects above through configuration files:
routine_options, sep_comp_options, rv_options = load_configuration_files(
    'routine_config.txt', 'sep_config.txt', 'rv_config.txt'
)
rv_options.fit_gui = False

limbd_A = estimate_linear_limbd(wavelength_RV_limit, 2.2, 4700, -0.6, 0.9, loc='Data/tables/atlasco.dat')
limbd_B = estimate_linear_limbd(wavelength_RV_limit, 4.4, 6000, -0.6, 2.0, loc='Data/tables/atlasco.dat')
print('limb A: ', limbd_A)
print('limb B: ', limbd_B)
rv_options.limbd_coef_A = limbd_A
rv_options.limbd_coef_B = limbd_B


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

    # Designate if spectrum should be used for spectral separation
    if filename in use_for_spectral_separation:
        spectral_separation_array_A = np.append(spectral_separation_array_A, i)
    if filename in use_for_spectral_separation:
        spectral_separation_array_B = np.append(spectral_separation_array_B, i)

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

sep_comp_options.weights = [
    136.23105373,109.39835465,134.94906447,162.83396452,155.867572  ,
    127.37994348,142.10365935,144.60878258,151.98552563,107.27208397,
    120.8815536 ,143.48170615,128.79343927,146.03304421, 53.70335185,
    85.46870772, 56.81408276, 45.07682331, 52.11218667, 54.19003598,
    29.53100066, 65.10729606, 67.169636  , 67.7176491 , 61.24181578,
    67.53547216, 51.13198608, 68.45115047, 74.82726776, 97.32111795,
    74.07968682, 42.22511101, 55.09990926, 73.08570312, 60.00733289,
    55.90384602, 73.76855699, 39.3608943 , 32.09984424, 38.13816986,
    54.36027962, 84.45022202, 53.95257176, 62.58881689
]
print(sep_comp_options.weights)

print(sep_comp_options.use_for_spectral_separation_A)
print(sep_comp_options.use_for_spectral_separation_B)
print(rv_options.ignore_at_phase_B)

# # Verify RA and DEC # #
RA, DEC = RA_array[0], DEC_array[0]

# # # Calculate Barycentric RV Corrections # # #
times = Time(date_array, scale='utc', location=observatory_location)
times.format = 'jd'
times.out_subfmt = 'long'
bc_rv_cor, _, _ = get_BC_vel(
    times, ra=RA, dec=DEC, starname=stellar_target, ephemeris='de432s', obsname=observatory_name
)
bc_rv_cor = bc_rv_cor/1000      # from m/s to km/s

# # # Calculate JDUTC to BJDTDB correction # # #
bjdtdb, _, _ = utc_tdb.JDUTC_to_BJDTDB(times, ra=RA, dec=DEC, starname=stellar_target, obsname=observatory_name)


# # Load template spectrum # #
wavelength_template_A, flux_template_A = spf.load_template_spectrum(template_spectrum_path_A)
flux_template_A = flux_template_A[0, :]     # continuum normalized spectrum only
wavelength_template_B, flux_template_B = spf.load_template_spectrum(template_spectrum_path_B)
flux_template_B = flux_template_B[0, :]
_, (flux_template_A, flux_template_B) = spf.resample_multiple_spectra(
    delta_v, (wavelength_template_A, flux_template_A), (wavelength_template_B, flux_template_B),
    wavelength_template=wavelength
)


# # Perform barycentric corrections # #
for i in range(0, flux_collection[0, :].size):
    flux_collection[:, i] = ssr.shift_spectrum(
        flux_collection[:, i], bc_rv_cor[i]-system_RV_estimate, delta_v
    )

# # Limit data-set to specified area (wavelength_RV_limit) # #
wavelength, flux_unbuffered_list, wavelength_buffered, flux_buffered_list, buffer_mask = \
    spf.limit_wavelength_interval_multiple_spectra(
        wavelength_RV_limit, wavelength, flux_collection, flux_template_A, flux_template_B,
        buffer_size=wavelength_buffer_size, even_length=True
    )
[flux_collection, flux_template_A, flux_template_B] = flux_unbuffered_list
[flux_collection_buffered, flux_template_A_buffered, flux_template_B_buffered] = \
    flux_buffered_list

# # Calculate broadening function RVs to use as initial guesses # #
# RV_guesses_A, _ = cRV.radial_velocities_of_multiple_spectra(
#    1-flux_collection, flux_template_A, rv_options, number_of_parallel_jobs=4,
#    plot=False, fit_two_components=False
# )
RV_guess_collection = np.empty((len(wavelength_collection_list), 2))

model = np.loadtxt('/home/sinkbaek/PycharmProjects/Seismic-dEBs/Binary_Analysis/JKTEBOP/kic10001167/kepler_pdcsap_22/model.out')
phase = np.mod(bjdtdb-2455028.0987612916, orbital_period_estimate, dtype=np.float64)/orbital_period_estimate
# RV_guess_collection[:, 0] = np.loadtxt('results/4265_5800_rvA.txt')[:, 1]
# RV_guess_collection[:, 1] = np.loadtxt('results/4265_5800_rvB.txt')[:, 1]
RV_guess_collection[:, 0] = np.interp(phase, model[:, 0], model[:, 6]) - system_RV_estimate
RV_guess_collection[:, 1] = np.interp(phase, model[:, 0], model[:, 7]) - system_RV_estimate


########################### SEPARATION ROUTINE CALLS #################################
# # #  Separate component spectra and calculate RVs iteratively for large interval # # #
if True:
    interval_results = ssr.spectral_separation_routine_multiple_intervals(
        wavelength_buffered, wavelength_intervals_full, flux_collection_buffered,
        flux_template_A_buffered,
        flux_template_B_buffered,
        RV_guess_collection,
        routine_options, sep_comp_options, rv_options,
        wavelength_buffer_size, time_values=bjdtdb - (2400000 + 55028.0987612916), period=orbital_period_estimate
    )
# # # Calculate error # # #
for i in range(len(wavelength_intervals)):
    # RV_guess_collection[:, 0] = np.interp(phase, model[:, 0], model[:, 6]) - system_RV_estimate
    # RV_guess_collection[:, 1] = np.interp(phase, model[:, 0], model[:, 7]) - system_RV_estimate
    RV_guess_collection[:, 0] = np.loadtxt('results/4400_5825_rvA.txt')[:, 1]
    RV_guess_collection[:, 1] = np.loadtxt('results/4400_5825_rvB.txt')[:, 1]
    interval_results = ssr.spectral_separation_routine_multiple_intervals(
        wavelength_buffered, [wavelength_intervals[i]], flux_collection_buffered,
        flux_template_A_buffered,
        flux_template_B_buffered,
        RV_guess_collection,
        routine_options, sep_comp_options, rv_options,
        wavelength_buffer_size, time_values=bjdtdb - (2400000 + 55028.0987612916), period=orbital_period_estimate
    )
plt.show(block=True)

# output is saved to results/.
