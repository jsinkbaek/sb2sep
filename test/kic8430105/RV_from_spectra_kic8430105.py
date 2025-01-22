import sys

import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation
import os
from barycorrpy import get_BC_vel, utc_tdb
import scipy.constants as scc
import matplotlib.pyplot as plt
import matplotlib

print('Using installed package sb2sep.')
from sb2sep import spectrum_processing_functions as spf
from sb2sep.storage_classes import RadialVelocityOptions, SeparateComponentsOptions, RoutineOptions, load_configuration_files
import sb2sep.spectral_separation_routine as ssr
from sb2sep.linear_limbd_coeff_estimate import estimate_linear_limbd
import sb2sep.calculate_radial_velocities as cRV
import warnings

matplotlib.use('TkAgg')

matplotlib.rcParams.update({'font.size': 25})


##################### VARIABLES AND OPTIONS FOR SCRIPT ##############################3
warnings.filterwarnings("ignore", category=UserWarning)
# plt.ion()
plt.ioff()
data_path = 'Data/'

observatory_location = EarthLocation.of_site("Roque de los Muchachos")
observatory_name = "Roque de los Muchachos"
stellar_target = "kic8430105"
wavelength_normalization_limit = (4000, 7000)   # Ångström, limit to data before performing continuum normalization
wavelength_RV_limit = (4000, 7000)              # Ångström, the area used after normalization
wavelength_buffer_size = 4.0                     # Ångström, padding included at ends of spectra. Useful when doing
                                                # wavelength shifts with np.roll()
wavelength_intervals_full = [(4000, 5800)]      # Ångström, the actual interval used.
wavelength_intervals = [                        # Intervals used for error calculation
    (4000, 4265), (4265, 4500), (4500, 4765), (4765, 5030), (5030, 5295), (5295, 5560), (5560, 5825), (5985, 6250),
    (6575, 6840)
]


file_exclude_list = []
# use_for_spectral_separation_A = [       # converted to index values during file load later (spectral_separation_array_A)
#     'FIDi080098_step011_merge.fits', 'FIDi090065_step011_merge.fits',
#     'FIBj150080_step011_merge.fits', 'FIDi130112_step011_merge.fits', 'FIBk030043_step011_merge.fits',
#     'FIBk050063_step011_merge.fits',
#     'FIBk140069_step011_merge.fits', 'FIDh160100_step011_merge.fits',
#     'FIBi230047_step011_merge.fits', 'FIBi240080_step011_merge.fits', "FIEh020092_step012_merge.fits"
#     ]
# not used: 'FIBk230070_step011_merge.fits', 'FIBk060011_step011_merge.fits',
# use_for_spectral_separation_B = [
#     'FIDi080098_step011_merge.fits', 'FIDi090065_step011_merge.fits',
#     'FIBj150080_step011_merge.fits', 'FIDi130112_step011_merge.fits', 'FIBk030043_step011_merge.fits',
#     'FIBk050063_step011_merge.fits',
#     'FIBk140069_step011_merge.fits', 'FIDh160100_step011_merge.fits',
#     'FIBi230047_step011_merge.fits', 'FIBi240080_step011_merge.fits', "FIEh020092_step012_merge.fits"
#    ]

delta_v = 1.0         # interpolation sampling resolution for spectrum in km/s

mass_A_estimate = 1.31  # both used to estimate RV_B
mass_B_estimate = 0.83

system_RV_estimate = 12.61          # to subtract before routine
orbital_period_estimate = 63.327     # for ignoring component B during eclipse

# # Initial fit parameters for rotational broadening function fit # #
limbd_A = estimate_linear_limbd(wavelength_RV_limit, 2.8, 4990, -0.45, 0.9, loc='Data/tables/atlasco.dat')
limbd_B = estimate_linear_limbd(wavelength_RV_limit, 4.5, 5700, -0.45, 2.0, loc='Data/tables/atlasco.dat')

# # Template Spectra # #
template_spectrum_path_A = 'Data/template_spectra/5000_30_m05p00.ms.fits'
template_spectrum_path_B = 'Data/template_spectra/5500_45_m05p00.ms.fits'

load_previous = True

# # Generate rv_options # #
routine_options, sep_comp_options, rv_options = load_configuration_files(
    'routine_config.txt', 'sep_config.txt', 'rv_config.txt'
)
# rv_options.fit_gui = True


##################### PREPARATION BEFORE SEPARATION ROUTINE CALLS #############################
# # Prepare collection lists and arrays # #
flux_collection_list = []
wavelength_collection_list = []
date_array = np.array([])
RA_array = np.array([])
DEC_array = np.array([])
spectral_separation_array_A = np.array([])
spectral_separation_array_B = np.array([])


# # # Load fits files, collect and normalize data # # #
i = 0
for filename in os.listdir(data_path):
    if 'merge.fits' in filename and '.lowSN' not in filename and filename not in file_exclude_list:
        # Load observation
        wavelength, flux, date, ra, dec = spf.load_program_spectrum(data_path+filename)
        date_array = np.append(date_array, date)
        RA_array = np.append(RA_array, ra*15.0)     # converts unit
        DEC_array = np.append(DEC_array, dec)

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

        # Designate if spectrum should be used for spectral separation
        # if filename in use_for_spectral_separation_A:
        #    spectral_separation_array_A = np.append(spectral_separation_array_A, i)
        # if filename in use_for_spectral_separation_B:
        #    spectral_separation_array_B = np.append(spectral_separation_array_B, i)

        # Append to collection
        wavelength_collection_list.append(wavelength)
        flux_collection_list.append(flux)
        i += 1

# sep_comp_options.use_for_spectral_separation_A = spectral_separation_array_A        # see notes earlier
# sep_comp_options.use_for_spectral_separation_B = spectral_separation_array_B

print(sep_comp_options.use_for_spectral_separation_A)
print(sep_comp_options.use_for_spectral_separation_B)
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

# # Limit templates
mask = (wavelength_template_A > wavelength_RV_limit[0]-100) & (wavelength_template_A < wavelength_RV_limit[1]+100)
wavelength_template_A, flux_template_A = wavelength_template_A[mask], flux_template_A[mask]
mask = (wavelength_template_B > wavelength_RV_limit[0]-100) & (wavelength_template_B < wavelength_RV_limit[1]+100)
wavelength_template_B, flux_template_B = wavelength_template_B[mask], flux_template_B[mask]

# # Resample to same wavelength grid, equi-spaced in velocity space # #
wavelength, (flux_collection, flux_template_A, flux_template_B) = spf.resample_multiple_spectra(
    delta_v, (wavelength_collection_list, flux_collection_list), (wavelength_template_A, flux_template_A),
    (wavelength_template_B, flux_template_B)
)

# # Perform barycentric corrections # #
for i in range(0, flux_collection[0, :].size):
    flux_collection[:, i] = ssr.shift_wavelength_spectrum(
        wavelength, flux_collection[:, i], bc_rv_cor[i]-system_RV_estimate, "makima", 1.0
    )

# # Shorten spectra if uneven # #
wavelength, [flux_collection, flux_template_A, flux_template_B] = \
    spf.make_spectrum_even(wavelength, [flux_collection, flux_template_A, flux_template_B])


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
# RV_guesses_A, extra_stuff = cRV.radial_velocities_of_multiple_spectra(
#    1-flux_collection, 1-flux_template_A, rv_options, number_of_parallel_jobs=2,
#    plot=False, fit_two_components=False
# )
# bf_velocity, bf_vals, bf_vals_smooth, model_vals = extra_stuff
# print(bf_velocity.shape, bf_vals.shape)
# for i in range(bf_vals_smooth.shape[1]):
#     plt.figure()
#     plt.plot(bf_velocity, bf_vals_smooth[:, i], "k-")
#     plt.plot(bf_velocity, model_vals[:, i], "r-")
# plt.show(block=True)

RV_guess_collection = np.empty((len(wavelength_collection_list), 2))
# RV_guess_collection[:, 0] = RV_guesses_A
# RV_guesses_B = -RV_guesses_A * (mass_A_estimate / mass_B_estimate)
# RV_guess_collection[:, 1] = RV_guesses_B

model = np.loadtxt('ebmodel.out')
phase = np.mod(bjdtdb-2454998.2336069366, orbital_period_estimate, dtype=np.float64)/orbital_period_estimate
RV_guess_collection[:, 0] = np.interp(phase, model[:, 0], model[:, 6])-system_RV_estimate
RV_guess_collection[:, 1] = np.interp(phase, model[:, 0], model[:, 7])-system_RV_estimate


########################### SEPARATION ROUTINE CALLS #################################
# # #  Separate component spectra and calculate RVs iteratively for large interval # # #
interval_results = ssr.spectral_separation_routine_multiple_intervals(
    wavelength_buffered, wavelength_intervals_full, flux_collection_buffered,
    flux_template_A_buffered,
    flux_template_B_buffered,
    RV_guess_collection,
    routine_options, sep_comp_options, rv_options,
    wavelength_buffer_size, time_values=bjdtdb-(2400000+54976.6348), period=orbital_period_estimate
)
plt.show(block=True)
# # # Calculate error by splitting into multiple independent intervals # # #
interval_results = ssr.spectral_separation_routine_multiple_intervals(
     wavelength_buffered, wavelength_intervals, flux_collection_buffered,
     flux_template_A_buffered,
     flux_template_B_buffered,
     RV_guess_collection,
     routine_options, sep_comp_options, rv_options,
     wavelength_buffer_size, time_values=bjdtdb-(2400000+54976.6348), period=orbital_period_estimate
)

# output is saved to Results/.
