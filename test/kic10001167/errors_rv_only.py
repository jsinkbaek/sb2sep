import sys

import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation
import os
from barycorrpy import get_BC_vel, utc_tdb
import scipy.constants as scc
import matplotlib.pyplot as plt
import matplotlib

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
    'FIGc030078_step011_merge.fits', 'FIGc110124_step011_merge.fits'
]
len_old = 31
folder_science = '/home/sinkbaek/Data/KIC10001167/'

wavelength_intervals = [                        # Intervals used for error calculation
    (4400, 4690), (4690, 4980), (4980, 5270), (5270, 5560), (5560, 5850),
    (5985, 6275), (6565, 6840)
]

use_for_spectral_separation = np.array([
    'FIDg060105_step011_merge.fits', 'FIDh200065_step011_merge.fits',
    'FIEh190105_step011_merge.fits', 'FIDh150097_step011_merge.fits',
    'FIBl050130_step011_merge.fits', 'FIEh020096_step012_merge.fits',
    'FIBk050060_step011_merge.fits', 'FIDg070066_step011_merge.fits',
    'FIDg080034_step011_merge.fits', 'FIBl080066_step011_merge.fits',
    'FIDg050102_step011_merge.fits', 'FIDh170096_step011_merge.fits',
    'FIDh160097_step011_merge.fits', 'FIBi240077_step011_merge.fits',
    'FIBi230039_step011_merge.fits', 'FIBi240074_step011_merge.fits',
    'FIBk060008_step011_merge.fits', 'FIFj100096_step011_merge.fits',
    'FIEh060100_step012_merge.fits', 'FIBk030040_step011_merge.fits',
    'FIBk140072_step011_merge.fits', 'FIDh100076_step011_merge.fits',
    'FIBk040034_step011_merge.fits', 'FIEf140066_step011_merge.fits',
    'FIBj150077_step011_merge.fits'
])
observatory_location = EarthLocation.of_site("Roque de los Muchachos")
observatory_name = "Roque de los Muchachos"
stellar_target = "kic10001167"
wavelength_normalization_limit = (4315, 7200)   # Ångström, limit to data before performing continuum normalization
wavelength_RV_limit = (4400, 7000)             # Ångström, the area used after normalization
wavelength_buffer_size = 4.0                     # Ångström, padding included at ends of spectra. Useful when doing
                                                # wavelength shifts with np.roll()
delta_v = 1.0         # interpolation sampling resolution for spectrum in km/s

system_RV_estimate = -103.40          # to subtract before routine
orbital_period_estimate = 120.39     # for ignoring component B during eclipse

# # Template Spectra # #
template_spectrum_path_A = 'Data/template_spectra/4750_25_m05p00.ms.fits'
template_spectrum_path_B = 'Data/template_spectra/6250_45_m05p00.ms.fits'
# convolve_templates_to_res = 80000

load_previous = True

# # Alternative generation of options objects above through configuration files:
routine_options, sep_comp_options, rv_options = load_configuration_files(
    'routine_config.txt', 'sep_config.txt', 'rv_config.txt'
)
rv_options.fit_gui = False

limbd_A = estimate_linear_limbd(wavelength_RV_limit, 2.2, 4700, -0.7, 0.9, loc='Data/tables/atlasco.dat')
limbd_B = estimate_linear_limbd(wavelength_RV_limit, 4.4, 5700, -0.7, 2.0, loc='Data/tables/atlasco.dat')
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

sep_comp_options.weights = np.sqrt(expt_array) / np.sqrt(np.max(expt_array))
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


RV_collection = np.empty((len_old, 2))

RV_collection[:, 0] = np.loadtxt('results_backup0227/4400_5825_rvA.txt')[:, 1]
RV_collection[:, 1] = np.loadtxt('results_backup0227/4400_5825_rvB.txt')[:, 1]

separated_flux_A, separated_flux_B = ssr.separate_component_spectra(
    flux_collection_buffered[:, :len_old], RV_collection[:, 0], RV_collection[:, 1], sep_comp_options
)
rv_options.evaluate_spectra_A = None
rv_options.evaluate_spectra_B = None

model = np.loadtxt('/home/sinkbaek/PycharmProjects/Seismic-dEBs/Binary_Analysis/JKTEBOP/kic10001167/kepler_pdcsap_12/model.out')
phase = np.mod(bjdtdb-2455028.0991277853, orbital_period_estimate, dtype=np.float64)/orbital_period_estimate
rv_guess_new = np.empty((len(files_science), 2))
rv_guess_new[:, 0] = np.interp(phase, model[:, 0], model[:, 6]) - system_RV_estimate
rv_guess_new[:, 1] = np.interp(phase, model[:, 0], model[:, 7]) - system_RV_estimate

wl_col, fl_col, tA_col, tB_col, sepA_col, sepB_col, mask_col = ssr._create_wavelength_intervals(
    wavelength_buffered, wavelength_intervals, flux_collection_buffered, flux_template_A_buffered,
    flux_template_B_buffered, wavelength_buffer_size, separated_flux_A, separated_flux_B
)
rvA_intervals = np.empty((len(files_science), len(wl_col)))
rvB_intervals = np.empty((len(files_science), len(wl_col)))

rv_options.evaluate_spectra_A = np.linspace(0, len(files_science)-1, len(files_science), dtype=int)
rv_options.evaluate_spectra_B = np.linspace(0, len(files_science)-1, len(files_science), dtype=int)
if False:
    for i in range(len(wl_col)):
        f1_ax1, f1_ax2, f1_ax3, f3_ax1, f4_ax1 = ssr._initialize_ssr_plots()
        rv_a, rv_b, (bf_res_A, bf_res_B) = ssr.recalculate_RVs(
            fl_col[i], sepA_col[i], sepB_col[i], rv_guess_new[:, 0], rv_guess_new[:, 1], tA_col[i], tB_col[i],
            mask_col[i],
            rv_options, time_values=bjdtdb - (2400000 + 55028.099127785), period=orbital_period_estimate,
            plot_ax_A=f3_ax1, plot_ax_B=f4_ax1
        )
        ssr._plot_ssr_iteration(
            f1_ax1, f1_ax2, f1_ax3, sepA_col[i], sepB_col[i], wl_col[i], tA_col[i], tB_col[i], rv_a, rv_b,
            bjdtdb - (2400000 + 55028.099127785), orbital_period_estimate, mask_col[i], sep_comp_options.rv_lower_limit,
            sep_comp_options.rv_proximity_limit
        )
        rvA_intervals[:, i] = rv_a
        rvB_intervals[:, i] = rv_b

    np.savetxt('rvA_intervals.txt', rvA_intervals)
    np.savetxt('rvB_intervals.txt', rvB_intervals)
else:
    rvA_intervals = np.loadtxt('rvA_intervals.txt')
    rvB_intervals = np.loadtxt('rvB_intervals.txt')


from astropy.stats import biweight_midvariance
def err_biw_midv_clip(rvs_, threshold=4, n_iter=10):
    ite = 0
    mask = np.ones(rvs_.shape, dtype=bool)
    while True:
        rvs_clipped = np.copy(rvs_)
        rvs_clipped[~mask] = np.nan
        biw_midv = biweight_midvariance(rvs_clipped, ignore_nan=True, axis=1)
        scale = np.sqrt(biw_midv)
        dev = np.abs(rvs_ - np.median(rvs_, axis=1).reshape(rvs_.shape[0], 1))
        mask = dev < threshold * scale.reshape(rvs_.shape[0], 1)
        if np.all((dev < threshold * scale.reshape(rvs_.shape[0], 1)) == mask) or ite >= n_iter:
            size_arr = np.ones(rvs_.shape)
            size_arr[~mask] = np.nan
            sizes = np.nansum(size_arr, axis=1)
            errs = scale / np.sqrt(sizes)
            return errs
        else:
            mask = dev < threshold * scale.reshape(rvs_.shape[0], 1)
            ite += 1


errs_A = err_biw_midv_clip(rvA_intervals)
errs_B = err_biw_midv_clip(rvB_intervals)
print('errs_A')
print(errs_A)
print('errs_B')
print(errs_B)
print('errs_A_simple')
errs_A_simple = np.std(rvA_intervals, axis=1) / np.sqrt(rvA_intervals.shape[1])
errs_B_simple = np.std(rvB_intervals, axis=1) / np.sqrt(rvB_intervals.shape[1])
print(errs_A_simple)
print('errs_B_simple')
print(errs_B_simple)
print()

errs_combined_A = np.copy(errs_A)
mask_A = errs_A-errs_A_simple > 0
errs_combined_A[mask_A] = errs_A_simple[mask_A]

errs_combined_B = np.copy(errs_B)
mask_B = errs_B-errs_B_simple > 0
errs_combined_B[mask_B] = errs_B_simple[mask_B]

idx_b = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

print(f'{np.mean(errs_A):.3f}    {np.mean(errs_A_simple):.3f}   {np.mean(errs_combined_A):.3f}')
print(f'{np.mean(errs_B[idx_b]):.3f}    {np.mean(errs_B_simple[idx_b]):.3f}   {np.mean(errs_combined_B[idx_b]):.3f}')
print('errs_combined_A')
print(errs_combined_A)
print('errs_combined_B')
print(errs_combined_B[idx_b])
np.savetxt('errs_A.txt', errs_A_simple)
np.savetxt('errs_B.txt', errs_B[idx_b])

rva_saved = np.loadtxt('results_backup0227/prepared/rvA_extra_points.dat')
rvb_saved = np.loadtxt('results_backup0227/prepared/rvB_extra_points.dat')
rva_saved[:, 2] = errs_A_simple
rvb_saved[:, 2] = errs_B[idx_b]
np.savetxt('rvA_prepared.dat', rva_saved)
np.savetxt('rvB_prepared.dat', rvb_saved)

plt.show()
