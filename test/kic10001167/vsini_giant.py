import sys

import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation
from barycorrpy import get_BC_vel, utc_tdb
import matplotlib.pyplot as plt
import matplotlib

try:
    from src.sb2sep import spectrum_processing_functions as spf
    from src.sb2sep.storage_classes import RadialVelocityOptions, SeparateComponentsOptions, RoutineOptions, load_configuration_files, FitParameters
    import src.sb2sep.spectral_separation_routine as ssr
    from src.sb2sep.linear_limbd_coeff_estimate import estimate_linear_limbd
    import src.sb2sep.calculate_radial_velocities as cRV
    from src.sb2sep.broadening_function_svd import BroadeningFunction
except ModuleNotFoundError:
    print('Using installed package sb2sep.')
    from sb2sep import spectrum_processing_functions as spf
    from sb2sep.storage_classes import RadialVelocityOptions, SeparateComponentsOptions, RoutineOptions, load_configuration_files, FitParameters
    import sb2sep.spectral_separation_routine as ssr
    from sb2sep.linear_limbd_coeff_estimate import estimate_linear_limbd
    import sb2sep.calculate_radial_velocities as cRV
    from sb2sep.broadening_function_svd import BroadeningFunction
import warnings

from sb2sep.rotational_broadening_function_fitting import fitting_routine_rotational_broadening_profile

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
    'FIGd010114_step011_merge.fits'
]
folder_science = '/home/sinkbaek/Data/KIC10001167/'
observatory_location = EarthLocation.of_site("Roque de los Muchachos")
observatory_name = "Roque de los Muchachos"
stellar_target = "kic10001167"
wavelength_normalization_limit = (4315, 7200)   # Ångström, limit to data before performing continuum normalization
wavelength_RV_limit = (4400, 7000)             # Ångström, the area used after normalization
wavelength_buffer_size = 2.0                     # Ångström, padding included at ends of spectra. Useful when doing
                                                # wavelength shifts with np.roll()
delta_v = 1.0         # interpolation sampling resolution for spectrum in km/s

system_RV_estimate = -103.40          # to subtract before routine
orbital_period_estimate = 120.39     # for ignoring component B during eclipse

wavelength_intervals = [(4400+i*200, 4400+i*200+200) for i in range(0, 7)]
wavelength_intervals.append((6000, 6200))
wavelength_intervals.append((6600, 6800))

# # Template Spectra # #
template_spectrum_path_A = 'Data/template_spectra/4750_25_m05p00.ms.fits'

load_previous = True

# RV options
limbd_A = estimate_linear_limbd(wavelength_RV_limit, 2.2, 4760, -0.4, 1.46, loc='Data/tables/atlasco.dat')

fit_params = FitParameters(
    4.0, spectral_resolution=67000, velocity_fit_width=100, limbd_coef=limbd_A, smooth_sigma=3.0, bf_velocity_span=300,
    vary_vsini=True, vary_limbd_coef=False, fitting_profile='RotBF', vary_continuum=False, continuum=0.0, gui=False
)
fit_params.vsini_limits = [0.01, 99.0]
print('limb A: ', limbd_A)


##################### PREPARATION BEFORE SEPARATION ROUTINE CALLS #############################
# # Prepare collection lists and arrays # #
flux_collection_list = []
wavelength_collection_list = []
date_array = np.array([])
RA_array = np.array([])
DEC_array = np.array([])
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
_, (flux_template_A, ) = spf.resample_multiple_spectra(
    delta_v, (wavelength_template_A, flux_template_A),
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
        wavelength_RV_limit, wavelength, flux_collection, flux_template_A,
        buffer_size=wavelength_buffer_size, even_length=True
    )
[flux_collection, flux_template_A] = flux_unbuffered_list
[flux_collection_buffered, flux_template_A_buffered] = flux_buffered_list


# Generate guesses for RV
model = np.loadtxt('/home/sinkbaek/PycharmProjects/Seismic-dEBs/Binary_Analysis/JKTEBOP/kic10001167/kepler_pdcsap_21/model.out')
phase = np.mod(bjdtdb-2455028.0991277853, orbital_period_estimate, dtype=np.float64)/orbital_period_estimate
rv_guess = np.empty((len(files_science), ))
rv_guess[:] = np.interp(phase, model[:, 0], model[:, 6]) - system_RV_estimate

# Create wavelength intervals
wl_col, fl_col, tA_col, _, mask_col = ssr._create_wavelength_intervals(
    wavelength_buffered, wavelength_intervals, flux_collection_buffered, flux_template_A_buffered,
    flux_template_A_buffered, wavelength_buffer_size
)


# Perform RV fits
vsini_values = np.empty((rv_guess.size, len(wl_col)))
for k in range(len(wl_col)):
    plt.figure()
    plt.plot(wl_col[k], fl_col[k][:, 0])
    plt.plot(wl_col[k], tA_col[k], 'k--')
    plt.show()

    plt.figure(figsize=(10, 30))
    plt.title(f'{np.min(wl_col[k]):.0f}--{np.max(wl_col[k]):.0f} Å')
    plt.xlabel('Velocity [km/s]')
    plt.yticks([])
    plt.ylim([-1.5*rv_guess.size-0.5, 2.0])
    for i in range(rv_guess.size):
        fit_params.RV = rv_guess[i]
        BF = BroadeningFunction(
            1-fl_col[k][:, i], 1-tA_col[k], fit_params.bf_velocity_span, delta_v,
            smooth_sigma=fit_params.bf_smooth_sigma
        )
        BF.solve()
        BF.smooth()
        fit, model = fitting_routine_rotational_broadening_profile(
            BF.velocity, BF.bf_smooth, fit_params, BF.smooth_sigma, delta_v
        )
        rv = fit.params['radial_velocity_cm'].value
        vsini = fit.params['vsini'].value

        plt.plot(BF.velocity, BF.bf/np.max(BF.bf)-1.5*i)
        plt.plot(BF.velocity, model / np.max(BF.bf) - 1.5*i, 'k--')
        plt.plot([rv, rv], [-1.5*i, -1.5*i+1], 'r--')
        plt.plot([rv_guess[i], rv_guess[i]], [-1.5*i, -1.5*i+1], 'b--')
        vsini_values[i, k] = vsini
    plt.savefig(f'vsini_figures/{np.min(wl_col[k]):.0f}--{np.max(wl_col[k]):.0f}Å.png')
    plt.close()
print(vsini_values)
print('Mean vsini: ', f'{np.mean(vsini_values):.4f} +- {np.std(vsini_values):.4f} km/s')
print()
print('Mean for different intervals:')
for i in range(vsini_values.shape[1]):
    print(f'{np.mean(vsini_values[:, i]):.4f} +- {np.std(vsini_values[:, i]):.4f} km/s')
print()
print('Mean for different spectra:')
for i in range(vsini_values.shape[0]):
    print(f'{np.mean(vsini_values[i, :]):.4f} +- {np.std(vsini_values[i, :]):.4f} km/s')


