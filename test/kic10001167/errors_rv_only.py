import glob
import sys

import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation
import os
from barycorrpy import get_BC_vel, utc_tdb
import scipy.constants as scc
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcol
import matplotlib.collections as mcoll
from numpy.polynomial import Polynomial

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


def colorline(
    x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width

    https://stackoverflow.com/questions/8500700/how-to-plot-a-gradient-color-line-in-matplotlib
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array

    https://stackoverflow.com/questions/8500700/how-to-plot-a-gradient-color-line-in-matplotlib
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


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
    'FIGd260101_step011_merge.fits', 'FIGe040084_step011_merge.fits'
]
asort = np.argsort(files_science)
print(np.array(files_science)[asort])
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

model = np.loadtxt('/home/sinkbaek/PycharmProjects/Seismic-dEBs/Binary_Analysis/JKTEBOP/kic10001167/kepler_pdcsap_16/model.out')
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

telluric_rv = np.loadtxt('tell_rv_tot.txt')
rva_reference = np.loadtxt('results_backup0227/prepared/rvA_extra_points.dat')
rva_reference[:, 1] = rva_reference[:, 1] + telluric_rv
rva_reference[:, 1] += 103.40

# # # # Measure average spectrograph systematics (line shape vs wavelength) using ALL spectra to correct overestimated uncertainty

# Make blue-red colormap
# https://stackoverflow.com/questions/25748183/python-making-color-bar-that-runs-from-red-to-blue
cm1 = mcol.LinearSegmentedColormap.from_list("bluered", ["b", "r"])
run_values = np.linspace(1, len(wavelength_intervals), len(wavelength_intervals))
run_values_2 = np.linspace(1, len(files_science), len(files_science))
print(run_values)
print(run_values_2)
cnorm = mcol.Normalize(vmin=run_values[0], vmax=run_values[-1])
cnorm_2 = mcol.Normalize(vmin=run_values_2[0], vmax=run_values_2[-1])
cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
cpick.set_array([])
cpick2 = cm.ScalarMappable(norm=cnorm, cmap=cm1)
cpick_2 = cm.ScalarMappable(norm=cnorm_2, cmap=cm1)
cpick_2.set_array([])
cpick2_2 = cm.ScalarMappable(norm=cnorm_2, cmap=cm1)

residual_intervals = rvA_intervals - rva_reference[:, 1].reshape((rva_reference.shape[0], 1))
residual_intervals_sorted = residual_intervals[asort, :]
xvs = [np.mean(wavelength_intervals[i]) for i in range(len(wavelength_intervals))]  # np.ones(residual_intervals.shape) * run_values.reshape(1, residual_intervals.shape[1])
xvs = np.repeat(np.array(xvs).reshape(1, len(xvs)), residual_intervals.shape[0], axis=0)
fit = Polynomial.fit(xvs.flatten(), residual_intervals.flatten(), 1)
# fitvals = fit(run_values)
fitvals = fit(xvs[0, :])
fig = plt.figure(figsize=(14, 10))
# plt.plot(run_values, fitvals, 'k--', label='Line fit', linewidth=3)
plt.plot(xvs[0, :], fitvals, 'k--', label='Line fit', linewidth=3)
for i in range(residual_intervals.shape[1]):
    # plt.scatter(np.ones(residual_intervals.shape[0]) * (i+1), residual_intervals_sorted[:, i], marker='D', color=cpick_2.to_rgba(run_values_2))
    plt.scatter(xvs[:, i], residual_intervals_sorted[:, i], marker='D',
                color=cpick_2.to_rgba(run_values_2))
plt.legend()
plt.xlabel('Mean interval wavelength [Å]')
plt.ylabel('Residual [km/s]')
fig.colorbar(cpick2_2, label='Spectrum number (chronological)')
plt.savefig('residuals_intervals_spectra.png', dpi=150)

fig = plt.figure(figsize=(14, 10))
for i in range(residual_intervals.shape[0]):
    plt.scatter(np.ones(residual_intervals.shape[1])*(i+1), residual_intervals_sorted[i, :], marker='D', color=cpick.to_rgba(run_values))
plt.xlabel('Spectrum number (chronological)')
plt.ylabel('Residual [km/s]')
fig.colorbar(cpick2, label='Interval number (from bluest to reddest)')
plt.savefig('residuals_spectra_intervals.png', dpi=150)

fig = plt.figure(figsize=(14, 10))
residuals_corrected = residual_intervals - fitvals.reshape(1, fitvals.size)
residuals_corrected = residuals_corrected[asort, :]
for i in range(residual_intervals.shape[0]):
    plt.scatter(np.ones(residual_intervals.shape[1])*(i+1), residuals_corrected[i, :], marker='D', color=cpick.to_rgba(run_values))
plt.xlabel('Spectrum number (chronological)')
plt.ylabel('Residual (corrected) [km/s]')
fig.colorbar(cpick2, label='Interval number (from bluest to reddest)')
plt.show()

rvA_intervals = rvA_intervals - fitvals.reshape(1, fitvals.size)
rvB_intervals = rvB_intervals - fitvals.reshape(1, fitvals.size)


# # # # Measure ThAr uncertainties
if False:
    from lyskryds.krydsorden import getCCF, getRV
    from astropy.stats import biweight_location

    thars_ = glob.glob('/media/sinkbaek/NOT_DATA/fiestool/Data/output/kic10001167/*thar.fits')
    thars__ = [x.replace('008_thar', '012_merge') for x in thars_]
    thars = [x.replace('007_thar', '011_merge') for x in thars__]
    thars.append('/media/sinkbaek/NOT_DATA/fiestool/Data/output/kic10001167/FIGe040083_step011_merge.fits')     # THIS WON'T GO WELL IF YOU GET NEWER DATA THAN THIS
    print(thars)
    wl_list = []
    fl_list = []
    expt_list = []
    for fn in thars:
        wl, fl, _, _, _, expt = spf.load_program_spectrum(fn)
        wl_list.append(wl)
        fl_list.append(fl)
        expt_list.append(expt)
    idx_factor4 = [30, 39, 32, 35, 38, 37, 34, 31, 40, 41, 42, 43, 44]
    dv = 1.0
    wl_thar, fls_thar = spf.resample_to_equal_velocity_steps(wl_list, dv, fl_list, wavelength_a=4048, wavelength_b=6850)
    plt.figure()
    fls_thar[:, idx_factor4] = fls_thar[:, idx_factor4] / 4
    # template_median = np.median(fls_thar, axis=1)
    template_thar = biweight_location(fls_thar, axis=1)
    plt.plot(wl_thar.reshape(wl_thar.size, 1), fls_thar)
    # plt.plot(wl_thar, template_median, 'k-', linewidth=3)
    plt.plot(wl_thar, template_thar, 'k--', linewidth=3)
    for i in range(fls_thar.shape[1]):
        plt.figure()
        plt.title(f'{i}')
        plt.plot(wl_thar, fls_thar[:, i])
        plt.plot(wl_thar, template_thar, 'k--')
    plt.show()

    drifts_combined = np.zeros(len(thars))

    for k in range(2):
        print(f'k = {k} of {1}')
        plt.figure()
        drifts = []
        errs_thar = []
        template_thar = biweight_location(fls_thar, axis=1)
        for i in range(fls_thar.shape[1]):
            print(f'i = {i} of {fls_thar.shape[1]-1}')
            vel, ccf = getCCF(fls_thar[:, i], template_thar, rvr=31)
            vel = vel * dv
            rv, err_ = getRV(vel, ccf, poly=False, new=True, zucker=False)
            label = f'{i}  {1000 * rv:.2f} m/s'
            plt.plot(vel * 1000, ccf / np.max(ccf) - i, label=label)
            plt.plot([rv * 1000, rv * 1000], [-i, 1 - i], 'k--')
            drifts.append(rv)
            errs_thar.append(err_)
        drifts_combined += drifts
        plt.xlabel('Velocity [m/s]')
        plt.legend(fontsize=10)
        for i in range(fls_thar.shape[1]):
            fls_thar[:, i] = ssr.shift_spectrum(fls_thar[:, i], -drifts[i], dv)
    print('Drifts')
    print(drifts_combined * 1000)
    print('Deviation from mean')
    print(np.abs(drifts_combined - np.mean(drifts_combined)) * 1000)
    print('Errs')
    errs_thar = np.array(errs_thar)
    print(errs_thar * 1000)
    drift_deviation_thar = np.abs(drifts_combined - np.mean(drifts_combined))

    asort_files = np.argsort(files_science)
    np.savetxt('thar_drifts.dat', drifts_combined)

    plt.show()



# Measure uncertainties
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

idx_b = [
    1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
    36,  # 37, 38, 39, 40
    41, 42, 43, 44
]

print(f'{np.mean(errs_A):.3f}    {np.mean(errs_A_simple):.3f}   {np.mean(errs_combined_A):.3f}')
print(f'{np.mean(errs_B[idx_b]):.3f}    {np.mean(errs_B_simple[idx_b]):.3f}   {np.mean(errs_combined_B[idx_b]):.3f}')
print('errs_combined_A')
print(errs_combined_A)
print('errs_combined_B')
print(errs_combined_B[idx_b])
np.savetxt('errs_A_corrected.txt', errs_A_simple)
np.savetxt('errs_B_corrected.txt', errs_B[idx_b])

rva_saved = np.loadtxt('results_backup0227/prepared/rvA_extra_points.dat')
rvb_saved = np.loadtxt('results_backup0227/prepared/rvB_extra_points.dat')

asort_files = np.argsort(files_science)

errors_thar = np.loadtxt('thar_uncertainties_sorted.dat')
errors_telluric = np.loadtxt('telluric_cross_validation_std.dat')[asort_files]

errs_A_simple[asort_files] = np.sqrt(errs_A_simple[asort_files]**2 + errors_thar**2 + errors_telluric**2)
errs_B[asort_files] = np.sqrt(errs_B[asort_files]**2 + errors_thar**2 + errors_telluric**2)

print('errs_with_thar_A')
print(errs_A_simple)
print('errs_with_thar_B')
print(errs_B[idx_b])

rva_saved[:, 2] = errs_A_simple
rvb_saved[:, 2] = errs_B[idx_b]
np.savetxt('rvA_prepared_err_thar_telluric.dat', rva_saved)
np.savetxt('rvB_prepared_err_thar_telluric.dat', rvb_saved)

jitter_term = 0.0914
rva_jitter = np.copy(rva_saved)
rva_jitter[:, 2] = np.sqrt(rva_jitter[:, 2]**2 + jitter_term**2)
rvb_jitter = np.copy(rvb_saved)
rvb_jitter[:, 2] = np.sqrt(rvb_jitter[:, 2]**2 + jitter_term**2)
np.savetxt('rvA_with_jitter.dat', rva_jitter)
np.savetxt('rvB_with_jitter.dat', rvb_jitter)

plt.show()
