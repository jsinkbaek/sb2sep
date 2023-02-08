"""
This collection of functions is used for general work on wavelength reduced échelle spectra.
"""
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import scipy.constants as scc
from scipy.interpolate import interp1d
from numpy.polynomial import Polynomial
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection, HandlerLine2D
import matplotlib
from scipy.ndimage import maximum_filter


def load_template_spectrum(template_spectrum_path: str):
    with fits.open(template_spectrum_path) as hdul:
        hdr = hdul[0].header
        flux = hdul[0].data
        wl0, delta_wl, naxis = hdr['CRVAL1'], hdr['CDELT1'], hdr['NAXIS1']
    wavelength = np.linspace(wl0, wl0+delta_wl*naxis, naxis)
    return wavelength, flux


def load_phoenix_template(template_spectrum_path: str, wave_ref_path: str):
    with fits.open(template_spectrum_path) as hdul:
        flux = hdul[0].data
    with fits.open(wave_ref_path) as hdul:
        wavelength = hdul[0].data
    size = int(wavelength.size / 10000)
    continuum = maximum_filter(flux, size=size)
    norm_flux = flux / continuum
    plt.figure()
    plt.plot(wavelength, norm_flux)
    plt.xlim([3000, 9000])
    plt.show(block=True)
    return wavelength, flux, norm_flux


def load_program_spectrum(program_spectrum_path: str):
    with fits.open(program_spectrum_path) as hdul:
        hdr = hdul[0].header
        flux = hdul[0].data
        wl0, delta_wl = hdr['CRVAL1'], hdr['CDELT1']
        RA, DEC = hdr['OBJRA'], hdr['OBJDEC']
        date = hdr['DATE-AVG']
        exptime = hdr['EXPTIME']
    wavelength = np.linspace(wl0, wl0 + delta_wl * flux.size, flux.size)
    return wavelength, flux, date, RA, DEC, exptime


def _resample_multiple_spectra(wavelength_template: np.ndarray, wavelength_list: list, flux_list: list):

    flux_resampled_collection = np.empty(shape=(wavelength_template.size, len(flux_list)))
    for i in range(0, len(flux_list)):
        flux_resampled_collection[:, i] = interp1d(wavelength_list[i], flux_list[i], kind='linear')(wavelength_template)
        # flux_resampled_collection[:, i] = np.interp(wavelength_template, wavelength_list[i], flux_list[i])

    return flux_resampled_collection


def _create_wavelength_template(
        wavelengths: list or np.ndarray, delta_v:float,  wavelength_a:float = None, wavelength_b:float = None,
        even_length=True
):
    if isinstance(wavelengths, list):
        if wavelength_a is None:
            wavelength_a = np.max([np.min(x) for x in wavelengths])
        if wavelength_b is None:
            wavelength_b = np.min([np.max(x) for x in wavelengths])
    else:
        if wavelength_a is None:
            wavelength_a = np.min(wavelengths)
        if wavelength_b is None:
            wavelength_b = np.max(wavelengths)

    speed_of_light = scc.c / 1000  # in km/s
    step_amnt = int(np.log10(wavelength_b / wavelength_a) / np.log10(1.0 + delta_v / speed_of_light))
    wavelength_template = wavelength_a * (1.0 + delta_v / speed_of_light) ** (np.linspace(1, step_amnt, step_amnt))

    if even_length is True and np.mod(wavelength_template.size, 2) != 0.0:
        wavelength_template = wavelength_template[:-1]

    return wavelength_template


def resample_to_equal_velocity_steps(
        wavelength: np.ndarray or list, delta_v: float, flux=None, wavelength_template: np.ndarray = None,
        wavelength_a=None, wavelength_b=None, resampled_len_even=True
):
    """
    Multi-use function that can be used to either: Create a new wavelength grid equi-distant in vel
    (without interpolation), create grid and resample flux to it, create mutual grid and resample flux for multiple
    spectra simultanously. It can also resample one (or more) spectra to a provided wavelength grid.
    See interpolate_to_equal_velocity_steps() for simpler illustration of what this function does.
    :param wavelength:              either a list of np.ndarray's, or a np.ndarray
    :param delta_v:                 desired spectrum resolution in velocity space in km/s
    :param flux:                    either a list of np.ndarray's, or a np.ndarray. Set to None if only
                                    wavelength_template should be returned.
    :param wavelength_template:    a np.ndarray with a provided resampled grid. Set to None if one should be calculated
    :param wavelength_a:            float, start of desired grid. If None, calculates from provided spectra
    :param wavelength_b:            float, end of desired grid. If None, calculates from provided spectra.
    :param resampled_len_even:      bool switch, sets if resampled grid should be kept on an even length
    :return:    either wavelength_template,
                      (wavelength_template, flux_resampled_collection),
                or    (wavelength_template, flux_resampled)
    """
    if wavelength_template is None:
        wavelength_template = _create_wavelength_template(
           wavelength, delta_v, wavelength_a, wavelength_b, resampled_len_even
        )
    elif resampled_len_even and np.mod(wavelength_template.size, 2) != 0.0:
        wavelength_template = wavelength_template[:-1]

    if flux is not None:
        if isinstance(flux, list) and isinstance(wavelength, list):
            flux_resampled_collection = _resample_multiple_spectra(wavelength_template, wavelength, flux)
            return wavelength_template, flux_resampled_collection
        elif isinstance(flux, np.ndarray) and isinstance(wavelength, np.ndarray):
            flux_resampled = interp1d(wavelength, flux, kind='linear')(wavelength_template)
            # flux_resampled = np.interp(wavelength_template, wavelength, flux)
            return wavelength_template, flux_resampled
        else:
            raise ValueError("flux and wavelength is neither a list (of arrays), an array, or None "
                             "(they must be the same type).")
    else:
        return wavelength_template


def resample_multiple_spectra(
        delta_v: float, *spectra, wavelength_template: np.ndarray = None, wavelength_a=None, wavelength_b=None,
        resampled_len_even=True
):
    """
    Resamples multiple spectra. Expects *spectra (traditionally *args in python) to be a collection of tuples with
    (wavelength, flux) for each. It is allowed to include multiple spectra in one of the tuples if done with
    ([wl_1, wl_2, ...], [flux_1, flux_2, ...]) since this function calls resample_to_equal_velocity_steps(). As such
    this function is purely a wrapper for convenience.
    :return: wavelength, (flux_1, flux_2, flux_3, ...)
    """
    flux_list = []
    for (wl, fl) in spectra:
        wavelength_template, flux_ = resample_to_equal_velocity_steps(
            wl, delta_v, fl, wavelength_template, wavelength_a, wavelength_b, resampled_len_even
        )
        flux_list.append(flux_)
    return wavelength_template, tuple(flux_list)


def interpolate_to_equal_velocity_steps(wavelength_collector_list: list, flux_collector_list: list, delta_v: float):
    """
    Resamples a set of spectra to the same wavelength grid equi-spaced in velocity map.
    :param wavelength_collector_list:   list of arrays, one array for each spectrum
    :param flux_collector_list:         list of arrays, one array for each spectrum
    :param delta_v:                     interpolation resolution for spectrum in km/s
    :return: wavelength, flux_collection_array
    """
    speed_of_light = scc.c / 1000       # in km/s

    # # Create unified wavelength grid equispaced in velocity # #
    wavelength_a = np.max([x[0] for x in wavelength_collector_list])
    wavelength_b = np.min([x[-1] for x in wavelength_collector_list])
    step_amnt = np.log10(wavelength_b / wavelength_a) / np.log10(1.0 + delta_v / speed_of_light) + 1
    wavelength = wavelength_a * (1.0 + delta_v / speed_of_light) ** (np.linspace(1, step_amnt, step_amnt))

    # # Interpolate to unified wavelength grid # #
    flux_collection_array = np.empty(shape=(wavelength.size, len(flux_collector_list)))
    for i in range(0, len(flux_collector_list)):
        flux_interpolator = interp1d(wavelength_collector_list[i], flux_collector_list[i], kind='linear')
        flux_collection_array[:, i] = flux_interpolator(wavelength)

    return wavelength, flux_collection_array


def _create_buffer_mask(wavelength: np.ndarray, wavelength_limits: tuple, buffer_size: float):
    buffer_lower = (wavelength > wavelength_limits[0] - buffer_size) & (wavelength < wavelength_limits[0])
    buffer_upper = (wavelength < wavelength_limits[1] - buffer_size) & (wavelength > wavelength_limits[1])
    buffer_mask = buffer_lower | buffer_upper
    return buffer_mask


def limit_wavelength_interval(
        wavelength_limits: tuple, wavelength: np.ndarray, flux: np.ndarray, buffer_size=None, buffer_mask=None,
        even_length=False
):
    selection_mask = (wavelength > wavelength_limits[0]) & (wavelength < wavelength_limits[1])
    selection_mask_buffered = np.zeros(wavelength.shape, dtype=bool)
    if buffer_size is not None and buffer_mask is None:
        buffer_mask = _create_buffer_mask(wavelength, wavelength_limits, buffer_size)
    if buffer_mask is not None:
        selection_mask_buffered = selection_mask | buffer_mask

    wavelength_new = wavelength[selection_mask]
    if flux.ndim == 1:
        flux_new = flux[selection_mask]
    elif flux.ndim == 2:
        flux_new = flux[selection_mask, :]
    else:
        raise ValueError('flux has wrong number of dimensions. Should be either 1 or 2.')

    if buffer_mask is not None:
        wavelength_buffered = wavelength[selection_mask_buffered]
        if flux.ndim == 1:
            flux_buffered = flux[selection_mask_buffered]
        elif flux.ndim == 2:
            flux_buffered = flux[selection_mask_buffered, :]
        else:
            raise ValueError('flux has wrong number of dimensions. Should be either 1 or 2.')

    if even_length is True:
        wavelength_new, flux_new = make_spectrum_even(wavelength_new, flux_new)

        if np.mod(wavelength.size, 2) != 0.0:
            raise ValueError('Input wavelength must be even length for even buffer_mask.')
        elif buffer_mask is not None:
            wavelength_buffered, flux_buffered = make_spectrum_even(wavelength_buffered, flux_buffered)
            intermediate_array = wavelength_buffered[np.in1d(wavelength_buffered, wavelength_new, invert=True)]
            buffer_mask = np.in1d(wavelength, intermediate_array)

    if buffer_mask is not None:
        buffer_mask_new = np.in1d(wavelength_buffered, wavelength_new, invert=True)
        return (wavelength_new, flux_new), (wavelength_buffered, flux_buffered, buffer_mask_new, buffer_mask)
    else:
        return wavelength_new, flux_new


def limit_wavelength_interval_multiple_spectra(
        wavelength_limits: tuple, wavelength: np.ndarray, *args, buffer_size=None, even_length=True
):
    """
    Expects all *args to be fluxes, either a collection of spectral fluxes (np.ndarray with ndims 2) or a single
    spectrum each (np.ndarray with ndims 1). They must all have been resampled to the same wavelength grid beforehand.
    :return: lists of wavelength limited spectra, including their wavelength values. Will also return buffered versions
              (with padding in the ends) if buffer_size is given, as well as a buffer mask that can be used to access
              the buffer or the unbuffered data from it.
    """
    buffer_mask_internal = None
    flux_buffered_list = []
    flux_unbuffered_list = []
    for arg in args:
        temp_0, temp_1 = limit_wavelength_interval(
            wavelength_limits, wavelength, arg, buffer_size, buffer_mask_internal, even_length
        )
        if buffer_size is None:
            wavelength_unbuffered = temp_0
            flux_unbuffered_list.append(temp_1)
        else:
            wavelength_unbuffered = temp_0[0]
            flux_unbuffered_list.append(temp_0[1])
            wavelength_buffered = temp_1[0]
            flux_buffered_list.append(temp_1[1])
            buffer_mask_return = temp_1[2]
            buffer_mask_internal = temp_1[3]

    if buffer_size is None:
        return wavelength_unbuffered, flux_unbuffered_list
    else:
        return wavelength_unbuffered, flux_unbuffered_list, wavelength_buffered, flux_buffered_list, buffer_mask_return


def make_spectrum_even(wavelength:np.ndarray, flux:np.ndarray or list):
    if np.mod(wavelength.size, 2) != 0.0:
        wavelength = wavelength[:-1]
        if isinstance(flux, np.ndarray):
            if flux.ndim == 1:
                flux = flux[:-1]
            elif flux.ndim == 2:
                flux = flux[:-1, :]
            else:
                raise ValueError('flux dimension must be 1 or 2.')
        elif isinstance(flux, list):
            for i in range(0, len(flux)):
                if flux[i].ndim == 1:
                    flux[i] = flux[i][:-1]
                elif flux[i].ndim == 2:
                    flux[i] = flux[i][:-1, :]
                else:
                    raise ValueError('flux[i] dimension must be 1 or 2.')
    return wavelength, flux


def moving_median_filter(flux: np.ndarray, window=51):
    """
    Filters the data using a moving median filter. Useful to capture some continuum trends.
    :param flux:        np.ndarray size (n, ) of y-values (fluxes) for each wavelength
    :param window:      integer, size of moving median filter
    :return:            filtered_flux, np.ndarray size (n, )
    """
    from scipy.ndimage import median_filter
    return median_filter(flux, size=window, mode='reflect')


def reduce_emission_lines(wavelength: np.ndarray, flux: np.ndarray, mf_window=401, std_factor=1.5, plot=False,
                          limit=None):
    """
    Reduces the effect of emission lines in a spectrum by median-filtering and cutting away all data points a certain
    point ABOVE the median variance of the data (flux - median flux).
    :param wavelength:              np.ndarray size (n, ) of wavelengths.
    :param flux:                    np.ndarray size (n, ) of fluxes for the spectrum
    :param mf_window:               int, window size of the median filter. Must be uneven.
    :param std_factor:              float, cutoff factor multiplier to the standard deviation of the variance.
    :param plot:                    bool, indicates whether illustrative plots should be shown
    :param limit:                   float, max distance from variance allowed. Default is None
    :return wavelength_reduced:     np.ndarray of wavelengths, masked to the reduced fluxes.
            flux_emission_reduced:  np.ndarray of fluxes for the spectrum, with reduced emission lines.
    """
    flux_filtered = moving_median_filter(flux, mf_window)
    variance = flux - flux_filtered
    mask = variance < std_factor * np.std(variance)
    wavelength_reduced = wavelength[mask]
    flux_emission_reduced = flux[mask]
    if limit is not None:
        mask = variance[mask] < limit
        wavelength_reduced = wavelength_reduced[mask]
        flux_emission_reduced = flux_emission_reduced[mask]

    if plot:
        plt.figure()
        plt.plot(wavelength, flux); plt.plot(wavelength, flux_filtered)
        plt.xlabel('Wavelength'); plt.ylabel('Flux'); plt.legend(['Unfiltered', 'Median Filtered'])
        plt.show(block=False)

        plt.figure()
        plt.plot(wavelength, variance); plt.plot(wavelength_reduced, variance[mask])
        plt.plot([np.min(wavelength), np.max(wavelength)], [std_factor*np.std(variance), std_factor*np.std(variance)],
                 'k')
        plt.plot([np.min(wavelength), np.max(wavelength)],
                 [-std_factor * np.std(variance), -std_factor * np.std(variance)], 'k--')
        plt.xlabel('Wavelength'); plt.ylabel('Variance'); plt.legend(['Full set', 'Set with reduced emission lines'])
        plt.show(block=False)

        plt.figure()
        plt.plot(wavelength, flux, linewidth=1)
        plt.plot(wavelength_reduced, flux_emission_reduced, '--', linewidth=0.8)
        plt.xlabel('Wavelength'); plt.ylabel('Flux'); plt.legend(['Full set', 'Set with reduced emission lines'])
        plt.show()

    return wavelength_reduced, flux_emission_reduced


def updatescatter(handle, orig):
    """https://stackoverflow.com/questions/24706125/setting-a-fixed-size-for-points-in-legend"""
    handle.update_from(orig)
    handle.set_sizes([64])


def updateline(handle, orig):
    """https://stackoverflow.com/questions/24706125/setting-a-fixed-size-for-points-in-legend"""
    handle.update_from(orig)
    handle.set_markersize(12)


def simple_normalizer(
        wavelength, flux, poly1deg=3, poly2deg=4, replace_negative=0.0, replace_non_finite=1.0,
        reduce_em_lines=True, plot=False, return_em_mask=False
):
    matplotlib.rcParams.update({'font.size': 20})
    filtered_flux = moving_median_filter(flux, flux.size//25)

    selection_mask_1 = flux/filtered_flux > 0.85
    polynomial_1 = Polynomial.fit(wavelength[selection_mask_1], flux[selection_mask_1], deg=poly1deg)

    selection_mask_2 = (flux/polynomial_1(wavelength) > 0.95) & (flux/polynomial_1(wavelength) < 1.1)
    polynomial_2 = Polynomial.fit(wavelength[selection_mask_2], flux[selection_mask_2], deg=poly2deg)

    normalized_flux = flux / polynomial_2(wavelength)

    if replace_negative:
        normalized_flux[normalized_flux <= 0] = replace_negative
    if replace_non_finite:
        normalized_flux[~np.isfinite(normalized_flux)] = replace_non_finite

    if reduce_em_lines:
        wav_ = wavelength[selection_mask_1]
        flu_ = normalized_flux[selection_mask_1]
        selection_mask_3 = np.zeros(normalized_flux.size, dtype=bool)
        stds = np.zeros(wavelength.size)
        medians = np.zeros(wavelength.size)
        for i in range(wavelength.size):
            mask = (wav_ > wavelength[i]-250) & (wav_ < wavelength[i]+250) & (flu_ > 1.0)
            flu_sel = flu_[mask]
            median = np.median(flu_sel)
            std = 1.4826 * (median-1.0)
            stds[i] = std
            medians[i] = median
            if normalized_flux[i] < 1 + 2.5*std:
                selection_mask_3[i] = True
        selection_mask_4 = selection_mask_3 & selection_mask_2
        polynomial_2 = Polynomial.fit(wavelength[selection_mask_4], flux[selection_mask_4], deg=poly2deg)
        normalized_flux = flux / polynomial_2(wavelength)

    if plot:
        plt.figure(figsize=(16, 9))
        plt.plot(wavelength, flux, '-', markersize=1)
        plt.plot(wavelength, filtered_flux)
        plt.plot(wavelength, polynomial_1(wavelength))
        plt.plot(wavelength, polynomial_2(wavelength))

        plt.xlabel('Wavelength [Å]', fontsize=22)
        plt.ylabel('Spectral Flux [e/s]', fontsize=22)
        if reduce_em_lines:
            plt.plot(wavelength[~selection_mask_3], flux[~selection_mask_3], '.', color='red', markersize=4)
            plt.legend(
                ['Flux', 'Median filtered flux', '3rd degree Polynomial', '4th degree polynomial', 'Removed data'],
                fontsize=20,
                handler_map={PathCollection: HandlerPathCollection(update_func=updatescatter),
                             plt.Line2D: HandlerLine2D(update_func=updateline)}
            )
            plt.ylim([0.0, 1.05*np.max(flux[selection_mask_3])])
        else:
            plt.legend(
                ['Flux', 'Median filtered flux', '3rd degree Polynomial', '4th degree polynomial'],
                fontsize=20,
                handler_map={PathCollection: HandlerPathCollection(update_func=updatescatter),
                             plt.Line2D: HandlerLine2D(update_func=updateline)}
            )
        plt.tight_layout()
        # plt.savefig('/home/sinkbaek/PycharmProjects/Seismic-dEBs/figures/report/RV/simple_normalizer/continuum_fit.png',
        #            dpi=400)

        if reduce_em_lines:
            plt.figure()
            plt.plot(wavelength, normalized_flux, '-', label='Normalized flux')
            plt.plot(wavelength, medians, 'k--', label='Median absolute deviation above 1')
            plt.plot(wavelength, 2.5*stds + 1, '--', label='2.5 * 1.4826 * MAD')
            plt.ylim(-0.05, 1+3.5*np.max(stds))
            plt.legend()

        plt.show(block=True)

    if reduce_em_lines:
        if return_em_mask:
            return wavelength[selection_mask_3], normalized_flux[selection_mask_3], selection_mask_3
        return wavelength[selection_mask_3], normalized_flux[selection_mask_3]
    else:
        return wavelength, normalized_flux


def save2col(column1: np.ndarray, column2: np.ndarray, filename: str):
    save_data = np.empty((column1.size, 2))
    save_data[:, 0] = column1
    save_data[:, 1] = column2
    np.savetxt(filename, save_data)


def savefig(
        filename: str, xs: np.ndarray, ys: np.ndarray, xlabel: str, ylabel: str, title=None, xlim=None, ylim=None,
        legend=None, legend_loc='upper left',figsize=(17.78, 10), dpi=400
):
    fig = plt.figure(figsize=figsize)
    for i in range(0, len(xs)):
        plt.plot(xs[i], ys[i], linewidth=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if title is not None:
        plt.title(title)
    if legend is not None:
        plt.legend(legend, loc=legend_loc)
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close(fig)




