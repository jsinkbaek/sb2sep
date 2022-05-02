import numpy as np
import matplotlib.pyplot as plt
import matplotlib


"""
This file is meant to be used with the additional data results provided by the spectral separation routine
(can be turned on by using the save_additional_results switch in spectral_separation_routine()). Default location for
these files are in RV/Data/additionals/spectral_separation/.

Two storage classes are provided, IntervalResult (where loaded data is stored) and RoutineResults (where multiple
IntervalResult objects are accessed). Datafiles from the separation routine can be loaded into these objects using 
load_routine_results(), which provides a RoutineResults instance that can be fed into the plotting functions.
load_routine_results() requests a folder path, as well as a list of "bulk" filenames, e.g. the wavelength-intervals. An
example would be:
load_routine_results(RV/Data/additionals/spectral_separation/, ['4500_5000', '5000_5500'])
"""


class IntervalResult:
    def __init__(
            self, time_values_A, time_values_B,
            wavelength, separated_spectrum_A, separated_spectrum_B, template_spectrum_A, template_spectrum_B,
            RV_A, RV_B, RV_A_initial, RV_B_initial,
            bf_velocity_A, bf_vals_A, bf_smooth_A, bf_model_vals_A,
            bf_velocity_B, bf_vals_B, bf_smooth_B, bf_model_vals_B,
            wavelength_a=None, wavelength_b=None
    ):
        self.time_values_A = time_values_A
        self.time_values_B = time_values_B
        self.wavelength = wavelength
        self.separated_spectrum_A = separated_spectrum_A
        self.separated_spectrum_B = separated_spectrum_B
        self.template_flux_A = template_spectrum_A
        self.template_flux_B = template_spectrum_B
        self.RV_A = RV_A
        self.RV_B = RV_B
        self.RV_A_initial = RV_A_initial
        self.RV_B_initial = RV_B_initial
        self.bf_velocity_A = bf_velocity_A
        self.bf_vals_A = bf_vals_A
        self.bf_smooth_A = bf_smooth_A
        self.bf_model_vals_A = bf_model_vals_A
        self.bf_velocity_B = bf_velocity_B
        self.bf_vals_B = bf_vals_B
        self.bf_smooth_B = bf_smooth_B
        self.bf_model_vals_B = bf_model_vals_B
        if wavelength_a is None:
            self.wavelength_a = int(np.round(np.min(wavelength)))
        if wavelength_b is None:
            self.wavelength_b = int(np.round(np.max(wavelength)))


class RoutineResults:
    def __init__(self, *args: IntervalResult):
        if len(args) != 0:
            self.interval_results = [x for x in args]
        else:
            self.interval_results = []

    @property
    def time_values_A(self):
        return [x.time_values_A for x in self.interval_results]

    @property
    def time_values_B(self):
        return [x.time_values_B for x in self.interval_results]

    @property
    def RV_A(self):
        return [x.RV_A for x in self.interval_results]

    @property
    def RV_B(self):
        return [x.RV_B for x in self.interval_results]

    @property
    def RV_A_initial(self):
        return [x.RV_A_initial for x in self.interval_results]

    @property
    def RV_B_initial(self):
        return [x.RV_B_initial for x in self.interval_results]

    @property
    def wavelengths(self, interval=None):
        if interval is not None:
            res = None
            for interval_result in self.interval_results:
                if interval_result.wavelength_a == interval[0] and interval_result.wavelength_b == interval[1]:
                    return interval_result.wavelength
            if res is None:
                raise ValueError('No interval found with the same wavelength_a and wavelength_B.')
        else:
            return [x.wavelength for x in self.interval_results]

    @property
    def wavelength_a(self):
        return [x.wavelength_a for x in self.interval_results]

    @property
    def wavelength_b(self):
        return [x.wavelength_b for x in self.interval_results]

    @property
    def interval(self):
        return [(x.wavelength_a, x.wavelength_b) for x in self.interval_results]

    @property
    def separated_spectra_A(self):
        return [x.separated_spectrum_A for x in self.interval_results]

    @property
    def separated_spectra_B(self):
        return [x.separated_spectrum_B for x in self.interval_results]

    @property
    def template_flux_A(self):
        return [x.template_flux_A for x in self.interval_results]

    @property
    def template_flux_B(self):
        return [x.template_flux_B for x in self.interval_results]

    @property
    def bf_results(self):
        return [
            [x.bf_velocity_A, x.bf_vals_A, x.bf_smooth_A, x.bf_model_vals_A, x.bf_velocity_B, x.bf_vals_B,
             x.bf_smooth_B, x.bf_model_vals_B]
            for x in self.interval_results
        ]

    def append_interval(self, new_result: IntervalResult):
        self.interval_results.append(new_result)


def load_routine_results(folder_path: str, filename_bulk_list: list):
    routine_results = RoutineResults()
    for filename_bulk in filename_bulk_list:
        try:
            rvA_array = np.loadtxt(folder_path+filename_bulk+'_rvA.txt')
            rvB_array = np.loadtxt(folder_path+filename_bulk+'_rvB.txt')
        except OSError:
            rvA_array = np.loadtxt(folder_path + '_rvA.txt')
            rvB_array = np.loadtxt(folder_path + '_rvB.txt')
        sep_array = np.loadtxt(folder_path+filename_bulk+'_sep_flux.txt')
        velA_array = np.loadtxt(folder_path+filename_bulk+'_velocities_A.txt')
        bfA_array = np.loadtxt(folder_path+filename_bulk+'_bfvals_A.txt')
        bfA_smooth_array = np.loadtxt(folder_path+filename_bulk+'_bfsmooth_A.txt')
        modelA_array = np.loadtxt(folder_path+filename_bulk+'_models_A.txt')
        velB_array = np.loadtxt(folder_path+filename_bulk+'_velocities_B.txt')
        bfB_array = np.loadtxt(folder_path+filename_bulk+'_bfvals_B.txt')
        bfB_smooth_array = np.loadtxt(folder_path+filename_bulk+'_bfsmooth_B.txt')
        modelB_array = np.loadtxt(folder_path+filename_bulk+'_models_B.txt')
        rvs_initial = np.loadtxt(folder_path+filename_bulk+'_rv_initial.txt')

        wavelength, separated_flux_A, separated_flux_B = sep_array[:, 0], sep_array[:, 1], sep_array[:, 2]
        template_flux_A, template_flux_B = sep_array[:, 3], sep_array[:, 4]
        if rvA_array[0, :].size != 2:
            time_values_A, RV_A = rvA_array[:, 0], rvA_array[:, 1:]
            time_values_B, RV_B = rvB_array[:, 0], rvB_array[:, 1:]
        else:
            time_values_A, RV_A = rvA_array[:, 0], rvA_array[:, 1]
            time_values_B, RV_B = rvB_array[:, 0], rvB_array[:, 1]
        RV_A_initial, RV_B_initial = rvs_initial[:, 0], rvs_initial[:, 1]

        interval_result = IntervalResult(
            time_values_A, time_values_B, wavelength, separated_flux_A, separated_flux_B, template_flux_A,
            template_flux_B, RV_A, RV_B, RV_A_initial, RV_B_initial, velA_array, bfA_array,
            bfA_smooth_array, modelA_array, velB_array, bfB_array, bfB_smooth_array, modelB_array
        )
        routine_results.append_interval(interval_result)
    return routine_results


def plot_rv_and_separated_spectra(
        evaluation_data: RoutineResults,
        period: float,
        block=False,
        color_A='b',
        color_B='r',
        fig=None,
        axs: list = None
):
    matplotlib.rcParams.update({'font.size': 20})
    for i in range(0, len(evaluation_data.interval_results)):
        if fig is None:
            fig = plt.figure(figsize=(16, 9))
        if axs is None:
            gspec = fig.add_gridspec(2, 2)
            ax1 = fig.add_subplot(gspec[0, :])
            ax2 = fig.add_subplot(gspec[1, 0])
            ax3 = fig.add_subplot(gspec[1, 1])
        else:
            ax1, ax2, ax3 = axs

        phase_A = np.mod(evaluation_data.time_values_A[i], period) / period
        phase_B = np.mod(evaluation_data.time_values_B[i], period) / period

        if evaluation_data.RV_A[i].ndim == 1:
            ax1.plot(phase_A, evaluation_data.RV_A[i], color_A+'*')
            ax1.plot(phase_B, evaluation_data.RV_B[i], color_B+'*')
        else:
            ax1.plot(phase_A, np.mean(evaluation_data.RV_A[i]), color_A + '*')
            ax1.plot(phase_B, np.mean(evaluation_data.RV_B[i]), color_B + '*')
        ax1.set_xlabel('Orbital Phase')
        ax1.set_ylabel('Radial Velocity - system velocity (km/s)', fontsize=15)

        ax2.plot(evaluation_data.wavelengths[i], evaluation_data.template_flux_A[i], '--', color='grey', linewidth=0.5)
        ax2.plot(evaluation_data.wavelengths[i], evaluation_data.separated_spectra_A[i], color_A, linewidth=2)
        ax2.set_xlabel('Wavelength (Å)')
        ax2.set_ylabel('Normalized Separated Flux', fontsize=15)

        ax3.plot(evaluation_data.wavelengths[i], evaluation_data.template_flux_B[i], '--', color='grey', linewidth=0.5)
        ax3.plot(evaluation_data.wavelengths[i], evaluation_data.separated_spectra_B[i], color_B, linewidth=2)
        ax2.set_xlabel('Wavelength (Å)')
        ax3.set_xlabel('Wavelength (Å)')
        fig.suptitle(f'Interval results {evaluation_data.wavelength_a[i]}-{evaluation_data.wavelength_b[i]} Å ')
        plt.tight_layout()

    plt.show(block=block)
    return fig, ax1, ax2, ax3


def plot_broadening_functions(evaluation_data: RoutineResults, block=False, xlim=None, smoothed=True):
    matplotlib.rcParams.update({'font.size': 25})
    for i in range(0, len(evaluation_data.interval_results)):
        fig = plt.figure(figsize=(16, 15))
        gspec = fig.add_gridspec(1, 2)
        ax1 = fig.add_subplot(gspec[0, 0])
        ax2 = fig.add_subplot(gspec[0, 1])

        bf_results = evaluation_data.bf_results[i]
        if smoothed is True:
            vel_A, bf_A, models_A = bf_results[0], bf_results[2], bf_results[3]
            vel_B, bf_B, models_B = bf_results[4], bf_results[6], bf_results[7]
        else:
            vel_A, bf_A, models_A = bf_results[0], bf_results[1], bf_results[3]
            vel_B, bf_B, models_B = bf_results[4], bf_results[5], bf_results[7]
        RV_A = evaluation_data.RV_A[i]
        RV_B = evaluation_data.RV_B[i]
        RV_B_initial = evaluation_data.RV_B_initial[i]

        if xlim is None:
            ax1.set_xlim([np.min(vel_A), np.max(vel_A)])
            ax2.set_xlim([np.min(vel_A), np.max(vel_A)])
        else:
            ax1.set_xlim(xlim)
            ax2.set_xlim(xlim)

        ax1.plot(np.zeros(shape=(2,)), [0.0, 1.05], '--', color='grey')
        ax2.plot(np.zeros(shape=(2,)), [0.0, 1.05], '--', color='grey')
        for k in range(0, vel_A[:, 0].size):
            offset = k/vel_A[:, 0].size
            scale = (0.5/vel_A[:, 0].size)/np.max(bf_A[k, :])
            ax1.plot(vel_A[k, :], 1 + scale*bf_A[k, :] - offset)
            ax1.plot(vel_A[k, :], 1 + scale*models_A[k, :] - offset, 'k--')
            # ax1.plot(np.ones(shape=(2,))*RV_A[k], [1-offset*1.01, 1+5/4 * scale*np.max(models_A[k, :])-offset],
            #          color='blue')

            scale = (0.5/vel_B[:, 0].size)/np.max(bf_B[k, :])
            ax2.plot(vel_B[k, :], 1 + scale*bf_B[k, :] - offset)
            ax2.plot(vel_B[k, :], 1 + scale*models_B[k, :] - offset, 'k--')
            # ax2.plot(np.ones(shape=(2,)) * RV_B[k], [1-offset*1.01, 1+5/4 * scale*np.max(models_B[k, :])-offset],
            #          color='blue')
            # ax2.plot(np.ones(shape=(2,)) * RV_B_initial[k], [1-offset*1.01, 1+5/4*scale*np.max(models_B[k, :])-offset],
            #          color='grey')
            if smoothed is True:
                ax1.set_ylabel('Normalized, smoothed Broadening Function')
            else:
                ax1.set_ylabel('Normalized Broadening Function')
            ax1.set_xlabel('Velocity Shift [km/s]')
            ax2.set_xlabel('Velocity Shift [km/s]')
            fig.suptitle(f'Interval results {evaluation_data.wavelength_a[i]}-{evaluation_data.wavelength_b[i]} Å ')
            ax1.tick_params(
                axis='y',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                right=False,  # ticks along the bottom edge are off
                left=False,  # ticks along the top edge are off
                labelleft=False)  # labels along the bottom edge are off
            ax2.tick_params(
                axis='y',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                right=False,  # ticks along the bottom edge are off
                left=False,  # ticks along the top edge are off
                labelleft=False)  # labels along the bottom edge are off
            plt.tight_layout()
    plt.show(block=block)


def _create_telluric_spectrum(wavelength_a=4000.0, wavelength_b=9500.0, plot=False):
    path_telluric = 'RV/Data/template_spectra/telluric_noheader.txt'

    def gaussian(x, amp, center, fwhm_):
        return amp * np.exp(-(x-center)**2 / (2*(fwhm_/2.35482)**2))

    wavelength = np.linspace(wavelength_a, wavelength_b, 5000)
    flux = np.ones(shape=wavelength.shape)
    telluric_data = np.loadtxt(path_telluric)

    for i in range(0, telluric_data[:, 0].size):
        w1, w2 = telluric_data[i, 0], telluric_data[i, 1]
        fwhm = w2 - w1
        mask = (wavelength > w1-2*fwhm) & (wavelength < w2+2*fwhm)
        residual_intensity, line_center = telluric_data[i, 2], telluric_data[i, 3]
        if mask.size != 0:
            flux[mask] -= gaussian(wavelength[mask], 1-residual_intensity, line_center, fwhm)

    if plot:
        plt.figure()
        plt.plot(wavelength, flux)
        plt.show(block=True)

    return wavelength, flux


def compare_interval_result(evaluation_data: RoutineResults, spectrum_number: int, block=True, ax=None, hide_bad=False):
    if ax is None:
        matplotlib.rcParams.update({'font.size': 25})
        fig, ax_ = plt.subplots(figsize=(16, 9))
    else:
        ax_ = ax

    w_a = np.empty((len(evaluation_data.interval_results), ))
    w_b = np.empty((len(evaluation_data.interval_results), ))
    RV_B = np.empty((len(evaluation_data.interval_results), ))
    w_a[:] = np.nan
    w_b[:] = np.nan
    RV_B[:] = np.nan
    for i in range(0, len(evaluation_data.interval_results)):
        if hide_bad is True and evaluation_data.RV_B_flags[i][spectrum_number] == 0.0:
            pass
        else:
            w_a[i] = evaluation_data.wavelength_a[i]
            w_b[i] = evaluation_data.wavelength_b[i]
            RV_B[i] = evaluation_data.RV_B[i][spectrum_number]

    ax_.errorbar(w_a + (w_b-w_a)/2, RV_B, xerr=(w_b-w_a)/2, fmt='*')
    telluric_wl, telluric_fl = _create_telluric_spectrum(np.min(w_a), np.max(w_b))
    ax_.plot(telluric_wl, telluric_fl*30-55, '--', color='grey', linewidth=0.5)

    if ax is None:
        ax_.set_xlabel('Wavelength (Å)')
        ax_.set_ylabel('Component B RV - SV (km/s)')
        plt.show(block=block)


def compare_interval_results_multiple_spectra(evaluation_data: RoutineResults, block=True, hide_bad=True):
    matplotlib.rcParams.update({'font.size': 25})
    fig, ax = plt.subplots(figsize=(16, 9))

    for i in range(0, evaluation_data.RV_B[0].size):
        compare_interval_result(evaluation_data, spectrum_number=i, ax=ax, hide_bad=hide_bad)

    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Component B RV - SV (km/s)')

    plt.show(block=block)


