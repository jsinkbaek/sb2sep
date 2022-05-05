import numpy as np


def load_configuration_files(loc_routine_file, loc_separation_file, loc_rv_file):
    # Load routine config
    col0, col1 = np.genfromtxt(loc_routine_file, dtype=str, usecols=(0, 1), unpack=True)
    routine_options = RoutineOptions()
    for index, value in enumerate(col0):
        if value == 'convergence_limit':
            routine_options.convergence_limit = float(col1[index])
        elif value == 'iteration_limit':
            routine_options.iteration_limit = int(col1[index])
        elif value == 'plot':
            routine_options.plot = col1[index] == 'True'        # False if col1[index] != 'True'
        elif value == 'verbose':
            routine_options.verbose = col1[index] == 'True'
        elif value == 'save_plot_path':
            if col1[index] == 'None':
                routine_options.save_plot_path = None
            else:
                routine_options.save_plot_path = col1[index]
        elif value == 'save_all_results':
            routine_options.save_all_results = col1[index] == 'True'
        elif value == 'save_path':
            routine_options.save_path = col1[index]
        elif value == 'return_unbuffered':
            routine_options.return_unbuffered = col1[index] == 'True'
        elif value == 'plot_order':
            routine_options.plot_order = int(col1[index])
        elif value == 'adjust_vsini':
            routine_options.adjust_vsini = col1[index] == 'True'
        elif value == 'delta_v':
            routine_options.delta_v = float(col1[index])
        elif value == 'filename_bulk':
            if col1[index] == 'None':
                routine_options.filename_bulk = None
            else:
                routine_options.filename_bulk = col1[index]
        else:
            raise KeyError(f'Routine options config file key {value} not supported.')

    # Load separated spectra subroutine config file
    col0, col1 = np.loadtxt(loc_separation_file, dtype=str, usecols=(0, 1), unpack=True)
    sep_options = SeparateComponentsOptions()
    for index, value in enumerate(col0):
        if value == 'delta_v':
            sep_options.delta_v = float(col1[index])
        elif value == 'convergence_limit':
            sep_options.convergence_limit = float(col1[index])
        elif value == 'max_iterations':
            sep_options.max_iterations = int(col1[index])
        elif value == 'rv_proximity_limit':
            sep_options.rv_proximity_limit = float(col1[index])
        elif value == 'rv_lower_limit':
            sep_options.rv_lower_limit = float(col1[index])
        elif value == 'use_for_spectral_separation_A':
            if col1[index] == 'None':
                sep_options.use_for_spectral_separation_A = None
            else:
                eval_list = eval(col1[index])
                sep_options.use_for_spectral_separation_A = np.array(eval_list)
        elif value == 'use_for_spectral_separation_B':
            if col1[index] == 'None':
                sep_options.use_for_spectral_separation_B = None
            else:
                eval_list = eval(col1[index])
                sep_options.use_for_spectral_separation_B = np.array(eval_list)
        elif value == 'weights':
            if col1[index] == 'None':
                sep_options.weights = None
            else:
                eval_list = eval(col1[index])
                sep_options.weights = np.array(eval_list)
        elif value == 'verbose':
            sep_options.verbose = col1[index] == 'True'
        else:
            raise KeyError(f'Separation options config file key {value} not supported.')

    # Load RV subroutine config
    col0, col1 = np.loadtxt(loc_rv_file, dtype=str, usecols=(0, 1), unpack=True)
    rv_options = RadialVelocityOptions()
    for index, value in enumerate(col0):
        if value == 'vsini_guess_A':
            rv_options.vsini_A = float(col1[index])
        elif value == 'vsini_guess_B':
            rv_options.vsini_B = float(col1[index])
        elif value == 'vary_vsini_A':
            rv_options.vary_vsini_A = col1[index] == 'True'
        elif value == 'vary_vsini_B':
            rv_options.vary_vsini_B = col1[index] == 'True'
        elif value == 'vsini_vary_limit_A':
            if col1[index] == 'None':
                rv_options.vsini_vary_limit_A = None
            else:
                rv_options.vsini_vary_limit_A = float(col1[index])
        elif value == 'vsini_vary_limit_B':
            if col1[index] == 'None':
                rv_options.vsini_vary_limit_B = None
            else:
                rv_options.vsini_vary_limit_B = float(col1[index])
        elif value == 'delta_v':
            rv_options.delta_v = float(col1[index])
        elif value == 'spectral_resolution':
            rv_options.spectral_resolution = float(col1[index])
        elif value == 'smooth_sigma_A':
            rv_options.bf_smooth_sigma_A = float(col1[index])
        elif value == 'smooth_sigma_B':
            rv_options.bf_smooth_sigma_B = float(col1[index])
        elif value == 'bf_velocity_span':
            rv_options.bf_velocity_span = float(col1[index])
        elif value == 'velocity_fit_width_A':
            rv_options.velocity_fit_width_A = float(col1[index])
        elif value == 'velocity_fit_width_B':
            rv_options.velocity_fit_width_B = float(col1[index])
        elif value == 'refit_width_A':
            if col1[index] == 'None':
                rv_options.refit_width_A = None
            else:
                rv_options.refit_width_A = float(col1[index])
        elif value == 'refit_width_B':
            if col1[index] == 'None':
                rv_options.refit_width_B = None
            else:
                rv_options.refit_width_B = float(col1[index])
        elif value == 'limbd_coef_A':
            rv_options.limbd_coef_A = float(col1[index])
        elif value == 'limbd_coef_B':
            rv_options.limbd_coef_B = float(col1[index])
        elif value == 'vary_limbd_coef_A':
            rv_options.vary_limbd_coef_A = col1[index] == 'True'
        elif value == 'vary_limbd_coef_B':
            rv_options.vary_limbd_coef_B = col1[index] == 'True'
        elif value == 'ignore_at_phase_A':
            if col1[index] == 'None':
                rv_options.ignore_at_phase_A = None
            else:
                rv_options.ignore_at_phase_A = eval(col1[index])
        elif value == 'ignore_at_phase_B':
            if col1[index] == 'None':
                rv_options.ignore_at_phase_B = None
            else:
                rv_options.ignore_at_phase_B = eval(col1[index])
        elif value == 'verbose':
            rv_options.verbose = col1[index] == 'True'
        elif value == 'convergence_limit':
            rv_options.convergence_limit = float(col1[index])
        elif value == 'iteration_limit':
            rv_options.iteration_limit = int(col1[index])
        elif value == 'rv_lower_limit':
            rv_options.rv_lower_limit = float(col1[index])
        elif value == 'evaluate_spectra_A':
            if col1[index] == 'None':
                rv_options.evaluate_spectra_A = None
            else:
                rv_options.evaluate_spectra_A = eval(col1[index])
        elif value == 'evaluate_spectra_B':
            if col1[index] == 'None':
                rv_options.evaluate_spectra_B = None
            else:
                rv_options.evaluate_spectra_B = eval(col1[index])
        else:
            raise KeyError(f'RV options config file key {value} not supported.')
    return routine_options, sep_options, rv_options


class InitialFitParameters:
    def __init__(
            self,
            vsini_guess=1.0,
            spectral_resolution=60000,
            velocity_fit_width=300,
            limbd_coef=0.68,
            smooth_sigma=4.0,
            bf_velocity_span=200,
            vary_vsini=False,
            vsini_vary_limit=None,
            vary_limbd_coef=False,
            RV=None,
            continuum=0.0,
            vary_continuum=True
    ):
        # Value for vsini, and whether or not to fit it
        self.vsini = vsini_guess
        self.vary_vsini = vary_vsini
        # Maximum change to vsini allowed each iteration in spectral_separation_routine()
        self.vsini_vary_limit = vsini_vary_limit
        # Data resolution
        self.spectral_resolution = spectral_resolution
        # How far away to include data in fitting procedure (rotational broadening function profile also masks
        # profile separately using (velocity - rv)/vsini )
        self.velocity_fit_width = velocity_fit_width
        # Linear limb darkening coefficient for rotational broadening function profile fit
        self.limbd_coef = limbd_coef
        self.vary_limbd_coef = vary_limbd_coef
        # Current RV values (used to update fit RV parameter limits correctly)
        self.RV = RV
        # Smoothing value (in km/s) of the convolved gaussian used in broadening function SVD (bf_smooth()).
        self.bf_smooth_sigma = smooth_sigma
        # Width of the broadening function (in velocity space)
        self.bf_velocity_span = bf_velocity_span
        # Continuum value, and whether to fit for it
        self.continuum = continuum
        self.vary_continuum = vary_continuum


class RadialVelocityOptions:
    def __init__(
            self,
            vsini_guess_A=1.0,
            vsini_guess_B=1.0,
            vary_vsini_A=True,
            vary_vsini_B=True,
            vsini_vary_limit_A=None,
            vsini_vary_limit_B=None,
            delta_v=1.0,
            spectral_resolution=60000,
            velocity_fit_width_A=50,
            velocity_fit_width_B=20,
            refit_width_A=None,
            refit_width_B=None,
            limbd_coef_A=0.68,
            limbd_coef_B=0.68,
            vary_limbd_coef_A=False,
            vary_limbd_coef_B=False,
            RV_A=None,
            RV_B=None,
            smooth_sigma_A=4.0,
            smooth_sigma_B=4.0,
            bf_velocity_span=200,
            ignore_at_phase_A=None,
            ignore_at_phase_B=None,
            verbose=False,
            iteration_limit=6,
            convergence_limit=5e-3,
            rv_lower_limit=0.0,
            print_prec=6,
            n_parallel_jobs=1,
            evaluate_spectra_A=None,
            evaluate_spectra_B=None
    ):
        # Value for vsini, and whether or not to fit it
        self.vsini_A = vsini_guess_A
        self.vsini_B = vsini_guess_B
        self.vary_vsini_A = vary_vsini_A
        self.vary_vsini_B = vary_vsini_B

        # Maximum change to vsini allowed each iteration in spectral_separation_routine()
        self.vsini_vary_limit_A = vsini_vary_limit_A
        self.vsini_vary_limit_B = vsini_vary_limit_B

        # Data resolution
        self.spectral_resolution = spectral_resolution
        self.delta_v = delta_v

        # How far away to include data in fitting procedure (rotational broadening function profile also masks
        # profile separately using (velocity - rv)/vsini )
        self.velocity_fit_width_A = velocity_fit_width_A
        self.velocity_fit_width_B = velocity_fit_width_B
        # Repeat and refit results with different width after honing in?
        self.refit_width_A = refit_width_A
        self.refit_width_B = refit_width_B

        # Linear limb darkening coefficient for rotational broadening function profile fit
        self.limbd_coef_A = limbd_coef_A
        self.limbd_coef_B = limbd_coef_B
        self.vary_limbd_coef_A = vary_limbd_coef_A
        self.vary_limbd_coef_B = vary_limbd_coef_B

        # Current RV values (used to update fit RV parameter limits correctly)
        self.RV_A = RV_A
        self.RV_B = RV_B

        # Smoothing value (in km/s) of the convolved gaussian used in broadening function SVD (bf_smooth()).
        self.bf_smooth_sigma_A = smooth_sigma_A
        self.bf_smooth_sigma_B = smooth_sigma_B

        # Width of the broadening function (in velocity space)
        self.bf_velocity_span = bf_velocity_span

        # Use if component should not be subtracted in a specific phase-area (fx. (0.7, 0.9)), if it is totally eclipsed
        self.ignore_at_phase_A = ignore_at_phase_A
        self.ignore_at_phase_B = ignore_at_phase_B

        # Can be 'False', 'True' (or 'all') or 'errors'
        self.verbose=verbose

        self.iteration_limit=iteration_limit
        self.convergence_limit=convergence_limit

        self.rv_lower_limit = rv_lower_limit

        self.print_prec = print_prec

        self.n_parallel_jobs = n_parallel_jobs

        self.evaluate_spectra_A = evaluate_spectra_A
        self.evaluate_spectra_B = evaluate_spectra_B


class SeparateComponentsOptions:
    def __init__(
            self,
            delta_v=1.0,
            convergence_limit=1e-2,
            max_iterations=10,
            rv_proximity_limit=0.0,
            rv_lower_limit=0.0,
            ignore_component_B=False,
            verbose=False,
            use_for_spectral_separation_A=None,
            use_for_spectral_separation_B=None,
            weights=None
    ):
        self.delta_v = delta_v
        self.convergence_limit = convergence_limit
        self.max_iterations = max_iterations
        self.rv_proximity_limit = rv_proximity_limit
        self.rv_lower_limit = rv_lower_limit
        self.ignore_component_B = ignore_component_B
        self.verbose = verbose
        self.use_for_spectral_separation_A = use_for_spectral_separation_A
        self.use_for_spectral_separation_B = use_for_spectral_separation_B
        self.weights = weights


class RoutineOptions:
    def __init__(
            self,
            convergence_limit=1E-5,
            iteration_limit=10,
            plot=True,
            verbose=True,
            return_unbuffered=True,
            save_plot_path=None,
            save_all_results=True,
            save_path='./',
            filename_bulk=None,
            plot_order=0,
            adjust_vsini=True,
            delta_v=1.0
    ):
        self.convergence_limit = convergence_limit
        self.iteration_limit = iteration_limit
        self.plot = plot
        self.verbose = verbose
        self.return_unbuffered = return_unbuffered
        self.save_plot_path = save_plot_path
        self.save_all_results = save_all_results
        self.save_path = save_path
        self.filename_bulk = filename_bulk
        self.plot_order = plot_order
        self.adjust_vsini = adjust_vsini
        self.delta_v = delta_v
