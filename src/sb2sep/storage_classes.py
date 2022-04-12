
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
            RV=None
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
            n_parallel_jobs=1
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
            time_values=None,
            convergence_limit=1E-5,
            iteration_limit=10,
            plot=True,
            verbose=True,
            return_unbuffered=True,
            save_plot_path=None,
            save_all_results=True,
            save_path='./',
            buffer_mask=None,
            filename_bulk=None
    ):
        self.time_values = time_values
        self.convergence_limit = convergence_limit
        self.iteration_limit = iteration_limit
        self.plot = plot
        self.verbose = verbose
        self.return_unbuffered = return_unbuffered
        self.save_plot_path = save_plot_path
        self.save_all_results = save_all_results
        self.save_path = save_path
        self.buffer_mask = buffer_mask
        self.filename_bulk = filename_bulk
