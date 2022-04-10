import numpy as np
from src.sb2sep import spectral_separation_routine as ssr
from src.sb2sep import spectrum_processing_functions as spf
from src.sb2sep.storage_classes import RadialVelocityOptions, SeparateComponentsOptions, RoutineOptions


sep_spectra = np.loadtxt('../kic8430105/results/4500_5825_sep_flux.txt')
wl = sep_spectra[:, 0]
flux = sep_spectra[:, 1]
template = sep_spectra[:, 3]
if np.mod(wl.size, 2) != 0.0:
    wl = wl[:-1]
    flux = flux[:-1]
    template = template[:-1]

flux_collection = np.zeros((wl.size, 6, 10))        # 6 orders, 10 observations
for i in range(0, 10):
    for j in range(0, 6):
        flux_collection[
        2500+int(j*(wl.size/6-0.1*wl.size/6)):j*wl.size//6 -2500, j, i
        ] = flux[2500+int(j*(wl.size/6-0.1*wl.size/6)):j*wl.size//6 -2500]
        flux_collection[:, j, i] = ssr.shift_spectrum(flux_collection[:, j, i], 30.0*(1-i/5.0), 1.0)
        flux_temp = np.zeros((wl.size, ))
        flux_temp[
        2500+int(j*(wl.size/6-0.1*wl.size/6)):j*wl.size//6 -2500
        ] = flux[2500+int(j*(wl.size/6-0.1*wl.size/6)):j*wl.size//6 -2500]
        flux_collection[:, j, i] += ssr.shift_spectrum(flux_temp, 30.0*(i/5.0-1.0), 1.0)

rv_options = RadialVelocityOptions(
    vsini_guess_A=4.0, vsini_guess_B=4.0, delta_v=1.0, spectral_resolution=67000, velocity_fit_width_A=60,
    velocity_fit_width_B=60, limbd_coef_A=0.68, limbd_coef_B=0.68, refit_width_A=10.0, refit_width_B=10.0,
    smooth_sigma_B=2.0, smooth_sigma_A=2.0, bf_velocity_span=300, iteration_limit=5, convergence_limit=5e-3
)
sep_comp_options = SeparateComponentsOptions(
    delta_v=1.0, convergence_limit=1e-2, max_iterations=10, rv_proximity_limit=5.0
)
routine_options = RoutineOptions(
    convergence_limit=1e-5, iteration_limit=2, plot=False, save_all_results=False
)

RV_guess = np.empty((10, 2))
RV_guess[:, 0] = np.linspace(30, -30, 10)+(0.2*np.random.rand(10)-0.2*0.5)
RV_guess[:, 1] = np.linspace(-30, 30, 10)+(0.2*np.random.rand(10)-0.2*0.5)

mask_orders = flux_collection.astype(bool)
results = ssr.spectral_separation_routine(
    flux_collection, template, template, wl, routine_options, sep_comp_options, rv_options, RV_guess, mask_orders

)
print(results)