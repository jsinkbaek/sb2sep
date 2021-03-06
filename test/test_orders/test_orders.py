import matplotlib
import numpy as np
from src.sb2sep import spectral_separation_routine as ssr
from src.sb2sep import spectrum_processing_functions as spf
from src.sb2sep.storage_classes import RadialVelocityOptions, SeparateComponentsOptions, RoutineOptions
import matplotlib.pyplot as plt
import time
import sys

sep_spectra = np.loadtxt('../kic8430105/results/sep_flux_renormalized.txt')
wl = sep_spectra[:, 0]
flux = sep_spectra[:, 1]
template = sep_spectra[:, 3]
if np.mod(wl.size, 2) != 0.0:
    wl = wl[:-1]
    flux = flux[:-1]
    template = template[:-1]

flux_collection = np.ones((wl.size, 6, 10))        # 6 orders, 10 observations
mask_orders = np.zeros((wl.size, 6, 10))
for i in range(0, 10):
    for j in range(0, 6):
        current_range = range(2500+int(j*(wl.size/6-0.1*wl.size/6)), int((j+1)*wl.size/6))
        if np.mod(len(current_range), 2) != 0.0:
            current_range = current_range[0:len(current_range)-1]
        flux_collection[current_range, j, i] = flux[current_range]
        mask_orders[current_range, j, i] = 1.0
        flux_collection[:, j, i] = ssr.shift_spectrum(flux_collection[:, j, i], 30.0*(1-i/5.0), 1.0)
        flux_temp = np.ones((wl.size, ))
        flux_temp[current_range] = flux[current_range]
        flux_collection[:, j, i] += ssr.shift_spectrum(flux_temp, 30.0*(i/5.0-1.0), 1.0)

mask_orders = mask_orders.astype(bool)
flux_collection = flux_collection / 2

rv_options = RadialVelocityOptions(
    vsini_guess_A=4.0, vsini_guess_B=4.0, delta_v=1.0, spectral_resolution=67000, velocity_fit_width_A=10.0,
    velocity_fit_width_B=10.0, limbd_coef_A=0.68, limbd_coef_B=0.68,
    smooth_sigma_B=2.0, smooth_sigma_A=2.0, bf_velocity_span=200, iteration_limit=10, convergence_limit=1e-2,
    verbose=True, n_parallel_jobs=4
)
sep_comp_options = SeparateComponentsOptions(
    delta_v=1.0, convergence_limit=1e-2, max_iterations=10, rv_proximity_limit=5.0, verbose=False, weights=[2.0]*10
)
routine_options = RoutineOptions(
    convergence_limit=1e-5, iteration_limit=2, plot=True, save_all_results=False,
    save_path='./', filename_bulk='test', verbose=True
)

RV_guess = np.empty((10, 2))
RV_guess[:, 0] = np.linspace(30, -30, 10)+(0.2*np.random.rand(10)-0.2*0.5)
RV_guess[:, 1] = np.linspace(-30, 30, 10)+(0.2*np.random.rand(10)-0.2*0.5)


start_time_1 = time.time()
results = ssr.spectral_separation_routine(
    flux_collection, template, template, wl, routine_options, sep_comp_options, rv_options, RV_guess, mask_orders,
    time_values=np.linspace(0, 0.99, 10), period=1.
)
print('--- %s seconds' % (time.time() - start_time_1))
