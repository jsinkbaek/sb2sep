import numpy as np
from scipy.signal import fftconvolve
import lmfit
import scipy.constants as scc


def gaussian_model(vel, amp, rv, sigma, broadening_sigma, continuum):
    n = vel.size
    model = np.ones(n) * continuum
    model += amp*np.exp(-(vel-rv)**2 / (2*sigma)**2)
    broadening = np.exp(-0.5*(vel/broadening_sigma)**2) / (np.sqrt(2*np.pi)*broadening_sigma)
    return fftconvolve(model, broadening, mode="same")


def weight_function(vel, bfvals, vel_fit_hwidth, rv):
    mask = (vel > rv - vel_fit_hwidth) & (vel < rv + vel_fit_hwidth)
    weight = np.zeros(bfvals.size)
    weight[mask] = 1.0
    return weight


def get_values(pars):
    amp, rv, sigma = pars["amp"].value, pars["rv"].value, pars["sigma"].value
    continuum, broad = pars["continuum"].value, pars["broadening_sigma"].value
    return amp, rv, sigma, continuum, broad


def residual_func(params, vel, bfvals, weights):
    pvals = get_values(params)
    return weights * (bfvals-gaussian_model(vel, *pvals))


def fitting_routine(vel, bfvals, mask, config, dv, resolution, rv):
    speed_c = scc.c / 1000.  # in km/s
    smooth_sigma = config["SmoothSigma"]
    vel_width = config["VelocityFitWidth"]
    gaussian_width = np.sqrt(((speed_c / resolution) / (2.354 * dv)) ** 2 + (smooth_sigma / dv) ** 2)
    pars = lmfit.Parameters()

    weights = weight_function(vel, bfvals, vel_width, rv)
    peak = np.argmax(bfvals * weights)
    if config["Amplitude"][0] == 0:
        params.add('amplitude', value=bfvals[peak], min=0.0, max=99 * bfvals[peak], vary=config["Amplitude"][1])
    else:
        params.add(
            'amp', value=config["Amplitude"][0], min=0.0, max=99 * config["Amplitude"][0], vary=config["Amplitude"][1]
        )
    params.add("rv", value=rv, min=rv-vel_width, max=vel+vel_width, vary=True)
    params.add("sigma", value=config["Sigma"][0], min=config["Sigma"][2],max=config["Sigma"][3],vary=config["Sigma"][1])
    params.add(
        "continuum", value=config["Continuum"][0], min=np.min(bfvals), max=np.max(bfvals), vary=config["Continuum"][1]
    )
    params.add("broadening_sigma", value=gaussian_width, vary=False)

    # Estimate uncertainty as STD of BF outside of RV region of interest
    maxmin_hwidth = (np.max(vel) - np.min(vel)) / 2
    dif = np.max([maxmin_hwidth / 3, vel_width])
    mask_err = (vel < rv - dif) | (vel > rv + dif)
    weights = weights / np.std(bfvals[mask_err])

    fit = lmfit.minimize(
        residual_func, params, args=(vel[mask], bfvals[mask], weights[mask]), xtol=1E-10, ftol=1E-10, max_nfev=500
    )

    pvals = get_values(fit.params)
    model = gaussian_model(vel, *pvals)
    return fit, model, pvals
