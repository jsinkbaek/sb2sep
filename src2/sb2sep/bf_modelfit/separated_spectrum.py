import numpy as np
from scipy.interpolate import Akima1Dinterpolator


def sep_model(vel, sep_intp, amp, rv, continuum):
    model = amp*sep_intp(vel-rv, fill_value=0.0)+continuum
    return model


def weight_function(vel, bfvals, vel_fit_hwidth, rv):
    mask = (vel > rv - vel_fit_hwidth) & (vel < rv + vel_fit_hwidth)
    weight = np.zeros(bfvals.size)
    weight[mask] = 1.0
    return weight


def get_values(pars):
    amp, rv, continuum = pars["amp"].value, pars["rv"].value, pars["continuum"].value
    return amp, rv, continuum


def residual_func(params, vel, bfvals, weights, sep_intp):
    pvals = get_values(params)
    return weights * (bfvals - sep_model(vel, sep_intp, *pvals))


def fitting_routine(vel, bfvals, mask, sep_bf, config, rv):
    vel_width = config["VelocityFitWidth"]
    pars = lmfit.Parameters()
    weights = weight_function(vel, bfvals, vel_width, rv)
    peak = np.argmax(bfvals * weights)
    if config["Amplitude"][0] == 0:
        params.add('amp', value=bfvals[peak], min=0.0, max=99 * bfvals[peak], vary=config["Amplitude"][1])
    else:
        params.add(
            'amp', value=config["Amplitude"][0], min=0.0, max=99 * config["Amplitude"][0], vary=config["Amplitude"][1]
        )
    params.add("rv", value=rv, min=rv-vel_width, max=vel+vel_width, vary=True)
    params.add(
        "continuum", value=config["Continuum"][0], min=np.min(bfvals), max=np.max(bfvals), vary=config["Continuum"][1]
    )

    # Estimate uncertainty as STD of BF outside of RV region of interest
    maxmin_hwidth = (np.max(vel) - np.min(vel)) / 2
    dif = np.max([maxmin_hwidth / 3, vel_width])
    mask_err = (vel < rv - dif) | (vel > rv + dif)
    weights = weights / np.std(bfvals[mask_err])

    sep_intp = Akima1Dinterpolator(vel, sep_bf, method="makima", extrapolate=False)
    fit = lmfit.minimize(
        residual_func, params, args=(vel[mask], bfvals[mask], weights[mask], sep_intp),
        xtol=1E-10, ftol=1E-10, max_nfev=500
    )

    pvals = get_values(fit.params)
    model = sep_model(vel, sep_intp, *pvals)
    return fit, model, pvals


