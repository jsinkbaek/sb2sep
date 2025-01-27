import numpy as np
from .shift_wavelength_spectrum import shift_wavelength_spectrum as shift_spec
from .bf_modelfit.rotbf import fitting_routine as fit_rotbf
from .bf_modelfit.gaussian import fitting_routine as fit_gauss
from .bf_modelfit.separated_spectrum import fitting_routine as fit_sepbf


def radial_velocity_subroutine(
        wavelength_grid, flux_collection, config,
        separated_fluxes, flux_templates, rv_sets, weight_sets, fitflag_sets, subtractflag_sets,
        output_queue, plot_queue, routine_iteration
):
    wl = wavelength_grid
    fl = flux_collection
    sep = separated_fluxes
    tl = flux_templates
    rvconf = config["radialvelocity"]
    print("#### Radial velocity subroutine ####")

    if rvconf["bfLinAlgPackage"] == "jax":
        from .broadening_function_svd_jax import BroadeningFunction
    elif rvconf["bfLinAlgPackage"] == "numpy":
        from .broadening_function_svd_numpy import BroadeningFunction
    else:
        raise ValueError("Unknown configuration for bfLinAlgPackage")

    delta_v = config["DeltaV"]
    n_spectra = fl.shape[1]
    n_components = len(rvconf["components"].keys())
    keys = rvconf["components"].keys()

    # Produce broadening function of separated spectra (in case it's needed for fitting)
    sep_bf = {}
    for i in range(n_components):
        component = keys[i]
        compconf = rvconf["components"][component]
        BFsep=BroadeningFunction(1-sep[i],1-tl[i],compconf["VelocitySpan"],config["DeltaV"],compconf["SmoothSigma"])
        BFsep.solve()
        if BFsep.smooth_sigma > 0.0:
            BFsep.smooth()
        sep_bf[component] = BFsep

    for k in range(n_spectra):
        iterations = 0
        while iterations < config["rvIterationLimit"]:
            diff = [0. for i in range(n_components)]

            for i in range(n_components):
                component = keys[i]
                compconf = rvconf["components"][component]      # Current component configuration
                if fitflag_sets[i][k] is False:
                    continue

                # Correct spectrum for other components
                fl_cor = 1-fl[:, k]
                for j in range(n_components):
                    if i == j or subtractflag_sets[j][k] is False:
                        continue
                    if fitflag_sets[j][k] is False:
                        if config["RVwhenNotFitted"] == "Skip":
                            continue
                    fl_cor -= shift_spec(wl, 1-sep[j], rv_sets[j][k], config["rvInterpolationMethod"], 0.0)

                # Shift template to spectrum reference frame, and calculate Broadening Function
                tshift = shift_spec(wl, 1-tl[i], rv_sets[i][k], config["rvInterpolationMethod"], 0.0)
                if compconf["BufferRange"] > 0:
                    tshift = tshift[compconf["BufferRange"]:-compconf["BufferRange"]]
                    fl_cor = fl_cor[compconf["BufferRange"]:-compconf["BufferRange"]]
                BF = BroadeningFunction(fl_cor,tshift,compconf["VelocitySpan"],config["DeltaV"],compconf["SmoothSigma"])
                BF.solve()
                vel = BF.velocity
                if compconf["SmoothSigma"] > 0.0:
                    BF.smooth()
                    bf = BF.bf_smooth
                else:
                    bf = BF.bf_vals

                # Mask out data around other components' RVs (if requested)
                velmask = np.ones_like(vel, dtype=bool)
                for j in range(n_components):
                    other_comp = keys[j]
                    excl_hwidth = rvconf["components"][other_comp]["ExcludeFromOtherFits"]
                    if i == j or subtractflag_sets[j][k] is False or excl_hwidth is False:
                        continue
                    if fitflag_sets[j][k] is False:
                        if config["RVwhenNotFitted"] == "Skip":
                            continue
                    rv_other = rv_sets[j][k]-rv_sets[i][k]
                    velmask = velmask & (np.abs(vel-rv_other) > excl_hwidth)

                # Perform the RV fit
                if compconf["FitFun"] == "RotBF":
                    output = fit_rotbf(
                        vel, bf, velmask, compconf, config["DeltaV"], config["SpectralResolution"], rv=0
                    )
                elif compconf["FitFun"] == "Gaussian":
                    output = fit_gauss(
                        vel, bf, velmask, compconf, config["DeltaV"], config["SpectralResolution"], rv=0
                    )
                elif compconf["FitFun"] == "SepBF":
                    output = fit_sepbf(vel, bf, velmask, sep_bf[component], compconf, rv=0)
                else:
                    raise NotImplementedError()

                rv_fit = output[0].params["rv"].value

                # Send to output writing and plotting processes
                tosend = {
                    "subroutine": "rv", "rv_iteration": iterations, "component": component, "spectrum": k,
                    "fitfun": compconf["FitFun"], "result": output, "data": (vel, bf), "mask": velmask,
                    "rv_prev": rv_sets[i][k], "iteration": routine_iteration, "component_number": i
                }
                output_queue.put(tosend)
                plot_queue.put(tosend)

                # Update RV
                rv_sets[i][k] += rv_fit
                diff[i] += rv_fit

            # Check convergence
            covergence = True
            diff = np.abs(np.array(diff))
            for i in range(n_components):
                if diff[i] > rvconf["components"][keys[i]]["Convergence"]
                    convergence = False
            # Print
            if conf["verbose"]:
                print(f"Spectrum {k}, iter {iterations}. RV diff:\t", end="")
                for i in range(n_components):
                    converg = rvconf["components"][keys[i]]["Convergence"]
                    decimal = str(converg)[::-1].find(".")
                    print(f"{diff[i]:.{decimal}f} km/s\t", end="")
                print()
            iterations += 1
            if convergence:
                break
    return rv_sets

