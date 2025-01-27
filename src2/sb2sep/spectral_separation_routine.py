import numpy as np
import matplotlib.pyplot as plt
from .handle_inlist import read_inlist
from .load_preprocessed_data import load_preprocessed_data
from .separate_components_subroutine import separate_components_subroutine
from multiprocessing import Process, Manager
from pathlib import Path
import os
import pandas as pd


def spectral_separation_routine(config, inlist_file):
    if config["SpectroscopyType"] == "LongSpec":
        spectral_separation_run(config, inlist_file)
        for wl_lims in config["ErrorWavelengthRegions"]:
            spectral_separation_run(config, inlist_file, error_wl_lims=wl_lims)
    elif config["SpectroscopyType"] == "OrderSpec":
        orange = config["OrderRange"]
        if len(orange) == 2:
            orders = np.arange(orange[0], orange[1]+1, dtype=int)
        else:
            orders = orange
        for order in orders:
            spectral_separation_run(config, inlist_file, ordernumber=order)
    else:
        raise ValueError("Unknown SpectroscopyType in configuration file. Please use either 'LongSpec' or 'OrderSpec'.")


def spectral_separation_run(config, inlist_file, ordernumber=None, error_wl_lims=None):
    flux_collection, wavelength_grid, flux_template_sets, barycorrs, bjds = load_preprocessed_data(
        config, inlist_file, ordernumber, error_wl_lims
    )
    wl = wavelength_grid
    fl = flux_collection
    manager = Manager()
    output_queue = manager.Queue()
    plot_queue = manager.Queue()

    n_components = len(config["radialvelocity"]["components"])
    n_spectra = fl.shape[1]
    convergence = config["RoutineConvergence"]
    maxiteration = config["RoutineMaxIterations"]
    filenames, rv_sets, weight_sets, fitflag_sets, subtractflag_sets = read_inlist(inlist_file)

    outproc = Process(
        target=output_process,
        args=(output_queue, wavelength_grid, config, filenames, barycorrs, bjds, ordernumber, error_wl_lims)
    )
    plotproc = Process(
        target=plot_process,
        args=(plot_queue, wavelength_grid, config, filenames, barycorrs, bjds, ordernumber, error_wl_lims)
    )

    iterations = 0
    print("Spectral Separation and RV extraction: ")
    while True:
        print(f'\nIteration {iterations}.')
        RMS = [-rv_sets[i][fitflag_sets[i]] for i in range(n_components)]

        sep_flux = separate_components_subroutine(
            wl, fl, config, rv_sets, weight_sets, fitflag_sets, subtractflag_sets, iterations
        )
        rv_sets = radial_velocity_subroutine(
            wl, fl, config, sep_flux, flux_template_sets, rv_sets, fitflag_sets, subtractflag_sets,
            output_queue, plot_queue, iterations
        )

        convergence = True
        print("RV difference since last routine iteration (km/s):\nMean\tMin\tMax")
        for i in range(n_components):
            comp = config["radialvelocity"]["components"].keys()[i]
            RMS[i] = np.abs(RMS[i]+rv_sets[i][fitflag_sets[i]])
            converg =  config["radialvelocity"]["components"][keys[i]]["Convergence"]
            decimal = str(converg)[::-1].find(".")
            if np.any(RMS[i] > converg):
                convergence = False
            print(f"{np.mean(RMS[i]):.{decimal}f}\t{np.min(RMS[i]):.{decimal}f}\t{np.max(RMS[i]):.{decimal}f}")
        iterations += 1

        if convergence:
            print(f"Convergence limit reached for all RVs of all components. Exiting routine.")
            break
        if iterations >= config["RoutineMaxIterations"]:
            print(f"Maximum iterations reached. Exiting routine.")
            break

    # Final spectral separation with final RVs
    sep_flux = separate_components_subroutine(
        wl, fl, config, rv_sets, weight_sets, fitflag_sets, subtractflag_sets, iterations
    )

    output_queue.put(None)
    plot_queue.put(None)
    outproc.join()
    plotproc.join()


def make_outpath(path_root, ordernumber, error_wl_lims):
    if ordernumber is not None:
        path = Path(os.path.join(path_root, str(ordernumber))).mkdir(exist_ok=True)
    elif error_wl_lims is not None:
        path = Path(
            os.path.join(path_root, f"{error_wl_lims[0]:.0f}_{error_wl_lims[1]:.0f}")
        ).mkdir(exist_ok=True)
    else:
        path = Path(path_root).mkdir(exist_ok=True)
    if "bf" in path_root:
        Path(os.path.join(path, "sepflux")).mkdir(exist_ok=True)
        Path(os.path.join(path, "bf")).mkdir(exist_ok=True)
    return path


def output_process(queue, wavelength, config, filenames, barycorrs, bjds, ordernumber, error_wl_lims):
    path = make_outpath(config["OutputFolder"], ordernumber, error_wl_lims)
    components = config["radialvelocity"]["components"].keys()
    n_components = len(comps)
    n_spectra = len(filenames)
    iteration = 0
    files = np.array(filenames)

    rvs = np.zeros((n_spectra, 2*n_components+3), dtype=object)
    rvs_fmt = ["%s"] + ["%.6f" for i in range(2*n_components+2)]
    rvs[:, 0] = files
    rvs[:, 1] = bjds
    rvs[:, 2] = barycorrs

    while True:
        val = queue.get()
        if val is None: break

        riteration = val["iteration"]

        if val["subroutine"] == "sep":
            sep_array = np.zeros((wavelength.size, n_components+1))
            sep_array[:, 0] = wavelength
            for i in range(n_components):
                sep_array[:, i+1] = val["sep_flux"][i]
            iterstr = f"iter{riteration}" if riteration < config["RoutineMaxIterations"] else "final"
            np.savetxt(os.path.join(path, "sepflux", f"sepflux_{iterstr}.txt"), sep_array)

        elif val["subroutine"] == "rv":
            iterstr = f"iter{riteration}" if riteration < config["RoutineMaxIterations"]-1 else "final"

            fit, model, pvals = val["output"]
            rv_old = val["rv_prev"]
            rv_new = rv_old+fit.params["rv"].value
            rv_err = fit.params["rv"].stderr
            specnum = val["spectrum"]
            compnum = val["component_number"]
            component = components[compnum]
            rvs[specnum, 3+compnum*2] = rv_new
            rvs[specnum, 4+compnum*2] = rv_err
            np.savetxt(os.path.join(path, "rv.txt"), rvs, fmt=rvs_fmt)

            vel, bf = val["data"]
            mask = val["mask"]
            bfspath = os.path.join(path, "bf", f"bf{component}_{iterstr}.txt")
            mskpath = os.path.join(path, "bf", f"mask{component}_{iterstr}.txt")
            modpath = os.path.join(path, "bf", f"model{component}_{iterstr}.txt")
            if os.path.exists(bfspath):
                bfs = np.loadtxt(bfspath)
                masks = np.loadtxt(mskpath)
                models = np.loadtxt(modpath)
            else:_fmt
                bfs = np.zeros((n_spectra+1, vel.size), dtype=np.float64)
                bfs[0, :] = vel
                masks = np.zeros((n_spectra+1, vel.size), dtype=np.float64)
                masks[0, :] = vel
                models = np.zeros((n_spectra+1, vel.size), dtype=np.float64)
                models[0, :] = vel
            bfs[specnum+1, :] = bf
            masks[specnum+1, :] = val["mask"]
            models[specnum+1, :] = model
            np.savetxt(bfspath, bfs, header="Row0: Velocity (km/s). RowN: BF_(N-1).")
            np.savetxt(mskpath, masks, header="Row0: Velocity (km/s). RowN: Mask_(N-1).")
            np.savetxt(modpath, models, header="Row0: Velocity (km/s). RowN: BFmodel_(N-1).")

            # Parameters
            if val["fitfun"] == "RotBF":
                header = """
                Rotational broadening function fit.\nCol0: File. Col1: Value. Col2: Formal error\nCol1: Amplitude
                Col3: RV (km/s)\nCol5: Vsini (km/s)\nCol7: Continuum\nCol9: Limb darkening
                """
            elif val["fitfun"] == "Gaussian":
                header = """
                Gaussian broadening function fit.\nCol0: File. Col1: Value. Col2: Formal error\nCol1: Amplitude
                Col3: RV (km/s)\nCol5: Sigma (km/s)\nCol7: Continuum
                """
            elif val["fitfun"] == "SepBF":
                header = """
                Broadening function fit with separated spectrum model.\nCol0: File. Col1: Value Col2: Formal error
                Col1: Amplitude\nCol3: RV (km/s)\nCol5: Continuum
                """
            else:
                header = """
                Broadening function fit with unknown function.\nCol0: File. Col1: Value. Col2: Formal error
                """
            fits_path = os.path.join(path, "bf", f"fits{component}_{iterstr}.txt")
            if os.path.exists(fits_path):
                fits = np.loadtxt(fits_path, dtype=object)
            else:
                fits = np.zeros((n_spectra, len(pvals)*2+1), dtype=object)
                fits[:, 0] = files
            fits_fmt = ["%s"] + ["%.6f" for i in range(2*len(pvals))]
            fits[specnum, 1::2] = pvals
            fits[specnum, 2::2] = np.array([x.stderr for x in fit.params])
            np.savetxt(fits_path, fits, fmt=fits_fmt, header=header)






def plot_process(queue, wavelength, config, filenames, barycorrs, bjds, ordernumber, error_wl_lims):
    path = make_outpath(config["Plotting"]["Folder"], ordernumber, error_wl_lims)
    iteration = 0
    components = config["radialvelocity"]["components"].keys()
    n_components = len(comps)
    n_spectra = len(filenames)
    files = np.array(filenames)
    if config["Plotting"]["Orbit"]:
        fig_orbit, ax_orbit = plt.subplots(2, 1, figsize=(16,12))
    if config["Plotting"]["OrbitalPhasesDetailed"]:
        fig_orbdet, ax_orbdet = plt.subplots(1, 1, figsize=(16,12))
    if config["Plotting"]["bfSpectra"]:
        figs_bf = [plt.figure(1, 1, figsize=(16, 9)) for i in range(n_components)]
        gs_bf = [figs_bf[i].add_gridspec(1, 1) for i in range(n_components)]
        axs_bf = [figs_bf[i].add_subplot(bf_gs[:, :]) for i in range(n_components)]
    if config["Plotting"]["River"]:
        figs_river = [plt.figure(1, 1, figsize=(16, 9)) for i in range(n_components+1)]
        gs_river = [figs_river[i].add_gridspec(1, 1) for i in range(n_components+1)]
        axs_river = [figs_river[i].add_subplot(bf_gs[:, :]) for i in range(n_components+1)]
    if config["Plotting"]["SeparatedSpectra"]:
        fig_sep, axs_sep = plt.subplots(n_components, 1, figsize=(16, 6*n_components))
    if config["Plotting"]["SeparatedDetailed"]:
        fig_sep, axs_sep = plt.subplots(n_components, 1, figsize=(16, 6*n_components))

    while True:
        val = queue.get()
        if val is None: break
        riteration = val["iteration"]

        if val["subroutine"] == "sep":

        elif val["subroutine"] == "rv":
            compnum = val["component_number"]
        else:
            raise ValueError(f"Unknown subroutine {val['subroutine']}.")







# TODO:
# Plotting process.
# Preprocessing functions (big).
# load_preprocessed_data
# Input generation function.
# RV estimate gui function (big).
# Binary RV fit
# Custom code input functionality
