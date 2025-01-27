#!../../.venv/bin/python
import fileinput
import glob
import sys


def generate_inputfiles(fundemental_setup, *fileglobs):
    pass


def resample_normalize_barycorr(input_file, config_file):
    pass


def estimate_radial_velocity(input_file, config_file):
    pass


def perform_spectral_separation(input_file, config_file):
    pass


def perform_binary_rv_fit(input_file, config_file):
    pass


if __name__ == "__main__":
    inpts = list(fileinput.input())
    procedure = inpts[0]
    if procedure == "geninput":
        setup = inpts[1]
        generate_inputfiles(setup, *inpts[2:])
        sys.exit(0)
    elif procedure == "limbd":
        from .linear_limbd_coeff_estimate import estimate_linear_limbd
        wla, wlb, logg, teff, feh, vmicro = inpts[1:]
        print(f"Linear limb darkening estimate: {estimate_linear_limbd((wla, wlb), logg, teff, feh, vmicro):.4f}")
        sys.exit(0)
    else:
        confile = inpts[1]
        infile = inpts[2]
    if procedure == "preprocess":
        resample_normalize_barycorr(infile, confile)
    elif procedure == "estimaterv":
        estimate_radial_velocity(infile, confile)
    elif procedure == "binaryrv":
        perform_binary_rv_fit(infile, confile)
    elif procedure == "run":
        perform_spectral_separation(infile, confile)
    else:
        raise NotImplementedError(f"Unknown procedure {procedure}")
