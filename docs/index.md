# Introduction

This python package is designed for the purpose of performing spectral separation and high precision radial velocity measurements of SB2 binary stellar systems. Its functionality is developed from the IDL code used in [Brogaard et al. 2018](https://academic.oup.com/mnras/article/476/3/3729/4833696), which performs the spectral separation method described in [González & Levato 2006](https://www.aanda.org/articles/aa/abs/2006/10/aa3177-05/aa3177-05.html) along with the broadening function method described in [Rucinski 1992](http://astro.utoronto.ca/~rucinski/manscr/CFHT92.pdf) and [Rucinski 2002](http://astro.utoronto.ca/~rucinski/manscr/bin_pub7.pdf) for radial velocity measurements.

## Table of Contents
1. [Installation](installation)
2. [The Spectral Separation Routine](quickstart)
3. [Routine options](routine_options)
4. [Accessing outputs](results)
5. [Evaluating output](evaluate)
6. [Preparing spectra](preparation)

Modules:

- spectral_separation_routine
- broadening_function_svd
- spectral_processing_functions
- storage_classes
- calculate_radial_velocities
- rotational_broadening_function_fitting
- linear_limbd_coeff_estimate
- evaluate_ssr_results


A script utilizing the modules of this package to perform radial velocity measurements on KIC8430105 can be found at [github](https://github.com/jsinkbaek/sb2sep/blob/main/test/kic8430105/RV_from_spectra_kic8430105.py)


[Next page: Installation](installation)

