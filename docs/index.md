# Introduction

This python package is designed with the goal of performing spectral separation and radial velocity calculation for SB2 binary systems. Its functionality is developed from the IDL code used in [Brogaard et al. 2018](https://academic.oup.com/mnras/article/476/3/3729/4833696), which performs the spectral separation method described in [González & Levato 2006](https://www.aanda.org/articles/aa/abs/2006/10/aa3177-05/aa3177-05.html) along with the broadening function method described in [Rucinski 1992](http://astro.utoronto.ca/~rucinski/manscr/CFHT92.pdf) and [Rucinski 2002](http://astro.utoronto.ca/~rucinski/manscr/bin_pub7.pdf) for radial velocity measurements.

## Table of Contents
1. [The Spectral Separation Routine](quickstart)
2. [Routine options](routine_options)
3. [Accessing outputs](results)
4. [Evaluating output](evaluate)

Modules:

- [spectral_separation_routine](ssr)
- [broadening_function_svd](bfsvd)
- [spectral_processing_functions](spf)
- [storage_classes](storage_classes)
- [calculate_radial_velocities](calcRV)
- [rotational_broadening_function_fitting](rotbf)
- [linear_limbd_coeff_estimate](limbd)
- [evaluate_ssr_results](evaluate)


A script utilizing the modules of this package to perform radial velocity measurements on KIC8430105 can be found at [github](https://github.com/jsinkbaek/sb2sep/blob/main/test/kic8430105/RV_from_spectra_kic8430105.py)


[``` Next page: The Spectral Separation Routine```](quickstart)

