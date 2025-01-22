import numpy as np
import scipy.interpolate as interp


def estimate_photometric_system(wavelength_range: np.ndarray or tuple):
    """
    Estimates which photometric system is appropriate to look up linear limb-darkening coefficients for.
    Wavelengths in Ångstrom units. Assumes passbands symmetric and simple, and follows peak wvl and fwhm from
    https://en.wikipedia.org/wiki/Photometric_system and
    https://en.wikipedia.org/wiki/Str%C3%B6mgren_photometric_system
    :param wavelength_range: np.ndarray([wavelength_a, wavelength_b]) limits on the wavelength range of spectrum.
    :return: string, pass-band indicator
    """
    passband_names = np.array(['u', 'v', 'b', 'y', 'U', 'B', 'V', 'R', 'I', 'J', 'H', 'K'])
    peak_wavelength = np.array([3500, 4110, 4670, 5470, 3650, 4450, 5510, 6580, 8060, 12200, 16300, 21900])
    fwhm = np.array([300, 190, 180, 230, 660, 940, 880, 1380, 1490, 2130, 3070, 3900])

    bands_wla = peak_wavelength - fwhm
    bands_wlb = peak_wavelength + fwhm

    diff_wla = np.abs(wavelength_range[0] - bands_wla)
    diff_wlb = np.abs(wavelength_range[1] - bands_wlb)

    best_pasband_idx = np.argmin(diff_wla + diff_wlb)
    return passband_names[best_pasband_idx]


def read_linear_limbd_param(logg_range=np.array([0, 7]), Trange=np.array([2000, 9000]), MH_range=-0.5, mTurb_range=2.0,
                            loc='../Data/tables/atlasco.dat', band='V'):
    if 'atlasco.dat' in loc:
        if band=='u': band_col=5
        elif band=='v': band_col=6
        elif band=='b': band_col=7
        elif band=='y': band_col=8
        elif band=='U': band_col=9
        elif band=='B': band_col=10
        elif band=='V': band_col=11
        elif band=='R': band_col=12
        elif band=='I': band_col=13
        elif band=='J': band_col=14
        elif band=='H': band_col=15
        elif band=='K': band_col=16
        else:
            raise ValueError(f'band variable {band} is in an incorrect format.')

        data = np.loadtxt(loc, usecols=(1, 2, 3, 4, band_col))
        mTurb= data[:, 0]
        logg = data[:, 1]
        Teff = data[:, 2]
        MH   = data[:, 3]

        if isinstance(MH_range, np.ndarray):
            MH_mask = (MH >= MH_range[0]) & (MH <= MH_range[1])
        else:
            MH_mask = MH == MH_range
            data = np.delete(data, 3, 1)
        if isinstance(Trange, np.ndarray):
            T_mask = (Teff >= Trange[0]) & (Teff <= Trange[1])
        else:
            T_mask = Teff == Trange
            data = np.delete(data, 2, 1)
        if isinstance(logg_range, np.ndarray):
            lg_mask = (logg >= logg_range[0]) & (logg <= logg_range[1])
        else:
            lg_mask = logg == logg_range
            data = np.delete(data, 1, 1)
        if isinstance(mTurb_range, np.ndarray):
            mT_mask = (mTurb >= mTurb_range[0]) & (mTurb <= mTurb_range[1])
        else:
            mT_mask = mTurb == mTurb_range
            data = np.delete(data, 0, 1)  # delete column with singular data value

        mask = lg_mask & T_mask & MH_mask & mT_mask
    else:
        raise IOError("Unknown datafile structure or wrongly defined location.")

    data = data[mask, :]
    return data


def interpolate_linear_limbd(
        logg: float, Teff: float, MH: float, mTurb: float, logg_range=np.array([0, 7]), Trange=np.array([2000, 9000]),
        MH_range=-0.5, mTurb_range=2.0, loc='../Data/tables/atlasco.dat', band='V'
):
    data = read_linear_limbd_param(logg_range, Trange, MH_range, mTurb_range, loc, band)
    vals = data[:, -1]
    points = data[:, 0:-1]
    eval_points = np.array([])
    # # Flexibility depending on size of parameter space (how many have grid points) and table structure used
    if 'atlasco.dat' in loc:
        if isinstance(mTurb_range, np.ndarray):
            eval_points = np.append(eval_points, mTurb)
        if isinstance(logg_range, np.ndarray):
            eval_points = np.append(eval_points, logg)
        if isinstance(Trange, np.ndarray):
            eval_points = np.append(eval_points, Teff)
        if isinstance(MH_range, np.ndarray):
            eval_points = np.append(eval_points, MH)
    else:
        raise IOError("Unknown datafile structure or wrongly defined location.")

    eval_points = np.reshape(eval_points, (1, eval_points.size))
    res = interp.griddata(points, vals, eval_points, method='cubic')

    return res[0]


def estimate_linear_limbd(
        wavelength_range: np.ndarray or tuple, logg: float, T: float, MH: float, mTurb: float,
        logg_range=np.array([0, 7]), Trange=np.array([2000, 9000]), MH_range=-0.5, mTurb_range=2.0,
        loc='../Data/tables/atlasco.dat', print_bool=False
):
    """
    Estimates linear limb-darkening coefficient by interpolating table values by A. Claret 2000
    https://ui.adsabs.harvard.edu/abs/2000A%26A...363.1081C/abstract
    Default table is the atlasco.dat table from https://cdsarc.unistra.fr/viz-bin/cat/J/A+A/363/1081
    Chooses an approximate photometric system by comparing wavelength range with photometric FWHM around effective max.
    """
    photometric_system = estimate_photometric_system(wavelength_range)
    linear_limbd_coef = interpolate_linear_limbd(logg, T, MH, mTurb, logg_range, Trange, MH_range, mTurb_range,
                                                 loc, photometric_system)
    if print_bool:
        print('Approximate Photometric system: ', photometric_system)
        print(linear_limbd_coef)
    return linear_limbd_coef


def main():
    estimate_linear_limbd((5300, 5700), 2.78, 5042, -0.49, 2.0, print_bool=True)
    estimate_linear_limbd((5300, 5700), 4.58, 5621, -0.49, 2.0, print_bool=True)


if __name__ == "__main__":
    main()
