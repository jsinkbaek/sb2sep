from numpy import interp
from scipy.interpolate import Akima1DInterpolator


def shift_wavelength_spectrum(wavelength, flux, radial_velocity_shift, method="linear", fill_value=1.0):
    wavelength_shifted = wavelength * (1 + radial_velocity_shift/(scc.speed_of_light / 1000))
    if method == "linear":
        flux_shifted = interp(wavelength, wavelength_shifted, flux)
    elif method == "makima":
        wavelength_shifted = wavelength * (1 + radial_velocity_shift / (scc.speed_of_light / 1000))
        invalid_region = ((wavelength_shifted > np.max(wavelength)) | (wavelength_shifted < np.min(wavelength)) |
                          (wavelength > np.max(wavelength_shifted)) | (wavelength < np.min(wavelength_shifted)))
        flux_shifted = Akima1DInterpolator(wavelength_shifted, flux, method="makima")(wavelength, extrapolate=True)
        flux_shifted[invalid_region] = fill_value
    else:
        raise ValueError(f"Unknown shift interpolation method {method}")

    return flux_shifted