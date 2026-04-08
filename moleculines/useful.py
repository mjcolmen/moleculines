import numpy as np
import astropy.units as u
from astropy.constants import c
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt

class Spectrum:
    """
    Container for a 1D spectrum with wavelength, flux, and optional uncertainties.

    Parameters
    ----------
    wave : array-like or `astropy.units.Quantity`
        Wavelength array.
    flux : array-like or `astropy.units.Quantity`
        Flux array corresponding to ``wave``.
    unc : array-like or `astropy.units.Quantity`, optional
        Uncertainty on the flux values. If None, no uncertainties are stored.

    Attributes
    ----------
    wave : `astropy.units.Quantity`
        Wavelength array.
    flux : `astropy.units.Quantity`
        Flux array.
    unc : `astropy.units.Quantity` or None
        Flux uncertainties.
    """

    def __init__(self, wave, flux, unc=None):
        self.wave = u.Quantity(wave)
        self.flux = u.Quantity(flux)
        self.unc  = None if unc is None else u.Quantity(unc)
        
    def plot(self, ax=None, label=None, color=None, lw=1.5,
             show_unc=True, alpha=0.25):
        """
        Plot the spectrum.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`, optional
            Axis to plot on. If None, a new figure and axis are created.
        label : str, optional
            Label for the plotted spectrum.
        color : str, optional
            Line color.
        lw : float, optional
            Line width.
        show_unc : bool, optional
            If True and uncertainties are available, plot shaded uncertainty region.
        alpha : float, optional
            Transparency of the uncertainty shading.

        Returns
        -------
        ax : `matplotlib.axes.Axes`
            Axis with the plotted spectrum.
        """
        if ax is None:
            fig, ax = plt.subplots()

        x = self.wave.to_value(self.wave.unit)
        y = self.flux.to_value(self.flux.unit)

        ax.plot(x, y, label=label, color=color, lw=lw)

        if show_unc and (self.unc is not None):
            e = self.unc.to_value(self.flux.unit)
            ax.fill_between(x, y - e, y + e, color=color, alpha=alpha, linewidth=0)

        ax.set_xlabel(f"Wavelength [{self.wave.unit}]")
        ax.set_ylabel(f"Flux [{self.flux.unit}]")
        if label is not None:
            ax.legend(frameon=False)

        return ax


def wave_to_vel(wave, lam0):
    """
    Convert wavelength to velocity using the Doppler relation.

    Parameters
    ----------
    wave : array-like or `astropy.units.Quantity`
        Observed wavelength(s).
    lam0 : float or `astropy.units.Quantity`
        Rest wavelength.

    Returns
    -------
    vel : `astropy.units.Quantity`
        Velocity relative to ``lam0`` in km/s.
    """
    wave = u.Quantity(wave)
    lam0 = u.Quantity(lam0).to(wave.unit)
    return ((wave - lam0) / lam0 * c).to(u.km/u.s)


def vel_to_wave(vel, lam0):
    """
    Convert velocity to wavelength using the Doppler relation.

    Parameters
    ----------
    vel : array-like or `astropy.units.Quantity`
        Velocity offset(s).
    lam0 : float or `astropy.units.Quantity`
        Rest wavelength.

    Returns
    -------
    wave : `astropy.units.Quantity`
        Doppler-shifted wavelength(s).
    """
    vel = u.Quantity(vel).to(u.km/u.s)
    lam0 = u.Quantity(lam0)
    return lam0 * (1 + (vel/c).to_value(u.dimensionless_unscaled))


def shift_to_rest_frame(spec, v_sys):
    """
    Shift a spectrum to the rest frame.

    Parameters
    ----------
    spec : `Spectrum`
        Input spectrum.
    v_sys : float or `astropy.units.Quantity`
        Systemic velocity (positive = redshift).

    Returns
    -------
    spec_rest : `Spectrum`
        Spectrum shifted to the rest frame.
    """
    return Spectrum(vel_to_wave(-u.Quantity(v_sys), spec.wave), spec.flux, spec.unc)


def read_spectrum(path, wave_col="wave", flux_col="flux", unc_col=None,
                  wave_unit=u.um, flux_unit=u.dimensionless_unscaled):
    """
    Read a spectrum from a file.

    Supports CSV/text tables and FITS binary tables.

    Parameters
    ----------
    path : str
        Path to the input file.
    wave_col : str, optional
        Name of the wavelength column.
    flux_col : str, optional
        Name of the flux column.
    unc_col : str, optional
        Name of the uncertainty column. If None, no uncertainties are read.
    wave_unit : `astropy.units.Unit`, optional
        Unit to assign to the wavelength column.
    flux_unit : `astropy.units.Unit`, optional
        Unit to assign to the flux column.

    Returns
    -------
    spec : `Spectrum`
        Loaded spectrum.
    """
    if path.lower().endswith((".fits", ".fit", ".fts")):
        tab = Table.read(path)
    else:
        tab = Table.read(path, format="ascii.csv")

    w = np.array(tab[wave_col], dtype=float) * wave_unit
    f = np.array(tab[flux_col], dtype=float) * flux_unit

    if unc_col is None:
        return Spectrum(w, f)

    uarr = np.array(tab[unc_col], dtype=float) * flux_unit
    return Spectrum(w, f, uarr)


def measure_line_flux_wave(spec, lam0, dlam, cont=True):
    """
    Measure line flux in wavelength space.

    Parameters
    ----------
    spec : `Spectrum`
        Input spectrum.
    lam0 : `astropy.units.Quantity`
        Line center wavelength.
    dlam : `astropy.units.Quantity`
        Half-width of the integration window.
    cont : bool, optional
        If True, subtract median continuum before integration.

    Returns
    -------
    flux : `astropy.units.Quantity`
        Integrated line flux.
    """
    m = (spec.wave > lam0-dlam) & (spec.wave < lam0+dlam)
    w = spec.wave[m].to_value(spec.wave.unit)
    f = spec.flux[m].to_value(spec.flux.unit)

    if cont:
        f = f - np.median(f)

    return np.trapz(f, w) * spec.flux.unit * spec.wave.unit


def measure_line_flux_vel(spec, lam0, dv, cont=True):
    """
    Measure line flux in velocity space.

    Parameters
    ----------
    spec : `Spectrum`
        Input spectrum.
    lam0 : `astropy.units.Quantity`
        Line center wavelength.
    dv : `astropy.units.Quantity`
        Half-width of the velocity window.
    cont : bool, optional
        If True, subtract median continuum before integration.

    Returns
    -------
    flux : `astropy.units.Quantity`
        Integrated line flux in velocity space.
    """
    v = wave_to_vel(spec.wave, lam0)
    m = (v > -dv) & (v < dv)

    vv = v[m].to_value(u.km/u.s)
    f  = spec.flux[m].to_value(spec.flux.unit)

    if cont:
        f = f - np.median(f)

    return np.trapz(f, vv) * spec.flux.unit * (u.km/u.s)
