import numpy as np
import astropy.units as u
from astropy.constants import c
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt

class Spectrum:
    def __init__(self, wave, flux, unc=None):
        self.wave = u.Quantity(wave)
        self.flux = u.Quantity(flux)
        self.unc  = None if unc is None else u.Quantity(unc)
        
    def plot(self, ax=None, label=None, color=None, lw=1.5,
             show_unc=True, alpha=0.25):
        """Plot wave vs flux (+ optional uncertainty shading)."""
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
    wave = u.Quantity(wave); lam0 = u.Quantity(lam0).to(wave.unit)
    return ((wave - lam0) / lam0 * c).to(u.km/u.s)

def vel_to_wave(vel, lam0):
    vel = u.Quantity(vel).to(u.km/u.s); lam0 = u.Quantity(lam0)
    return lam0 * (1 + (vel/c).to_value(u.dimensionless_unscaled))

def shift_to_rest_frame(spec, v_sys):
    return Spectrum(vel_to_wave(-u.Quantity(v_sys), spec.wave), spec.flux, spec.unc)

def read_spectrum(path, wave_col="wave", flux_col="flux", unc_col=None,
                  wave_unit=u.um, flux_unit=u.dimensionless_unscaled):
    """
    Reads:
      - .csv/.txt (table with columns)
      - .fits (binary table; columns)
    """
    if path.lower().endswith((".fits", ".fit", ".fts")):
        tab = Table.read(path)  # usually reads first table HDU
    else:
        tab = Table.read(path, format="ascii.csv")  # CSV

    w = np.array(tab[wave_col], dtype=float) * wave_unit
    f = np.array(tab[flux_col], dtype=float) * flux_unit
    if unc_col is None:
        return Spectrum(w, f)
    uarr = np.array(tab[unc_col], dtype=float) * flux_unit
    return Spectrum(w, f, uarr)



def measure_line_flux_wave(spec, lam0, dlam, cont=True):
    m = (spec.wave > lam0-dlam) & (spec.wave < lam0+dlam)
    w = spec.wave[m].to_value(spec.wave.unit)
    f = spec.flux[m].to_value(spec.flux.unit)
    if cont: f = f - np.median(f)
    return np.trapz(f, w) * spec.flux.unit * spec.wave.unit

def measure_line_flux_vel(spec, lam0, dv, cont=True):
    v = wave_to_vel(spec.wave, lam0)
    m = (v > -dv) & (v < dv)
    vv = v[m].to_value(u.km/u.s)
    f  = spec.flux[m].to_value(spec.flux.unit)
    if cont: f = f - np.median(f)
    return np.trapz(f, vv) * spec.flux.unit * (u.km/u.s)
