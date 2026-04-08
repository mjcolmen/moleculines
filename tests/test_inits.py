import pytest 
from moleculines.useful import read_spectrum, Spectrum, measure_line_flux_wave,  measure_line_flux_vel
import astropy.units as u
import numpy as np

@pytest.fixture
def csv_spec_path(tmp_path):
    """
    Create a tiny CSV spectrum on disk so read_spectrum(path) can read it.
    Columns: wave, flux
    """
    wave_um = np.linspace(9.95, 10.05, 200)  # um
    # simple Gaussian-ish line on a flat continuum
    flux_jy = 1.0 + 0.2 * np.exp(-0.5 * ((wave_um - 10.0) / 0.005) ** 2)

    p = tmp_path / "spec.csv"
    p.write_text("wave,flux\n" + "\n".join(f"{w},{f}" for w, f in zip(wave_um, flux_jy)))
    return p

@pytest.fixture
def spec_jy_um():
    wave = np.linspace(9.95, 10.05, 200) * u.um
    flux = (1.0 + 0.2 * np.exp(-0.5 * ((wave.to_value(u.um) - 10.0) / 0.005) ** 2)) * u.Jy
    return Spectrum(wave, flux)

def test_read_spectrum(csv_spec_path):
    spec = read_spectrum(
        str(csv_spec_path),
        wave_col="wave",
        flux_col="flux",
        wave_unit=u.um,
        flux_unit=u.Jy,
    )
    assert isinstance(spec, Spectrum)
    assert spec.wave.unit == u.um
    assert spec.flux.unit == u.Jy

def test_integrated_flux_units_wave(spec_jy_um):
    integrated = measure_line_flux_wave(
        spec_jy_um,
        lam0=10.0 * u.um,
        dlam=0.02 * u.um,
        cont=True,
    )
    assert integrated.unit == u.Jy * u.um

def test_integrated_flux_units_vel(spec_jy_um):
    integrated = measure_line_flux_vel(
        spec_jy_um,
        lam0=10.0 * u.um,
        dv=10.0 * (u.km / u.s),
        cont=True,
    )
    assert integrated.unit == u.Jy * (u.km / u.s)	