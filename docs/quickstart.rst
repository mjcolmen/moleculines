Quick Start
===========

Basic Usage
-----------

Load a spectrum from a file:

.. code-block:: python

    from moleculines import read_spectrum

    spec = read_spectrum("spectrum.csv",
                         wave_col="wavelength",
                         flux_col="flux",
                         unc_col="unc")

Plot the spectrum:

.. code-block:: python

    spec.plot()

Convert between wavelength and velocity:

.. code-block:: python

    from moleculines import wave_to_vel

    vel = wave_to_vel(spec.wave, lam0=15.0)  # microns

Measure a line flux:

.. code-block:: python

    from moleculines import measure_line_flux_wave
    import astropy.units as u

    flux = measure_line_flux_wave(spec,
                                 lam0=15.0*u.um,
                                 dlam=0.05*u.um)

Working with Units
------------------

All quantities are handled using ``astropy.units``. Inputs without units
are automatically converted into `Quantity` objects, but it is recommended
to provide units explicitly for clarity and correctness.

Example:

.. code-block:: python

    import astropy.units as u

    lam0 = 15.0 * u.um
    dv   = 20.0 * u.km / u.s

    flux = measure_line_flux_vel(spec, lam0, dv)

