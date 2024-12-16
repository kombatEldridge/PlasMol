import logging
import numpy as np
import meep as mp

class ContinuousSource:
    def __init__(self,
                 sourceCenter,
                 sourceSize,
                 frequency=None,
                 start_time=0,
                 end_time=1e+20,
                 width=0,
                 fwidth=float("inf"),
                 slowness=3.0,
                 wavelength=None,
                 is_integrated=True,
                 component=mp.Ez):
        """
        Initializes a ContinuousSource object.

        Args:
            frequency (float): The frequency of the continuous source.
            sourceCenter (tuple): The center coordinates of the source.
            sourceSize (tuple): The size dimensions of the source.
            is_integrated (bool): If True, integrates the source over time.
            component (mp.Vector3): The component of the electric field for the source.
        """
        logging.debug(f"Initializing ContinuousSource")
        self.sourceCenter = mp.Vector3(sourceCenter)
        self.sourceSize = mp.Vector3(sourceSize[0], sourceSize[1], sourceSize[2])
        kwargs = {
            'frequency': 1 / wavelength if wavelength else float(frequency),
            'is_integrated': is_integrated,
            'start_time': start_time,
            'end_time': end_time,
            'width': max(width, 1 / fwidth) if fwidth not in [None, float("inf")] else width,
            'slowness': slowness,
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.source = mp.Source(
            mp.ContinuousSource(**filtered_kwargs),
            component=component,
            center=self.sourceCenter,
            size=self.sourceSize,
        )


class GaussianSource:
    def __init__(self,
                 sourceCenter,
                 sourceSize,
                 frequency=None,
                 width=0,
                 fwidth=float("inf"),
                 start_time=0,
                 cutoff=5.0,
                 is_integrated=True,
                 wavelength=None,
                 component=mp.Ez):
        """
        Initializes a GaussianSource object.

        Args:
            frequencyCenter (float): The center frequency of the Gaussian source.
            width (float): The width of the Gaussian source in wavelengths.
            fwidth (float): The width of the Gaussian source in frequency.
            sourceCenter (tuple): The center coordinates of the source.
            sourceSize (tuple): The size dimensions of the source.
            is_integrated (bool): If True, integrates the source over time.
            component (mp.Vector3): The component of the electric field for the source.
        """
        logging.debug(f"Initializing GaussianSource")
        self.sourceCenter = mp.Vector3(sourceCenter)
        self.sourceSize = mp.Vector3(sourceSize[0], sourceSize[1], sourceSize[2])
        kwargs = {
            'frequency': 1 / wavelength if wavelength else float(frequency),
            'width': max(width, 1 / fwidth) if fwidth not in [None, float("inf")] else width,
            'start_time': start_time,
            'cutoff': cutoff,
            'is_integrated': is_integrated,
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.source = mp.Source(
            mp.GaussianSource(**filtered_kwargs),
            component=component,
            center=self.sourceCenter,
            size=self.sourceSize,
        )


class ChirpedSource:
    def __init__(self,
                 sourceCenter,
                 sourceSize,
                 frequency=1,
                 wavelength=None,
                 width=0.2,
                 peakTime=15,
                 chirpRate=-0.5,
                 start_time=-1e20,
                 end_time=1e20,
                 is_integrated=True,
                 component=mp.Ez):
        """
        Initializes a ChirpedSource object.

        Args:
            frequency (float): The frequency of the pulse source in units of 1/wavelength.
            wavelength (float): The wavelength of the pulse in microns.
            width (float): The width of the pulse source in 1/fs**2.
            peakTime (float): The time at which the pulse is maximum in fs.
            sourceCenter (tuple): The center coordinates of the source.
            sourceSize (tuple): The size dimensions of the source.
            is_integrated (bool): If True, integrates the source over time.
            component (mp.Vector3): The component of the electric field for the source.
        """
        logging.debug(f"Initializing ChirpedSource")
        self.conversionFactor = 3.378555833184493 # from many tests in PulseTest
        self.sourceCenter = mp.Vector3(sourceCenter)
        self.sourceSize = mp.Vector3(sourceSize[0], sourceSize[1], sourceSize[2])
        kwargs = {
            'start_time': start_time,
            'end_time': end_time,
            'center_frequency': center_frequency,
            'fwidth': fwidth,
            'is_integrated': is_integrated,
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}

        frequency = 1/wavelength if wavelength else (frequency * 2.99792458e8 * 1e6 * self.conversionFactor * 1e-15)
        peakTime = peakTime * (1/self.conversionFactor)
        width = width * (self.conversionFactor**2)
        chirpRate = chirpRate * (self.conversionFactor**2)
        func = lambda t: np.exp(1j * 2 * np.pi * frequency * (t - peakTime)) * np.exp(-width * (t - peakTime) ** 2 + 1j * chirpRate * (t - peakTime) ** 2)

        self.source = mp.Source(
            src=mp.CustomSource(src_func=func, **filtered_kwargs),
            component=component,
            center=self.sourceCenter,
            size=self.sourceSize,
        )

class PulseSource:
    def __init__(self,
                 sourceCenter,
                 sourceSize,
                 frequency=0.6,
                 wavelength=None,
                 width=0.01,
                 peakTime=30,
                 start_time=-1e20,
                 end_time=1e20,
                 is_integrated=True,
                 component=mp.Ez):
        """
        Initializes a PulseSource object.

        Args:
            frequency (float): The frequency of the pulse source in units of 1/wavelength.
            wavelength (float): The wavelength of the pulse in microns.
            width (float): The width of the pulse source in 1/fs**2.
            peakTime (float): The time at which the pulse is maximum in fs.
            sourceCenter (tuple): The center coordinates of the source.
            sourceSize (tuple): The size dimensions of the source.
            is_integrated (bool): If True, integrates the source over time.
            component (mp.Vector3): The component of the electric field for the source.
        """
        logging.debug(f"Initializing PulseSource")
        self.conversionFactor = 3.378555833184493 # from many tests in PulseTest
        self.sourceCenter = mp.Vector3(sourceCenter)
        self.sourceSize = mp.Vector3(sourceSize[0], sourceSize[1], sourceSize[2])
        kwargs = {
            'start_time': start_time,
            'end_time': end_time,
            'is_integrated': is_integrated,
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        frequency = 1/wavelength if wavelength else (frequency * 2.99792458e8 * 1e6 * self.conversionFactor * 1e-15)
        peakTime = peakTime * (1/self.conversionFactor)
        width = width * (self.conversionFactor**2)
        func = lambda t: np.exp(1j * 2 * np.pi * frequency * (t - peakTime)) * np.exp(-width * (t - peakTime) ** 2)

        self.source = mp.Source(
            src=mp.CustomSource(src_func=func, **filtered_kwargs),
            component=component,
            center=self.sourceCenter,
            size=self.sourceSize,
        )







