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
                 fwidth=np.inf,
                 cutoff=None,
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
            'frequency': frequency,
            'is_integrated': is_integrated,
            'start_time': start_time,
            'end_time': end_time,
            'width': width,
            'fwidth': fwidth,
            'cutoff': cutoff,
            'slowness': slowness,
            'wavelength': wavelength,
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.source = mp.Source(
            mp.ContinuousSource(**filtered_kwargs),
            center=self.sourceCenter,
            size=self.sourceSize,
            component=component
        )


class GaussianSource:
    def __init__(self,
                 sourceCenter,
                 sourceSize,
                 frequencyCenter=None,
                 frequencyWidth=0,
                 fwidth=np.inf,
                 start_time=0,
                 cutoff=5.0,
                 is_integrated=True,
                 wavelength=None,
                 component=mp.Ez):
        """
        Initializes a GaussianSource object.

        Args:
            frequencyCenter (float): The center frequency of the Gaussian source.
            frequencyWidth (float): The width of the Gaussian source in frequency.
            sourceCenter (tuple): The center coordinates of the source.
            sourceSize (tuple): The size dimensions of the source.
            is_integrated (bool): If True, integrates the source over time.
            component (mp.Vector3): The component of the electric field for the source.
        """
        logging.debug(f"Initializing GaussianSource")
        self.sourceCenter = mp.Vector3(sourceCenter)
        self.sourceSize = mp.Vector3(sourceSize[0], sourceSize[1], sourceSize[2])
        kwargs = {
            'frequency': frequencyCenter,
            'width': frequencyWidth,
            'fwidth': fwidth,
            'start_time': start_time,
            'cutoff': cutoff,
            'is_integrated': is_integrated,
            'wavelength': wavelength,
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.source = mp.Source(
            mp.GaussianSource(**filtered_kwargs),
            center=self.sourceCenter,
            size=self.sourceSize,
            component=component
        )

