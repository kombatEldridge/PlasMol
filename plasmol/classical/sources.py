# classical/sources.py
import logging
import meep as mp
import numpy as np

class MEEPSOURCE:
    def __init__(self,
                 source_type,
                 sourceCenter,
                 sourceSize,
                 frequency=None,
                 wavelength=None,
                 component='z',
                 amplitude=1.0,
                 is_integrated=True,
                 **kwargs):
        """
        Initializes a MEEPSOURCE object, which can create Continuous, Gaussian, or Custom sources.

        This class unifies the creation of different source types in Meep. Specify the `source_type`
        as 'continuous', 'gaussian', or 'custom'. Common parameters are shared across types, while
        type-specific parameters are passed via **kwargs.

        Args:
            source_type (str): The type of source ('continuous', 'gaussian', or 'custom').
            sourceCenter (tuple or mp.Vector3): The center coordinates of the source (x, y, z).
            sourceSize (tuple or mp.Vector3): The size dimensions of the source (sx, sy, sz).
            frequency (float, optional): The center frequency of the source (in Meep units: c/distance).
            wavelength (float, optional): The center wavelength in microns (alternative to frequency).
            component (str, optional): The electric field component ('x', 'y', or 'z'). Default: 'z'.
            amplitude (complex, optional): Overall complex amplitude multiplying the source. Default: 1.0.
            is_integrated (bool, optional): If True, integrates the source over time (dipole moment).
                Default: True (note: Meep's default is False; this follows previous implementation).
            **kwargs: Type-specific parameters (see below).

        Type-Specific Parameters (**kwargs):
            - For 'continuous':
                - start_time (float, optional): Starting time. Default: 0.
                - end_time (float, optional): End time. Default: 1e+20.
                - width (float, optional): Temporal width of smoothing. Default: 0.
                - fwidth (float, optional): Frequency width (synonym for 1/width). Default: float('inf').
                - slowness (float, optional): Controls gradual turn-on. Default: 3.0.

            - For 'gaussian':
                - start_time (float, optional): Starting time. Default: 0.
                - cutoff (float, optional): Number of widths before cutoff. Default: 5.0.
                - width (float, optional): Temporal width. Default: 0 (but typically set >0).
                - fwidth (float, optional): Frequency width (synonym for 1/width). Default: float('inf').

            - For 'custom':
                - src_func (callable, required): Function f(t) returning complex number for time t (in Meep units).
                - start_time (float, optional): Starting time. Default: -1e+20.
                - end_time (float, optional): End time. Default: 1e+20.
                - fwidth (float, optional): Bandwidth in frequency units. Default: 0.

        Raises:
            ValueError: If neither frequency nor wavelength is provided (for 'continuous' or 'gaussian'),
                        if source_type is invalid, or if required kwargs are missing (e.g., src_func for 'custom').

        Notes on Replicating Previous "Chirped" and "Pulse" Sources:
            The previous ChirpedSource and PulseSource were custom implementations. You can replicate
            them using source_type='custom' by defining an appropriate src_func. Note that these used
            a specific conversion factor (3.378555833184493) to handle units (e.g., fs to Meep time).
            Adjust your func accordingly if working in specific units.

            Example for Pulse (non-chirped):
                conversion_factor = 3.378555833184493
                frequency_meep = (1 / wavelength if wavelength else frequency) * 2.99792458e8 * 1e6 * conversion_factor * 1e-15
                peak_time_meep = peakTime * (1 / conversion_factor)  # peakTime in fs
                width_meep = width * (conversion_factor ** 2)  # width in 1/fs**2
                def pulse_func(t):
                    return np.exp(1j * 2 * np.pi * frequency_meep * (t - peak_time_meep)) * np.exp(-width_meep * (t - peak_time_meep) ** 2)
                # Then: MEEPSOURCE(source_type='custom', ..., src_func=pulse_func)

            Example for Chirped:
                # Similar to above, but add chirp term:
                chirp_rate_meep = chirpRate * (conversion_factor ** 2)  # chirpRate in 1/fs**2
                def chirped_func(t):
                    return np.exp(1j * 2 * np.pi * frequency_meep * (t - peak_time_meep)) * np.exp(-width_meep * (t - peak_time_meep) ** 2 + 1j * chirp_rate_meep * (t - peak_time_meep) ** 2)
                # Then: MEEPSOURCE(source_type='custom', ..., src_func=chirped_func)

            For more details on CustomSource, see Meep documentation: https://meep.readthedocs.io/en/latest/Python_User_Interface/#customsource

        """
        self.source_type = source_type.lower().strip()
        logging.debug(f"Initializing MEEPSOURCE with type: {self.source_type}")

        # Common parameter processing
        if frequency is None and wavelength is None and self.source_type != 'custom':
            raise ValueError(f"Either 'frequency' or 'wavelength' must be provided for {self.source_type} source.")

        char_to_field = {'x': mp.Ex, 'y': mp.Ey, 'z': mp.Ez}
        try:
            self.component = char_to_field[component.lower().strip()]
        except KeyError:
            raise ValueError(f"Invalid component '{component}'; must be 'x', 'y', or 'z'.")

        self.sourceCenter = mp.Vector3(*sourceCenter) if not isinstance(sourceCenter, mp.Vector3) else sourceCenter
        self.sourceSize = mp.Vector3(*sourceSize) if not isinstance(sourceSize, mp.Vector3) else sourceSize

        # Compute frequency if wavelength is provided
        freq = (1 / wavelength if wavelength is not None else frequency) if self.source_type != 'custom' else None

        # Type-specific source time creation
        if self.source_type == 'continuous':
            start_time = kwargs.get('start_time', 0)
            end_time = kwargs.get('end_time', 1e+20)
            width = kwargs.get('width', 0)
            fwidth = kwargs.get('fwidth', float('inf'))
            slowness = kwargs.get('slowness', 3.0)
            if fwidth not in [None, float('inf')]:
                width = max(width, 1 / fwidth)
            src_time = mp.ContinuousSource(
                frequency=freq,
                start_time=start_time,
                end_time=end_time,
                width=width,
                slowness=slowness,
                is_integrated=is_integrated
            )

        elif self.source_type == 'gaussian':
            start_time = kwargs.get('start_time', 0)
            cutoff = kwargs.get('cutoff', 5.0)
            width = kwargs.get('width', 0)
            fwidth = kwargs.get('fwidth', float('inf'))
            if fwidth not in [None, float('inf')]:
                width = max(width, 1 / fwidth)
            src_time = mp.GaussianSource(
                frequency=freq,
                width=width,
                start_time=start_time,
                cutoff=cutoff,
                is_integrated=is_integrated
            )

        elif self.source_type == 'custom':
            src_func = kwargs.get('src_func')
            if src_func is None:
                raise ValueError("For 'custom' source_type, 'src_func' must be provided in kwargs.")
            start_time = kwargs.get('start_time', -1e+20)
            end_time = kwargs.get('end_time', 1e+20)
            fwidth = kwargs.get('fwidth', 0)
            src_time = mp.CustomSource(
                src_func=src_func,
                start_time=start_time,
                end_time=end_time,
                is_integrated=is_integrated,
                fwidth=fwidth
            )

        else:
            raise ValueError(f"Invalid source_type '{self.source_type}'; must be 'continuous', 'gaussian', or 'custom'.")

        # Create the final Meep Source object
        self.source = mp.Source(
            src=src_time,
            component=self.component,
            center=self.sourceCenter,
            size=self.sourceSize,
            amplitude=amplitude
        )


# ------------------------------------------ #
#             Additional custom              #
#            classes for sources             #
#             can be added here              #
#           and need to be added to          #
#   the `utils/input/params.py` file under   #
#          the `getSource()` method          #
# ------------------------------------------ #