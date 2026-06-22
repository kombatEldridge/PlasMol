# classical/sources.py
import logging
import meep as mp
import numpy as np
from plasmol.utils import constants

class MEEPSOURCE:
    def __init__(self,
                 source_type,
                 source_center,
                 source_size,
                 component,
                 is_integrated,
                 amplitude=1,
                 **kwargs):
        """
        Initializes a MEEPSOURCE object, which can create Continuous, Gaussian, or Custom sources.

        This class unifies the creation of different source types in Meep. Specify the `source_type`
        as 'continuous', 'gaussian', or <custom>. Common parameters are shared across types, while
        type-specific parameters are passed via **kwargs.

        Args:
            source_type (str): The type of source ('continuous', 'gaussian', or <custom>).
            source_center (tuple or mp.Vector3): The center coordinates of the source (x, y, z).
            source_size (tuple or mp.Vector3): The size dimensions of the source (sx, sy, sz).
            component (str, optional): The electric field component ('x', 'y', or 'z'). Default: 'z'.
            amplitude (complex, optional): Overall complex amplitude multiplying the source. Default: 1.0.
            is_integrated (bool, optional): If True, integrates the source over time (dipole moment).
                Default: True (note: Meep's default is False; this follows previous implementation).
            **kwargs: Type-specific parameters (see below).

        Type-Specific Parameters (**kwargs):
            - For 'continuous':
                - frequency (float, optional): The center frequency of the source (in Meep units: c/distance).
                - wavelength (float, optional): The center wavelength in microns (alternative to frequency).
                - start_time (float, optional): Starting time. Default: 0.
                - end_time (float, optional): End time. Default: 1e+20.
                - width (float, optional): Temporal width of smoothing. Default: 0.
                - fwidth (float, optional): Frequency width (synonym for 1/width). Default: float('inf').
                - slowness (float, optional): Controls gradual turn-on. Default: 3.0.

            - For 'gaussian':
                - frequency (float, optional): The center frequency of the source (in Meep units: c/distance).
                - wavelength (float, optional): The center wavelength in microns (alternative to frequency).
                - start_time (float, optional): Starting time. Default: 0.
                - cutoff (float, optional): Number of widths before cutoff. Default: 5.0.
                - width (float, optional): Temporal width. Default: 0 (but typically set >0).
                - fwidth (float, optional): Frequency width (synonym for 1/width). Default: float('inf').

            - For 'kick' (thin Gaussian delta-kick approximation for broadband plasmon excitation):
                - peak_time (float, optional): Center time of the thin pulse. Default: 0.0.
                - width (float, optional): Temporal width (extremely small for "thin"/delta-like). Default: 1e-5.
                - cutoff (float, optional): Number of widths before cutoff. Default: 5.0.
                - fwidth (float, optional): Frequency width (synonym for 1/width). Default: float('inf').
                - frequency (float, optional): Carrier frequency (use 0 for non-oscillatory; non-zero may be needed to avoid NaN in some setups). Default: 0.0.

            - For <custom>:
                - start_time (float, optional): Starting time. Default: -1e+20.
                - end_time (float, optional): End time. Default: 1e+20.
                - fwidth (float, optional): Bandwidth in frequency units. Default: 0.
                - **kwargs.

            For more details on CustomSource, see Meep documentation: https://meep.readthedocs.io/en/latest/Python_User_Interface/#customsource
        """
        self.source_type = source_type
        logging.debug(f"Initializing MEEPSOURCE with type: {self.source_type}")

        char_to_field = {'x': mp.Ex, 'y': mp.Ey, 'z': mp.Ez}
        self.component = char_to_field[component]

        self.sourceCenter = mp.Vector3(*source_center) if not isinstance(source_center, mp.Vector3) else source_center
        self.sourceSize = mp.Vector3(*source_size) if not isinstance(source_size, mp.Vector3) else source_size

        # Compute frequency if wavelength is provided
        wavelength = kwargs.get('wavelength', None)
        frequency = kwargs.get('frequency', None)
        if wavelength is not None and frequency is None:
            frequency = 1 / wavelength

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
                frequency=frequency,
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
                frequency=frequency,
                width=width,
                start_time=start_time,
                cutoff=cutoff,
                is_integrated=is_integrated
            )
        elif self.source_type == 'kick':
            start_time = kwargs.get('start_time', 1)
            start_time_meep = start_time / constants.convertTimeMeep2fs
            
            end_time = kwargs.get('end_time', 1)
            end_time_meep = end_time / constants.convertTimeMeep2fs

            width  = kwargs.get('width', 0.1)  
            width_meep = width / constants.convertTimeMeep2fs

            def delta_kick_func(t):
                return amplitude * np.exp(-((t - start_time_meep)**2) / (2 * width_meep**2))

            src_time = mp.CustomSource(
                src_func=delta_kick_func,
                start_time=start_time_meep,
                end_time=end_time_meep,
                is_integrated=is_integrated
            )
        else:
            src_func = walk_through_src_funcs(self.source_type)
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

        # Create the final Meep Source object
        self.source = mp.Source(
            src=src_time,
            center=self.sourceCenter,
            size=self.sourceSize,
            component=self.component,
            amplitude=amplitude
        )

# ------------------------------------------ #
#             Additional custom              #
#            classes for sources             #
#             can be added here              #
# ------------------------------------------ #

def walk_through_src_funcs(src_func):
    if src_func == "PAPER_PULSE_CHEN2010":
        src_func = paper_pulse_chen2010
    else:
        raise ValueError(f"Invalid source function '{src_func}'; must be added to the list of supported functions within `sources.py`.")
    return src_func

CONVERSION_FACTOR = 3.378555833184493

def paper_pulse_chen2010(t):
    """
    Exact pulse from Chen et al. 2010 (eq. 36)
    t is in Meep time units
    """
    # t0_fs = 10.0
    # sigma_fs = 0.7
    # lambda_nm = 600.0

    # # Convert to Meep units using your factor
    # t0 = t0_fs / CONVERSION_FACTOR
    # sigma = sigma_fs / CONVERSION_FACTOR

    # # Angular frequency in Meep units
    # omega_meep = (2 * np.pi * 2.99792458e8 * 1e6 * CONVERSION_FACTOR * 1e-15) / (lambda_nm * 1e-9)

    # return np.exp(-((t - t0) / sigma)**2) * np.sin(omega_meep * t)

    t0_fs = 10.0
    sigma_fs = 0.7
    lambda_um = 0.600   # 600 nm

    from plasmol.utils import constants
    t0_meep = t0_fs / constants.convertTimeMeep2fs
    sigma_meep = sigma_fs / constants.convertTimeMeep2fs
    omega_meep = 2 * np.pi / lambda_um 

    return np.exp(-((t - t0_meep) / sigma_meep)**2) * np.sin(omega_meep * t)

# WIPs
# def pulse(t):
#     frequency_meep = (1 / wavelength if wavelength else frequency) * 2.99792458e8 * 1e6 * CONVERSION_FACTOR * 1e-15
#     peak_time_meep = peakTime * (1 / CONVERSION_FACTOR)  # peakTime in fs
#     width_meep = width * (CONVERSION_FACTOR ** 2)  # width in 1/fs**2
#     return np.exp(1j * 2 * np.pi * frequency_meep * (t - peak_time_meep)) * np.exp(-width_meep * (t - peak_time_meep) ** 2)

# def chirped(t):
#     # Similar to above, but add chirp term:
#     chirp_rate_meep = chirpRate * (CONVERSION_FACTOR ** 2)  # chirpRate in 1/fs**2
#     return np.exp(1j * 2 * np.pi * frequency_meep * (t - peak_time_meep)) * np.exp(-width_meep * (t - peak_time_meep) ** 2 + 1j * chirp_rate_meep * (t - peak_time_meep) ** 2)
