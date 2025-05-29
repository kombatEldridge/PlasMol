# electric_field.py
import numpy as np
import logging 

logger = logging.getLogger("main")

C_NM_FS = 299.792458    # Speed of light in nm/fs
A0 = 5.29177210903e-11  # Bohr radius in m
D_AU_NM = 1e-9 / A0     # Distance conv in au/nm
D_NM_CM = 1e-7          # Distance conv in nm/cm
T_AU_FS = 41.3413733    # Time conv in au/fs
E_EV_AU = 27.21138602   # Energy conv in eV/au
E_NM_EV = 1239.84193    # Energy conv in eV nm
C_AU = 137.035999       # Speed of light in atomic units

class ELECTRICFIELD:
    """
    A class to generate and manage electric field components over time.

    Supports Gaussian-enveloped oscillatory ('pulse') or Gaussian ('kick') field shapes.
    """
    def __init__(self, times, dir, wavelength_nm=None, frequency_cm1=None, energy_ev=None, energy_au=None,
                 peak_time_au=None, peak_time_fs=None, width_steps=None, width_duration_au=None, 
                 width_duration_fs=None, dt=None, smoothing=True, shape='pulse', intensity_au=0.05,
                 smoothing_ramp_duration_au=None, smoothing_ramp_duration_fs=10):
        """
        Initialize the electric field object.

        Configures the field based on shape, intensity, and timing parameters. For 'pulse',
        exactly one energy/wavelength parameter must be provided. All times are converted to atomic units.

        Parameters:
        times : np.ndarray
            Array of time points in atomic units where the field is evaluated.
        dir : str
            Direction of the field ('x', 'y', or 'z').
        wavelength_nm : float, optional
            Wavelength in nanometers (for 'pulse').
        frequency_cm1 : float, optional
            Frequency in cm^-1 (for 'pulse').
        energy_ev : float, optional
            Energy in electron volts (for 'pulse').
        energy_au : float, optional
            Energy in atomic units (for 'pulse').
        peak_time_au : float, optional
            Time of pulse peak in atomic units.
        peak_time_fs : float, optional
            Time of pulse peak in femtoseconds.
        width_steps : int, optional
            Number of time steps for field width.
        width_duration_au : float, optional
            Field width duration in atomic units.
        width_duration_fs : float, optional
            Field width duration in femtoseconds.
        dt : float, optional
            Time step in atomic units (required with width_steps).
        smoothing : bool, optional
            Whether to apply smoothing (default True, disabled for 'kick').
        shape : str, optional
            Field shape ('pulse' or 'kick', default 'pulse').
        intensity_au : float, optional
            Field intensity in atomic units (default 0.05).
        smoothing_ramp_duration_au : float, optional
            Smoothing ramp duration in atomic units.
        smoothing_ramp_duration_fs : float, optional
            Smoothing ramp duration in femtoseconds (default 10).

        Returns:
        None
        """
        self.exc_store = {}
        self.empty = np.array([0.0, 0.0, 0.0])

        self.shape = shape
        if intensity_au == 0.05:
            logger.debug("Warning: Intensity of Electric field being set to default value of 0.05 au.")
        self.intensity_au = intensity_au

        self.smoothing = smoothing
        if self.smoothing:
            if smoothing_ramp_duration_au is not None:
                self.ramp_duration_au = smoothing_ramp_duration_au
            else:
                self.ramp_duration_au = smoothing_ramp_duration_fs * T_AU_FS

        provided = sum([wavelength_nm is not None, frequency_cm1 is not None, energy_ev is not None, energy_au is not None])
        if shape == 'pulse':
            if provided != 1:
                raise ValueError("For shape='pulse', exactly one of wavelength_nm, frequency_cm1, energy_ev, or energy_au must be provided.")
            if wavelength_nm is not None:
                self.wavelength_nm = wavelength_nm
            elif frequency_cm1 is not None:
                self.wavelength_nm = D_NM_CM / frequency_cm1
            elif energy_ev is not None:
                self.wavelength_nm = E_NM_EV / energy_ev
            elif energy_au is not None:
                E_ev = energy_au * E_EV_AU 
                self.wavelength_nm = E_NM_EV / E_ev
            self.wavelength_au = self.wavelength_nm * D_AU_NM
        elif shape == 'kick':
            if provided > 0:
                logger.debug("Warning: For shape='kick', wavelength_nm, frequency_cm1, energy_ev, energy_au are not used.")
            if self.smoothing:
                logger.debug("Smoothing turned off for shape='kick'")
                self.smoothing = False
        else:
            raise ValueError("Invalid shape. Must be 'pulse' or 'kick'.")

        provided = sum([peak_time_fs is not None, peak_time_au is not None])
        if provided != 1:
            raise ValueError("Exactly one of peak_time_fs or peak_time_au must be provided.")
        if (peak_time_fs is not None):
            self.peak_time_au = peak_time_fs * T_AU_FS
        elif (peak_time_au is not None):
            self.peak_time_au = peak_time_au

        provided = sum([width_steps is not None, width_duration_fs is not None, width_duration_au is not None])
        if provided != 1:
            raise ValueError("Exactly one of width_steps, width_duration_fs, or width_duration_au must be provided.")
        if (width_steps is not None):
            if (dt is None):
                raise ValueError(f"You must provide dt when specifying width_steps.")
            self.width_au = width_steps * dt
        elif (width_duration_fs is not None):
            self.width_au = width_duration_fs * T_AU_FS
        elif (width_duration_au is not None):
            self.width_au = width_duration_au

        self.times = times
        self.dir = dir
        self.field = self.build_field(self.times, self.dir)

    def build_field(self, times, dir):
        """
        Compute the electric field for a given array of times.

        Generates the field based on the shape: 'pulse' (oscillatory) or 'kick' (Gaussian),
        applying it along the specified direction with optional smoothing.

        Parameters:
        times : np.ndarray
            Array of times in atomic units.
        dir : str
            Direction of the field ('x', 'y', or 'z').

        Returns:
        np.ndarray
            Array of shape (len(times), 3) with field components [x, y, z].
        """
        t = np.asarray(times) # must be in au
        if self.shape == 'pulse':
            omega = 2 * np.pi * C_AU / self.wavelength_au
            carrier  = np.exp(1j * omega * (t - self.peak_time_au))
            envelope = np.exp(-((t - self.peak_time_au)**2) / (2 * self.width_au**2))
            active_component = self.intensity_au * np.real(carrier * envelope)
        elif self.shape == 'kick':
            envelope = np.exp(-((t - self.peak_time_au)**2) / (2 * self.width_au**2))
            active_component = self.intensity_au * envelope

        if self.smoothing:
            t_start = t[0]
            window = np.ones_like(t)
            mask = t < (t_start + self.ramp_duration_au)
            window[mask] = 0.5 * (1 - np.cos(np.pi * (t[mask] - t_start) / self.ramp_duration_au))
            active_component *= window

        field = np.zeros((len(t), 3))
        dir = dir.lower()
        if dir == 'x':
            field[:, 0] = active_component
        elif dir == 'y':
            field[:, 1] = active_component
        elif dir == 'z':
            field[:, 2] = active_component
        else:
            raise ValueError("Invalid direction. Must be 'x', 'y', or 'z'.")
        return field

