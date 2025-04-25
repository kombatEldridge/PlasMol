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

class ELECTRICFIELD:
    """
    Generates and manages electric field components over time using a Gaussian-enveloped oscillatory function or a kick shape.
    """
    def __init__(self, times, dir, wavelength_nm=None, frequency_cm1=None, energy_ev=None, energy_au=None,
                 peak_time_au=None, peak_time_fs=None, width_steps=None, width_duration_au=None, 
                 width_duration_fs=None, dt=None, smoothing=True, shape='pulse', intensity_au=0.05,
                 smoothing_ramp_duration_au=None, smoothing_ramp_duration_fs=10):
        """
        Initialize the electric field generator and storage.

        Parameters:
            times (np.arange): Times in au.
            dir (str): 'x', 'y', or 'z'
            wavelength_nm (float): Wavelength in nm.
            frequency_cm1 (float): Frequency in cm^{-1}.
            energy_ev (float): Energy in eV.
            energy_au (float): Energy in au.
            peak_time_au (float): Time of pulse peak in au.
            peak_time_fs (float): Time of pulse peak in fs.
            width_steps (int): Number of time steps to keep Electric Field active.
            width_duration_au (float): Time to keep Electric Field active in au.
            width_duration_fs (float): Time to keep Electric Field active in fs.
            dt (float): Time step in au.
            shape (str): 'pulse' or 'kick'.
            intensity (float): Intensity for electric field.
            smoothing (bool): Whether to apply smoothing at the start.
            smoothing_ramp_duration_au (float): Duration for smoothing from start to finish in au.
            smoothing_ramp_duration_fs (float): Duration for smoothing from start to finish in fs.
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
        Compute the electric field for a time range.

        Parameters:
            times (np.arange): Times in au.
            dir (str): 'x', 'y', or 'z'

        Returns:
            np.ndarray: Field components [x, y, z]
        """
        t = np.asarray(times) # must be in au
        if self.shape == 'pulse':
            active_component = np.real(self.intensity_au * np.exp(1j * 2 * np.pi * (1 / self.wavelength_au) * (t - self.peak_time_au)) * \
                            np.exp(-self.width_au * (t - self.peak_time_au) ** 2))
        elif self.shape == 'kick':
            active_component = self.intensity_au * np.exp(-((t - self.peak_time_au)**2) / (2 * self.width_au**2))

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

    def update_stored_fields(self, t, dt, dir):
        """
        Compute and store fields for t - 2*dt, t - dt, t, t + dt.

        Parameters:
            t (float): Current time in fs
            dt (float): Time step in fs
            dir (str): Direction 'x', 'y', or 'z'
        """
        times = [t - 2*dt, t - dt, t, t + dt]
        fields = self.get_field_at(times, dir)
        self.exc_store['exc_t_minus_2dt'] = fields[0]
        self.exc_store['exc_t_minus_dt'] = fields[1]
        self.exc_store['exc_t'] = fields[2]
        self.exc_store['exc_t_plus_dt'] = fields[3]

    def get_exc_t_plus_dt(self):
        """Get the stored field at t + dt."""
        return self.exc_store.get('exc_t_plus_dt', self.empty)

    def set_exc_t_plus_dt(self, exc_t_plus_dt):
        """Set the stored field at t + dt."""
        self.exc_store['exc_t_plus_dt'] = exc_t_plus_dt

    def get_exc_t(self):
        """Get the stored field at t."""
        return self.exc_store.get('exc_t', self.empty)

    def set_exc_t(self, exc_t):
        """Set the stored field at t."""
        self.exc_store['exc_t'] = exc_t

    def get_exc_t_minus_dt(self):
        """Get the stored field at t - dt."""
        return self.exc_store.get('exc_t_minus_dt', self.empty)

    def set_exc_t_minus_dt(self, exc_t_minus_dt):
        """Set the stored field at t - dt."""
        self.exc_store['exc_t_minus_dt'] = exc_t_minus_dt

    def get_exc_t_minus_2dt(self):
        """Get the stored field at t - 2*dt."""
        return self.exc_store.get('exc_t_minus_2dt', self.empty)

    def set_exc_t_minus_2dt(self, exc_t_minus_2dt):
        """Set the stored field at t - 2*dt."""
        self.exc_store['exc_t_minus_2dt'] = exc_t_minus_2dt