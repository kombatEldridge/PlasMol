# electric_field_generator.py
import numpy as np

C_NM_FS = 299.792458  # speed of light in nm/fs

class ElectricFieldGenerator:
    """
    Generates electric field components over time using a Gaussian-enveloped oscillatory function.
    """
    def __init__(self, wavelength, peak_time, width, smoothing=True, shape='pulse', dt_fs=None, kappa=None):
        """
        Initialize the generator with waveform parameters.

        Parameters:
            wavelength (float): Wavelength in nanometers
            peak_time (float): Time of pulse peak in femtoseconds
            width (float): Width parameter of Gaussian envelope in fs^{-2}
        """
        self.wavelength = wavelength
        self.peak_time = peak_time
        self.width = width
        self.frequency = C_NM_FS / wavelength
        self.smoothing = smoothing
        self.shape = shape
        self.dt_fs = dt_fs
        self.kappa = kappa


    def get_field_at(self, query_times, dir):
        """
        Returns the electric field components at the specified times.

        Parameters:
            query_times (array-like): Time values in femtoseconds to evaluate the field at
            dir (str): The direction ('x', 'y', or 'z') for the active component

        Returns:
            np.ndarray: A 2D array with columns [x, y, z] representing the field
        """
        # Convert query_times to a NumPy array
        t = np.asarray(query_times)
        
        # Compute the complex electric field
        if self.shape == 'pulse':
            complex_field = np.exp(1j * 2 * np.pi * self.frequency * (t - self.peak_time)) * \
                            np.exp(-self.width * (t - self.peak_time) ** 2)
        elif self.shape == 'kick':
            complex_field = self.kappa * np.exp(-((t - self.peak_time)**2) / (2 * self.width**2))
        
        # Extract the active component and apply threshold
        active_component = np.real(complex_field)
        active_component[abs(active_component) < 1e-20] = 0
        
        # Define a smoothing window at the start
        if self.smoothing:
            ramp_duration = 10  # Adjust this (in fs) to control the smoothing length
            t_start = t[0]  # Assume t is sorted and starts at the beginning
            window = np.ones_like(t)
            mask = t < (t_start + ramp_duration)
            window[mask] = 0.5 * (1 - np.cos(np.pi * (t[mask] - t_start) / ramp_duration))
        
            # Apply the window to the active component
            active_component *= window

        # Initialize a 2D array for the field components (all zeros initially)
        field = np.zeros((len(t), 3))
        
        # Map direction to column index and assign active component
        dir = dir.lower()  # Make direction case-insensitive
        if dir == 'x':
            field[:, 0] = active_component  # x-component
        elif dir == 'y':
            field[:, 1] = active_component  # y-component
        elif dir == 'z':
            field[:, 2] = active_component  # z-component
        else:
            raise ValueError("Invalid direction. Must be 'x', 'y', or 'z'.")
        
        # Return the field as a 2D array with columns [x, y, z]
        return field