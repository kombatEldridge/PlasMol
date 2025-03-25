# interpolation.py
import numpy as np
from scipy.interpolate import interp1d

class ElectricFieldInterpolator:
    """
    Interpolates electric field components over time using cubic interpolation.
    """
    def __init__(self, time_values, electric_x, electric_y, electric_z):
        """
        Initialize the interpolator with time and electric field data.

        Parameters:
            time_values (list or array-like): Time values.
            electric_x (list or array-like): Electric field component in x.
            electric_y (list or array-like): Electric field component in y.
            electric_z (list or array-like): Electric field component in z.
        """
        self.interp_x = interp1d(time_values, electric_x, kind='cubic', fill_value="extrapolate")
        self.interp_y = interp1d(time_values, electric_y, kind='cubic', fill_value="extrapolate")
        self.interp_z = interp1d(time_values, electric_z, kind='cubic', fill_value="extrapolate")

    def get_field_at(self, query_times):
        """
        Returns the interpolated electric field components at the specified times.

        Parameters:
            query_times (array-like): Time values to interpolate.

        Returns:
            np.ndarray: A 2D array with columns [x, y, z] representing the interpolated field.
        """
        x_interp = self.interp_x(query_times)
        y_interp = self.interp_y(query_times)
        z_interp = self.interp_z(query_times)

        # Set values below 1e-20 to 0
        x_interp[abs(x_interp) < 1e-20] = 0
        y_interp[abs(y_interp) < 1e-20] = 0
        z_interp[abs(z_interp) < 1e-20] = 0

        return np.column_stack((x_interp, y_interp, z_interp))