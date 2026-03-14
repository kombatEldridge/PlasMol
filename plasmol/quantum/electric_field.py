# quantum/electric_field.py
import logging 
import numpy as np

from plasmol import constants

logger = logging.getLogger("main")

class ELECTRICFIELD:
    def __init__(self, 
                 times, 
                 source_type, 
                 intensity, 
                 peak_time, 
                 width_steps, 
                 dt, 
                 component, 
                 **kwargs):

        self.source_type = source_type
        logging.debug(f"Initializing MEEPSOURCE with type: {self.source_type}")

        self.times = times
        self.intensity_au = intensity
        self.peak_time_au = peak_time
        self.width_steps = width_steps
        self.dt = dt
        self.width_au = self.width_steps * self.dt
        self.component = component

        if self.source_type == 'pulse':
            self.frequency_um = kwargs.get('frequency', None)
            if self.frequency_um is None:
                self.wavelength_um = kwargs.get('wavelength')
            else:
                self.wavelength_um = 1/self.frequency_um
            self.wavelength_au = self.wavelength_um * constants.D_AU_UM
        elif self.source_type == 'kick':
            pass
        # elif self.source_type == 'custom_shape':
            # ------------------------------------ #
            #              Additional              #
            #    custom sources can be defined     #
            #      below and supported here        #
            #          with add'l params           #
            # ------------------------------------ #
        else:
            raise ValueError("Invalid shape. Must be 'pulse' or 'kick'.")
        
        t = np.asarray(self.times)
        if self.source_type == 'pulse':
            omega = 2 * np.pi * constants.C_AU / self.wavelength_au
            carrier  = np.exp(1j * omega * (t - self.peak_time_au))
            envelope = np.exp(-((t - self.peak_time_au)**2) / (2 * self.width_au**2))
            active_component = self.intensity_au * np.real(carrier * envelope)
        elif self.source_type == 'kick':
            envelope = np.exp(-((t - self.peak_time_au)**2) / (2 * self.width_au**2))
            active_component = np.zeros_like(t)
            mask = np.isclose(t, self.peak_time_au, atol=1e-10)
            if np.any(mask):
                active_component[mask] = self.intensity_au
            else:
                raise ValueError("No exact match found for peak time in time array.")
        # elif self.source_type == 'custom_shape':
            # ------------------------------------ #
            #              Additional              #
            #    custom sources can be defined     #
            #      here and supported above        #
            #             as commented             #
            # ------------------------------------ #

        self.field = np.zeros((len(t), 3))
        if self.component == 'x':
            self.field[:, 0] = active_component
        elif self.component == 'y':
            self.field[:, 1] = active_component
        elif self.component == 'z':
            self.field[:, 2] = active_component
        else:
            raise ValueError("Invalid component. Must be 'x', 'y', or 'z'.")
