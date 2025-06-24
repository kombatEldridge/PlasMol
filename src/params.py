# params.py
import logging
import sources
import meep as mp
import numpy as np
from meep.materials import Au_JC_visible as Au
from meep.materials import Ag
import constants

logger = logging.getLogger("main")

class PARAMS:
    """
    Container for simulation parameters given from input files and cli inputs.
    """
    def __init__(self, preparams):
        self.preparams = preparams
        self.type = self.preparams["simulation_type"]
        self.dt = self.preparams["dt"]
        self.t_end = self.preparams["t_end"]
        self.restart = self.preparams["restart"]
        self.eField_path = self.preparams["eField_path"]
        self.pField_path = self.preparams["pField_path"]
        self.pField_Transform_path = self.preparams["pField_Transform_path"]
        self.eField_vs_pField_path = self.preparams["eField_vs_pField_path"]
        self.eV_spectrum_path = self.preparams["eV_spectrum_path"]
        self.chkfile_path = self.preparams["chkfile"]["path"]
        self.chkfile_freq = self.preparams["chkfile"]["frequency"]
        self.molecule_coords = self.preparams["rttddft"]["geometry"]["molecule_coords"]
        self.basis = self.preparams["rttddft"]['basis']
        self.charge = self.preparams["rttddft"]['charge']
        self.spin = self.preparams["rttddft"]['spin']
        self.xc = self.preparams["rttddft"]['xc']
        self.check_tolerance = self.preparams["rttddft"]['check_tolerance']
        self.propagator = self.preparams["rttddft"]["propagator"]
        self.maxiter = self.preparams["rttddft"]["maxiter"]
        self.pc_convergence = self.preparams["rttddft"]["pc_convergence"]

        if self.type == 'PlasMol':
            self.buildRttddftParams()
            self.buildMeepParams()
        elif self.type == 'RT-TDDFT':
            self.buildRttddftParams()
        elif self.type == 'Meep':
            self.buildMeepParams()

    def buildRttddftParams(self):
        return True

    def buildMeepParams(self):
        self.simParams = self.getSimulation()
        self.meepmolecule = self.getMolecule()
        self.sourceType = self.getSource()
        self.symmetries = self.getSymmetry()
        self.objectNP = self.getObject()
        self.hdf5 = self.gethdf5()
        
        if 'resolution' in self.simParams:
            dtAlt = (0.5 / self.simParams["resolution"]) * constants.convertTimeMeep2Atomic
            if not np.isclose(dtAlt, self.dt):
                logger.info(f"Resolution given in simulation parameters does not generate given time step 'dt'. Ignoring given resolution, using new resolution: {newResolution}")
                newResolution = round(0.5 / (self.dt / constants.convertTimeMeep2Atomic))
                self.simParams["resolution"] = newResolution
        else:
            newResolution = round(0.5 / (self.dt / constants.convertTimeMeep2Atomic))
            self.simParams["resolution"] = newResolution

    def getMolecule(self):
        if self.preparams.get("molecule", None) is None:
            logger.info('No molecule chosen for simulation. Continuing without it.')
            return None
        
        return self.preparams['molecule']


    def getSimulation(self):
        if self.preparams.get("simulation", None) is None:
            raise RuntimeError('No simulation parameters chosen for simulation. Exiting.')
        
        return self.preparams["simulation"]


    def getSource(self):
        if self.preparams.get("source", None) is None:
            logger.info('No source chosen for simulation. Continuing without it.')
            return None

        source_type = self.preparams['source']['sourceType']
        if source_type == 'continuous':
            source_params = {
                key: value for key, value in self.preparams['source'].items()
                if key in ['frequency', 'wavelength', 'start_time', 'end_time', 
                           'width', 'fwidth', 'slowness', 'is_integrated', 'component']
            }
            source = sources.ContinuousSource(
                sourceCenter=self.preparams['source']['sourceCenter'],
                sourceSize=self.preparams['source']['sourceSize'],
                **source_params
            )
        elif source_type == 'gaussian':
            source_params = {
                key: value for key, value in self.preparams['source'].items()
                if key in ['frequency', 'wavelength', 'width', 'fwidth', 'start_time', 
                           'cutoff', 'is_integrated', 'component']
            }
            source = sources.GaussianSource(
                sourceCenter=self.preparams['source']['sourceCenter'],
                sourceSize=self.preparams['source']['sourceSize'],
                **source_params
            )
        elif source_type == 'chirped':
            source_params = {
                key: value for key, value in self.preparams['source'].items()
                if key in ['frequency', 'wavelength', 'width', 'peakTime', 'chirpRate', 
                           'start_time', 'end_time', 'is_integrated', 'component']
            }
            source = sources.ChirpedSource(
                sourceCenter=self.preparams['source']['sourceCenter'],
                sourceSize=self.preparams['source']['sourceSize'],
                **source_params
            )
        elif source_type == 'pulse':
            source_params = {
                key: value for key, value in self.preparams['source'].items()
                if key in ['frequency', 'wavelength', 'width', 'peakTime', 'start_time', 'end_time', 'is_integrated', 'component']
            }

            source = sources.PulseSource(
                sourceCenter=self.preparams['source']['sourceCenter'],
                sourceSize=self.preparams['source']['sourceSize'],
                **source_params
            )
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

        return source


    def getObject(self):
        if self.preparams.get("object", None) is None:
            logger.info('No object chosen for simulation. Continuing without it.')
            return None

        if self.preparams['object']['material'] == 'Au':
            material = Au
        elif self.preparams['object']['material'] == 'Ag':
            material = Ag
        else:
            raise ValueError(f"Unsupported material type: {self.preparams['object']['material']}")

        objectNP = mp.Sphere(radius=self.preparams['object']['radius'], center=self.preparams['object']['center'], material=material)
        return objectNP


    def getSymmetry(self):
        if self.preparams['simulation'].get("symmetries", None) is None:
            logger.info('No symmetries chosen for simulation. Continuing without them.')
            return None
        
        sym = self.preparams['simulation']['symmetries']
        symmetries = []
        for i in range(len(sym)):
            if sym[i] in ['X', 'Y', 'Z']:
                if i + 1 < len(sym):
                    try:
                        phase = int(sym[i + 1])
                    except ValueError:
                        raise ValueError(
                            f"Symmetry '{sym[i]}' is not followed by a valid integer.")

                    if sym[i] == 'X':
                        symmetries.append(mp.Mirror(mp.X, phase=phase))
                    elif sym[i] == 'Y':
                        symmetries.append(mp.Mirror(mp.Y, phase=phase))
                    elif sym[i] == 'Z':
                        symmetries.append(mp.Mirror(mp.Z, phase=phase))
                else:
                    raise ValueError(f"Symmetry '{sym[i]}' has no value following it.")
        if not symmetries:
            raise ValueError(f"Unsupported symmetry type: {sym}")
        else:
            return symmetries


    def gethdf5(self):
        if self.preparams.get("hdf5", None) is None:
            logger.info('No picture output chosen for simulation. Continuing without it.')
            return None

        if any(key not in self.preparams['hdf5'] for key in ['timestepsBetween', 'intensityMin', 'intensityMax']):
            raise ValueError("If you want to generate pictures, you must provide timestepsBetween, intensityMin, and intensityMax.")

        if 'imageDirName' not in self.preparams['hdf5']:
            import os 
            from datetime import datetime
            self.preparams['hdf5']['imageDirName'] = f"meep-{datetime.now().strftime('%m%d%Y_%H%M%S')}"
            logger.info(f"Directory for images: {os.path.abspath(self.preparams['hdf5']['imageDirName'])}")

        return self.preparams['hdf5']

