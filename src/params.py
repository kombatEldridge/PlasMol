# params.py
import logging
import numpy as np

logger = logging.getLogger("main")

class PARAMS:
    """
    Container for simulation parameters given from input files and cli inputs.
    """
    def __init__(self, preparams):
        self.preparams = preparams
        self.type = self.preparams["simulation_type"]
        self.restart = self.preparams["args"]["restart"]
        self.buildSettingsParams()

        if self.type == 'PlasMol':
            self.buildQuantumParams()
            self.buildMeepParams()
        elif self.type == 'Quantum':
            self.buildQuantumParams()
        elif self.type == 'Meep':
            self.buildMeepParams()

        delattr(self, 'preparams')


    def buildQuantumParams(self):
        self.pField_path = self.preparams["quantum"]["files"]["pField_path"]
        self.pField_Transform_path = self.preparams["quantum"]["files"]["pField_Transform_path"]
        self.eField_vs_pField_path = self.preparams["quantum"]["files"]["eField_vs_pField_path"]
        self.eV_spectrum_path = self.preparams["quantum"]["files"]["eV_spectrum_path"]
        self.chkfile_path = self.preparams["quantum"]["files"]["chkfile"]["path"]
        self.chkfile_freq = self.preparams["quantum"]["files"]["chkfile"]["frequency"]
        self.molecule_coords = self.preparams["quantum"]["rttddft"]["geometry"]["molecule_coords"]
        self.basis = self.preparams["quantum"]["rttddft"]['basis']
        self.charge = self.preparams["quantum"]["rttddft"]['charge']
        self.spin = self.preparams["quantum"]["rttddft"]['spin']
        self.xc = self.preparams["quantum"]["rttddft"]['xc']
        self.propagator = self.preparams["quantum"]["rttddft"]["propagator"].lower()
        self.check_tolerance = self.preparams["quantum"]["rttddft"]['check_tolerance']
        self.transform = True if "transform" in self.preparams["quantum"]["rttddft"] else False

        if 'source' in self.preparams['quantum']:
            if not self.type == 'Quantum':
                logger.warning("Source block found in quantum section, but full PlasMol simulation is available. Ignoring source in quantum section. For full PlasMol simulation, please add source to meep section.")
            else:
                self.shape = self.preparams["quantum"]["source"]['shape']
                self.wavelength_nm = self.preparams["quantum"]["source"]['wavelength_nm']
                self.peak_time_au = self.preparams["quantum"]["source"]['peak_time_au']
                self.width_steps = self.preparams["quantum"]["source"]['width_steps']
                self.intensity_au = self.preparams["quantum"]["source"]['intensity_au']
                self.dir = self.preparams["quantum"]["source"]['dir']

        if self.propagator == 'step':
            pass
        elif self.propagator == 'magnus2':
            self.maxiter = self.preparams["quantum"]["rttddft"]["maxiter"]
            self.pc_convergence = self.preparams["quantum"]["rttddft"]["pc_convergence"]
        elif self.propagator == 'rk4':
            pass
        else:
            raise ValueError(f"Unsupported propagator: {self.propagator}. Please provide in the molecule input file one of the acceptable Density matrix propagators: step, rk4, or magnus2.")
        


    def buildMeepParams(self):
        import meep as mp

        def getMolecule(self):
            if self.preparams['meep'].get("molecule", None) is None:
                logger.info('No molecule chosen for simulation. Continuing without it.')
                return None
            
            return self.preparams['meep']['molecule']

        def getSimulation(self):
            if self.preparams['meep'].get("simulation", None) is None:
                raise RuntimeError('No simulation parameters chosen for simulation. Exiting.')
            
            return self.preparams['meep']["simulation"]

        def getSource(self):
            import sources

            if self.preparams['meep'].get("source", None) is None:
                logger.info('No source chosen for simulation. Continuing without it.')
                return None

            source_type = self.preparams['meep']['source']['sourceType']
            if source_type == 'continuous':
                source_params = {
                    key: value for key, value in self.preparams['meep']['source'].items()
                    if key in ['frequency', 'wavelength', 'start_time', 'end_time', 
                            'width', 'fwidth', 'slowness', 'is_integrated', 'component']
                }
                source = sources.ContinuousSource(
                    sourceCenter=self.preparams['meep']['source']['sourceCenter'],
                    sourceSize=self.preparams['meep']['source']['sourceSize'],
                    **source_params
                )
            elif source_type == 'gaussian':
                source_params = {
                    key: value for key, value in self.preparams['meep']['source'].items()
                    if key in ['frequency', 'wavelength', 'width', 'fwidth', 'start_time', 
                            'cutoff', 'is_integrated', 'component']
                }
                source = sources.GaussianSource(
                    sourceCenter=self.preparams['meep']['source']['sourceCenter'],
                    sourceSize=self.preparams['meep']['source']['sourceSize'],
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
                    key: value for key, value in self.preparams['meep']['source'].items()
                    if key in ['frequency', 'wavelength', 'width', 'peakTime', 'start_time', 'end_time', 'is_integrated', 'component']
                }

                source = sources.PulseSource(
                    sourceCenter=self.preparams['meep']['source']['sourceCenter'],
                    sourceSize=self.preparams['meep']['source']['sourceSize'],
                    **source_params
                )
            else:
                raise ValueError(f"Unsupported source type: {source_type}")

            return source

        def getObject(self):
            if self.preparams['meep'].get("object", None) is None:
                logger.info('No object chosen for simulation. Continuing without it.')
                return None

            if self.preparams['meep']['object']['material'] == 'Au':
                from meep.materials import Au_JC_visible as Au
                material = Au
            elif self.preparams['meep']['object']['material'] == 'Ag':
                from meep.materials import Ag
                material = Ag
            else:
                raise ValueError(f"Unsupported material type: {self.preparams['meep']['object']['material']}")

            objectNP = mp.Sphere(radius=self.preparams['meep']['object']['radius'], center=self.preparams['meep']['object']['center'], material=material)
            return objectNP

        def getSymmetry(self):
            if self.preparams['meep']['simulation'].get("symmetries", None) is None:
                logger.info('No symmetries chosen for simulation. Continuing without them.')
                return None
            
            sym = self.preparams['meep']['simulation']['symmetries']
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
            if self.preparams['meep'].get("hdf5", None) is None:
                logger.info('No picture output chosen for simulation. Continuing without it.')
                return None

            if any(key not in self.preparams['meep']['hdf5'] for key in ['timestepsBetween', 'intensityMin', 'intensityMax']):
                raise ValueError("If you want to generate pictures, you must provide timestepsBetween, intensityMin, and intensityMax.")

            if 'imageDirName' not in self.preparams['meep']['hdf5']:
                import os 
                from datetime import datetime
                self.preparams['meep']['hdf5']['imageDirName'] = f"meep-{datetime.now().strftime('%m%d%Y_%H%M%S')}"
                logger.info(f"Directory for images: {os.path.abspath(self.preparams['meep']['hdf5']['imageDirName'])}")

            return self.preparams['meep']['hdf5']
        
        self.simParams = getSimulation(self)
        self.meepmolecule = getMolecule(self)
        self.sourceType = getSource(self)
        self.symmetries = getSymmetry(self)
        self.objectNP = getObject(self)
        self.hdf5 = gethdf5(self)

        import constants
        if 'resolution' in self.simParams:
            dtAlt = (0.5 / self.simParams["resolution"]) * constants.convertTimeMeep2Atomic
            if not np.isclose(dtAlt, self.dt):
                logger.info(f"Resolution given in simulation parameters does not generate given time step 'dt'. Ignoring given resolution, using new resolution: {newResolution}")
                newResolution = round(0.5 / (self.dt / constants.convertTimeMeep2Atomic))
                self.simParams["resolution"] = newResolution
        else:
            newResolution = round(0.5 / (self.dt / constants.convertTimeMeep2Atomic))
            self.simParams["resolution"] = newResolution


    def buildSettingsParams(self):
        self.dt = self.preparams["settings"]["dt"]
        self.t_end = self.preparams["settings"]["t_end"]
        self.eField_path = self.preparams["settings"]["eField_path"]

