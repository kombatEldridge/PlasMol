# utils/input/params.py
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
        self.do_nothing = self.preparams["args"]["do_nothing"]
        self.buildSettingsParams()

        if self.type == 'PlasMol':
            self.buildQuantumParams()
            self.buildclassicalParams()
        elif self.type == 'Quantum':
            self.buildQuantumParams()
        elif self.type == 'Classical':
            self.buildclassicalParams()

        delattr(self, 'preparams')


    def buildQuantumParams(self):
        self.pField_path = self.preparams["quantum"]["files"]["pField_path"]
        self.pField_Transform_path = self.preparams["quantum"]["files"]["pField_Transform_path"]
        self.eField_vs_pField_path = self.preparams["quantum"]["files"]["eField_vs_pField_path"]
        self.eV_spectrum_path = self.preparams["quantum"]["files"]["eV_spectrum_path"]
        self.checkpoint_path = self.preparams["quantum"]["files"]["checkpoint"]["path"] if "checkpoint" in self.preparams["quantum"]["files"] else None
        self.checkpoint_freq = self.preparams["quantum"]["files"]["checkpoint"]["frequency"] if "checkpoint" in self.preparams["quantum"]["files"] else None
        self.molecule_coords = self.preparams["quantum"]["rttddft"]["geometry"]["molecule_coords"]
        self.molecule_atoms = self.preparams["quantum"]["rttddft"]["geometry"]["atoms"]
        self.atoms = self.preparams["quantum"]["rttddft"]["geometry"]["atoms"]
        self.basis = self.preparams["quantum"]["rttddft"]['basis']
        self.charge = self.preparams["quantum"]["rttddft"]['charge']
        self.spin = self.preparams["quantum"]["rttddft"]['spin']
        self.xc = self.preparams["quantum"]["rttddft"]['xc']
        self.mu = self.preparams["quantum"]["rttddft"].get('mu', None)
        self.propagator = self.preparams["quantum"]["rttddft"]["propagator"].lower()
        self.check_tolerance = self.preparams["quantum"]["rttddft"]['check_tolerance']
        self.transform = True if "transform" in self.preparams["quantum"]["rttddft"] else False
        self.fourier_gamma = self.preparams["quantum"]["rttddft"].get("fourier_gamma")
        self.damping = 'static' if "damping" in self.preparams["quantum"]["rttddft"] else None
        self.mu_damping = self.preparams["quantum"]["rttddft"].get("mu_damping", 0)
        if self.damping is not None:
            if 'dynamic' in self.preparams["quantum"]["rttddft"]["damping"]:
                self.damping = 'dynamic'
            else:
                logger.warning("No damping type specified, defaulting to static.")
            self.gam0 = self.preparams["quantum"]["rttddft"]["damping"].get("gam0", None)
            self.xi = self.preparams["quantum"]["rttddft"]["damping"].get("xi", None)
            self.eps0 = self.preparams["quantum"]["rttddft"]["damping"].get("eps0", None)
            self.clamp = self.preparams["quantum"]["rttddft"]["damping"].get("clamp", None)

        if "comparison" in self.preparams["quantum"]:
            self.bases = self.preparams["quantum"]["comparison"].get("bases", [])
            self.xcs = self.preparams["quantum"]["comparison"].get("xcs", [])
            self.num_virtual = self.preparams["quantum"]["comparison"].get("num_virtual")
            self.num_occupied = self.preparams["quantum"]["comparison"].get("num_virtual")
            self.y_min = self.preparams["quantum"]["comparison"].get("y_min")
            self.y_max = self.preparams["quantum"]["comparison"].get("y_max")
            if not self.bases or not self.xcs:
                raise ValueError("Comparison mode requires 'bases' and 'xcs' lists in the 'comparison' block.")
            # Optionally override other params if needed, e.g., disable source if present
            if hasattr(self, 'shape'):
                logger.warning("Comparison mode ignores source; no time propagation.")
                delattr(self, 'shape')  # Prevent field setup

        if 'source' in self.preparams['quantum']:
            if not self.type == 'Quantum':
                logger.warning("Source block found in quantum section, but full PlasMol simulation is available. Ignoring source in quantum section. For full PlasMol simulation, please add source to classical section.")
            else:
                self.shape = self.preparams["quantum"]["source"]['shape']
                self.peak_time_au = self.preparams["quantum"]["source"]['peak_time_au']
                self.width_steps = self.preparams["quantum"]["source"]['width_steps']
                self.intensity_au = self.preparams["quantum"]["source"]['intensity_au']
                self.wavelength_nm = self.preparams["quantum"]["source"]['wavelength_nm'] if self.shape == 'pulse' else None
                self.dir = self.preparams["quantum"]["source"]['dir'] if not self.transform else None
                # if you want to add a custom shape you must add support for the relevant parameters here

        if self.propagator == 'step':
            pass
        elif self.propagator == 'magnus2':
            self.maxiter = self.preparams["quantum"]["rttddft"]["maxiter"]
            self.pc_convergence = self.preparams["quantum"]["rttddft"]["pc_convergence"]
        elif self.propagator == 'rk4':
            pass
        else:
            raise ValueError(f"Unsupported propagator: {self.propagator}. Please provide in the molecule input file one of the acceptable Density matrix propagators: step, rk4, or magnus2.")
        


    def buildclassicalParams(self):
        import meep as mp

        def getMoleculeLocation(self):
            if self.preparams['classical'].get("molecule", None) is None:
                logger.info('No molecule chosen for simulation. Continuing without it.')
                return None
            
            return self.preparams['classical']['molecule']

        def getSimulationParams(self):
            if self.preparams['classical'].get("simulation", None) is None:
                raise RuntimeError('No simulation parameters chosen for simulation. Exiting.')
            
            return self.preparams['classical']["simulation"]

        def getSource(self):
            from plasmol.classical import sources_old

            if self.preparams['classical'].get("source", None) is None:
                logger.info('No source chosen for simulation. Continuing without it.')
                return None

            source_type = self.preparams['classical']['source']['sourceType']
            if source_type == 'continuous':
                source_params = {
                    key: value for key, value in self.preparams['classical']['source'].items()
                    if key in ['frequency', 'wavelength', 'start_time', 'end_time', 
                            'width', 'fwidth', 'slowness', 'is_integrated', 'component']
                }
                source = sources_old.ContinuousSource(
                    sourceCenter=self.preparams['classical']['source']['sourceCenter'],
                    sourceSize=self.preparams['classical']['source']['sourceSize'],
                    **source_params
                )
            elif source_type == 'gaussian':
                source_params = {
                    key: value for key, value in self.preparams['classical']['source'].items()
                    if key in ['frequency', 'wavelength', 'width', 'fwidth', 'start_time', 
                            'cutoff', 'is_integrated', 'component']
                }
                source = sources_old.GaussianSource(
                    sourceCenter=self.preparams['classical']['source']['sourceCenter'],
                    sourceSize=self.preparams['classical']['source']['sourceSize'],
                    **source_params
                )
            elif source_type == 'chirped':
                source_params = {
                    key: value for key, value in self.preparams['classical']['source'].items()
                    if key in ['frequency', 'wavelength', 'width', 'peakTime', 'chirpRate', 
                            'start_time', 'end_time', 'is_integrated', 'component']
                }
                source = sources_old.ChirpedSource(
                    sourceCenter=self.preparams['classical']['source']['sourceCenter'],
                    sourceSize=self.preparams['classical']['source']['sourceSize'],
                    **source_params
                )
            elif source_type == 'pulse':
                source_params = {
                    key: value for key, value in self.preparams['classical']['source'].items()
                    if key in ['frequency', 'wavelength', 'width', 'peakTime', 'start_time', 'end_time', 'is_integrated', 'component']
                }

                source = sources_old.PulseSource(
                    sourceCenter=self.preparams['classical']['source']['sourceCenter'],
                    sourceSize=self.preparams['classical']['source']['sourceSize'],
                    **source_params
                )
            # elif ....
                # ------------------------------------ #
                #          Additional custom           #
                #         classes for sources          #
                #           can be added to            #
                #        `classical/sources.py`        #
                #    and then added here for support   #
                # ------------------------------------ #
            else:
                raise ValueError(f"Unsupported source type: {source_type}")

            return source

        def getNanoparticle(self):
            if self.preparams['classical'].get("object", None) is None:
                logger.info('No object chosen for simulation. Continuing without it.')
                return None

            if self.preparams['classical']['object']['material'] == 'Au':
                from meep.materials import Au_JC_visible as Au
                material = Au
            elif self.preparams['classical']['object']['material'] == 'Ag':
                from meep.materials import Ag
                material = Ag
            else:
                raise ValueError(f"Unsupported material type: {self.preparams['classical']['object']['material']}")

            objectNP = mp.Sphere(radius=self.preparams['classical']['object']['radius'], center=self.preparams['classical']['object']['center'], material=material)
            return objectNP

        def getSymmetry(self):
            if self.preparams['classical']['simulation'].get("symmetries", None) is None:
                logger.info('No symmetries chosen for simulation. Continuing without them.')
                return None
            
            sym = self.preparams['classical']['simulation']['symmetries']
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
            if self.preparams['classical'].get("hdf5", None) is None:
                logger.info('No picture output chosen for simulation. Continuing without it.')
                return None

            if any(key not in self.preparams['classical']['hdf5'] for key in ['timestepsBetween', 'intensityMin', 'intensityMax']):
                raise ValueError("If you want to generate pictures, you must provide timestepsBetween, intensityMin, and intensityMax.")

            if 'imageDirName' not in self.preparams['classical']['hdf5']:
                import os 
                from datetime import datetime
                self.preparams['classical']['hdf5']['imageDirName'] = f"classical-{datetime.now().strftime('%m%d%Y_%H%M%S')}"
                logger.info(f"Directory for images: {os.path.abspath(self.preparams['classical']['hdf5']['imageDirName'])}")

            return self.preparams['classical']['hdf5']
        
        self.simulation_params = getSimulationParams(self)
        self.molecule_position = getMoleculeLocation(self)
        self.source = getSource(self)
        self.symmetry = getSymmetry(self)
        self.nanoparticle = getNanoparticle(self)
        self.hdf5 = gethdf5(self)

        from plasmol import constants
        if 'resolution' in self.simulation_params:
            dtAlt = (0.5 / self.simulation_params["resolution"]) * constants.convertTimeMeep2Atomic
            if not np.isclose(dtAlt, self.dt):
                logger.info(f"Resolution given in simulation parameters does not generate given time step 'dt'. Ignoring given resolution, using new resolution: {newResolution}")
                newResolution = round(0.5 / (self.dt / constants.convertTimeMeep2Atomic))
                self.simulation_params["resolution"] = newResolution
        else:
            newResolution = round(0.5 / (self.dt / constants.convertTimeMeep2Atomic))
            self.simulation_params["resolution"] = newResolution


    def buildSettingsParams(self):
        self.dt = self.preparams["settings"]["dt"]
        self.t_end = self.preparams["settings"]["t_end"]
        self.eField_path = self.preparams["settings"]["eField_path"]

