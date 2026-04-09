from datetime import datetime

# Each entry is a tuple:
# (attribute_name, path_as_list, is_section_dict, boolean_name, default_value, section_condition, data_type, description, units)
param_defs = [
    # Settings (always required)
    ('dt', ['settings', 'dt'], False, None, None, None, (int, float), "Time step", "a.u."),
    ('t_end', ['settings', 't_end'], False, None, None, None, (int, float), "End time", "a.u."),

    # Plasmon params
    ('plasmon_dict', ['plasmon'], True, "has_plasmon", None, 'plasmon', dict, None, None),

    # Plasmon simulation params
    ('plasmon_simulation_dict', ['plasmon', 'simulation'], True, 'has_simulation', None, 'plasmon', dict, None, None),
    ('plasmon_tolerance_field_e', ['plasmon', 'simulation', "tolerance_field_e"], False, 'has_simulation', 1e-12, 'plasmon', (int, float), "Minimum |E| before quantum propagation is triggered", "a.u."),
    ('plasmon_cell_length', ['plasmon', 'simulation', "cell_length"], False, 'has_simulation', 0.1, 'plasmon', (int, float), "Length of the simulation cell (used if cell_volume not provided)", "μm"),
    ('plasmon_cell_volume', ['plasmon', 'simulation', "cell_volume"], False, 'has_simulation', None, 'plasmon', (int, float), "Simulation cell volume (overrides cell_length if provided)", "μm"),
    ('plasmon_pml_thickness', ['plasmon', 'simulation', "pml_thickness"], False, 'has_simulation', 0.01, 'plasmon', (int, float), "Thickness of the PML absorbing boundary layers (recommended to be close to half the largest wavelength)", "μm"),
    ('plasmon_symmetries', ['plasmon', 'simulation', 'symmetries'], False, 'has_simulation', None, 'plasmon', list, "Symmetry operations (axis followed by phase) e.g. ['Y', 1, 'Z', -1]", None),
    ('plasmon_surrounding_material_index', ['plasmon', 'simulation', "surrounding_material_index"], False, 'has_simulation', 1.33, 'plasmon', (int, float), "Refractive index of the surrounding medium", None),

    # Plasmon source params
    ('plasmon_source_dict', ['plasmon', 'source'], True, "has_plasmon_source", None, 'plasmon', dict, None, None),
    ('plasmon_source_type', ['plasmon', 'source', 'type'], False, 'has_plasmon_source', None, 'plasmon', str, "Type of source ('continuous', 'gaussian', or 'custom')", None),
    ('plasmon_source_center', ['plasmon', 'source', 'center'], False, 'has_plasmon_source', None, 'plasmon', list, "Center coordinates of the source", "μm"),
    ('plasmon_source_size', ['plasmon', 'source', 'size'], False, 'has_plasmon_source', None, 'plasmon', list, "Size of the source volume (for 3D simulations, propagation dimension should be zero)", "μm"),
    ('plasmon_source_component', ['plasmon', 'source', 'component'], False, 'has_plasmon_source', None, 'plasmon', str, "Electric field component the source acts on", None),
    ('plasmon_source_amplitude', ['plasmon', 'source', 'amplitude'], False, 'has_plasmon_source', 1, 'plasmon', (int, float), "Overall amplitude multiplying the source", None),
    ('plasmon_source_is_integrated', ['plasmon', 'source', 'is_integrated'], False, 'has_plasmon_source', True, 'plasmon', bool, "Whether the source is integrated over time (dipole moment)", None),
    ('plasmon_source_additional_parameters', ['plasmon', 'source', 'additional_parameters'], False, 'has_plasmon_source', None, 'plasmon', dict, "Type-specific parameters (frequency, wavelength, width, etc.)", "μm"),

    # Nanoparticle params
    ('nanoparticle_dict', ['plasmon', 'nanoparticle'], True, "has_nanoparticle", None, 'plasmon', dict, None, None),
    ('nanoparticle_material', ['plasmon', 'nanoparticle', 'material'], False, 'has_nanoparticle', None, 'plasmon', str, "Material name from meep.materials (e.g. 'Au_JC_visible')", None),
    ('nanoparticle_radius', ['plasmon', 'nanoparticle', 'radius'], False, 'has_nanoparticle', None, 'plasmon', (int, float), "Radius of the spherical nanoparticle", "μm"),
    ('nanoparticle_center', ['plasmon', 'nanoparticle', 'center'], False, 'has_nanoparticle', [0,0,0], 'plasmon', list, "Center coordinates of the nanoparticle", "μm"),

    # Images params
    ('images_dict', ['plasmon', 'images'], True, "has_images", None, 'plasmon', dict, None, None),
    ('images_timesteps_between', ['plasmon', 'images', 'timesteps_between'], False, 'has_images', None, 'plasmon', int, "Number of Meep timesteps between PNG frame outputs", None),
    ('images_additional_parameters', ['plasmon', 'images', 'additional_parameters'], False, 'has_images', ["-Zc dkbluered", "-S 10"], 'plasmon', list, "Additional arguments passed to Meep's output_png (from h5topng)", None),
    ('images_dir_name', ['plasmon', 'images', 'dir_name'], False, 'has_images', f"plasmol-{datetime.now().strftime('%m%d%Y_%H%M%S')}", 'plasmon', str, "Directory name where PNG frames will be saved", None),
    ('images_make_gif', ['plasmon', 'images', 'make_gif'], False, 'has_images', True, 'plasmon', bool, "Automatically create animated GIF from the PNG frames after simulation", None),

    # Molecule position param
    ('plasmol_molecule_position', ['plasmon', 'molecule_position'], True, "has_molecule_position", None, 'plasmon', list, "Position of the quantum molecule inside the Meep cell", "μm"),

    # Molecule params
    ('molecule_dict', ['molecule'], True, "has_molecule", None, 'molecule', dict, None, None),
    ('molecule_geometry', ['molecule', 'geometry'], False, 'has_molecule', None, 'molecule', list, "Molecular geometry as list of atom+coord entries", None),
    ('molecule_geometry_units', ['molecule', 'geometry_units'], False, 'has_molecule', None, 'molecule', str, "Units of the geometry coordinates", None),
    ('molecule_basis', ['molecule', 'basis'], False, 'has_molecule', None, 'molecule', str, "Basis set name (e.g. '6-31g')", None),
    ('molecule_charge', ['molecule', 'charge'], False, 'has_molecule', None, 'molecule', int, "Total molecular charge", None),
    ('molecule_spin', ['molecule', 'spin'], False, 'has_molecule', None, 'molecule', int, "Spin multiplicity minus one (0 = singlet)", None),
    ('molecule_xc', ['molecule', 'xc'], False, 'has_molecule', None, 'molecule', str, "Exchange-correlation functional", None),
    ('molecule_lrc_parameter', ['molecule', 'lrc_parameter'], False, 'has_molecule', None, 'molecule', float, "Long-range correction parameter (mu/omega) for RSH functionals", "a.u."),
    ('molecule_propagator_str', ['molecule', 'propagator', "type"], False, 'has_molecule', 'magnus2', 'molecule', str, "Time-propagation algorithm", None),
    ('molecule_pc_convergence', ['molecule', 'propagator', "pc_convergence"], False, 'has_molecule', 1e-12, 'molecule', (int, float), "Predictor-corrector convergence threshold (Magnus2 only)", "a.u."),
    ('molecule_max_iterations', ['molecule', 'propagator', "max_iterations"], False, 'has_molecule', 200, 'molecule', int, "Maximum predictor-corrector iterations (Magnus2 only)", None),
    ('molecule_hermiticity_tolerance', ['molecule', 'hermiticity_tolerance'], False, 'has_molecule', 1e-12, 'molecule', (int, float), "Tolerance for checking Hermitian matrices", "a.u."),
    
    # Source params (molecule section)
    ('molecule_source_dict', ['molecule', 'source'], True, "has_molecule_source", None, 'molecule', dict, None, None),
    ('molecule_source_type', ['molecule', 'source', 'type'], False, 'has_molecule_source', None, 'molecule', str, "Shape of the external electric field source", None),
    ('molecule_source_intensity', ['molecule', 'source', 'intensity'], False, 'has_molecule_source', None, 'molecule', (int, float), "Electric field intensity", "a.u."),
    ('molecule_source_peak_time', ['molecule', 'source', 'peak_time'], False, 'has_molecule_source', None, 'molecule', (int, float), "Time at which the pulse/kick peaks", "a.u."),
    ('molecule_source_width_steps', ['molecule', 'source', 'width_steps'], False, 'has_molecule_source', None, 'molecule', int, "Width of the pulse in time steps", None),
    ('molecule_source_component', ['molecule', 'source', 'component'], False, 'has_molecule_source', None, 'molecule', str, "Direction of the electric field", None),
    ('molecule_source_additional_parameters', ['molecule', 'source', 'additional_parameters'], False, 'has_molecule_source', None, 'molecule', dict, "Additional parameters (wavelength or frequency for pulse)", "μm"),

    # Fourier params
    ('fourier_dict', ['molecule', 'modifiers', 'fourier'], True, "has_fourier", None, 'molecule', dict, None, None),
    ('fourier_gamma', ['molecule', 'modifiers', 'fourier', 'gamma'], False, 'has_fourier', 0.01, 'molecule', (int, float), "Broadening factor for Fourier transformed spectrum", "a.u."),
    ('fourier_npz_filepath', ['molecule', 'modifiers', 'fourier', 'npz_filepath'], False, 'has_fourier', None, 'molecule', str, "File path for npz file containing imaginary absorption and frequencies", None),
    ('fourier_spectrum_filepath', ['molecule', 'modifiers', 'fourier', 'spectrum_filepath'], False, 'has_fourier', None, 'molecule', str, "Output file path for the absorption spectrum plot", None),
    ('fourier_damping_gamma', ['molecule', 'modifiers', 'fourier', 'damping_gamma'], False, 'has_fourier', None, 'molecule', (int, float), "Artificial damping applied to polarization field for better FFT resolution", "a.u."),

    # Lopata Broadening params
    ('broadening_dict', ['molecule', 'modifiers', 'broadening'], True, "has_broadening", None, 'molecule', dict, None, None),
    ('broadening_type', ['molecule', 'modifiers', 'broadening', "type"], False, 'has_broadening', None, 'molecule', str, "Type of broadening (static or dynamic)", None),
    ('broadening_gam0', ['molecule', 'modifiers', 'broadening', "gam0"], False, 'has_broadening', 1.0, 'molecule', (int, float), "Base broadening strength", "a.u."),
    ('broadening_xi', ['molecule', 'modifiers', 'broadening', "xi"], False, 'has_broadening', 0.5, 'molecule', (int, float), "Energy-dependent broadening exponent", None),
    ('broadening_eps0', ['molecule', 'modifiers', 'broadening', "eps0"], False, 'has_broadening', 0.05, 'molecule', (int, float), "Reference energy for broadening", "a.u."),
    ('broadening_clamp', ['molecule', 'modifiers', 'broadening', "clamp"], False, 'has_broadening', 100, 'molecule', (int, float), "Maximum allowed broadening value", "a.u."),

    # Comparison mode params
    ('comparison_dict', ['molecule', 'modifiers', 'comparison'], True, "has_comparison", None, 'molecule', dict, None, None),
    ('comparison_bases', ['molecule', 'modifiers', 'comparison', 'bases'], False, 'has_comparison', None, 'molecule', list, "List of basis sets to compare", None),
    ('comparison_xcs', ['molecule', 'modifiers', 'comparison', 'xcs'], False, 'has_comparison', None, 'molecule', list, "List of exchange-correlation functionals to compare", None),
    ('comparison_lrc_parameters', ['molecule', 'modifiers', 'comparison', 'lrc_parameters'], False, 'has_comparison', None, 'molecule', dict, "Long-range correction parameters for RSH functionals", None),
    ('comparison_num_virtual', ['molecule', 'modifiers', 'comparison', 'num_virtual'], False, 'has_comparison', None, 'molecule', int, "Number of virtual orbitals to show in MO plot", None),
    ('comparison_num_occupied', ['molecule', 'modifiers', 'comparison', 'num_occupied'], False, 'has_comparison', None, 'molecule', int, "Number of occupied orbitals to show in MO plot", None),
    ('comparison_y_min', ['molecule', 'modifiers', 'comparison', 'y_min'], False, 'has_comparison', None, 'molecule', (int, float), "Minimum energy for MO plot (Hartree)", "Ha"),
    ('comparison_y_max', ['molecule', 'modifiers', 'comparison', 'y_max'], False, 'has_comparison', None, 'molecule', (int, float), "Maximum energy for MO plot (Hartree)", "Ha"),
    ('comparison_index_min', ['molecule', 'modifiers', 'comparison', 'index_min'], False, 'has_comparison', None, 'molecule', (int, float), "Lowest MO index to plot (1-indexed)", None),
    ('comparison_index_max', ['molecule', 'modifiers', 'comparison', 'index_max'], False, 'has_comparison', None, 'molecule', (int, float), "Highest MO index to plot (1-indexed)", None),
    ('comparison_dir_name', ['molecule', 'modifiers', 'comparison', 'dir_name'], False, 'has_comparison', f"img-{datetime.now().strftime('%m%d%Y_%H%M%S')}", 'molecule', str, "Directory name for comparison plots", None),

    # Checkpointing params
    ('checkpoint_dict', ['molecule', 'files', 'checkpoint'], True, "has_checkpoint", None, 'molecule', dict, None, None),
    ('checkpoint_filepath', ['molecule', 'files', 'checkpoint', 'filepath'], False, 'has_checkpoint', None, 'molecule', str, "Path to the checkpoint .npz file", None),
    ('checkpoint_snapshot_frequency', ['molecule', 'files', 'checkpoint', 'frequency'], False, 'has_checkpoint', None, 'molecule', int, "Number of timesteps between checkpoint saves", None),

    # Files
    ('field_e_filepath', ['molecule', 'files', 'field_e_filepath'], False, None, "field_e.csv", 'molecule', str, "File path for electric field at molecule position", None),
    ('field_p_filepath', ['molecule', 'files', 'field_p_filepath'], False, None, "field_p.csv", 'molecule', str, "File path for induced polarization at molecule position", None),
    ('spectra_e_vs_p_filepath', ['molecule', 'files', 'spectra_e_vs_p_filepath'], False, None, f"field_e_vs_p-{datetime.now().strftime('%m%d%Y_%H%M%S')}.png", 'molecule', str, "File path for electric vs polarization field plot", None),
]