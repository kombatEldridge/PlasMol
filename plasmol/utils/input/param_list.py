from datetime import datetime

# Each entry is a tuple: (attribute_name, path_as_list, section_condition, data_types)
# - attribute_name: str, the name to set as self.<attribute_name>
# - path_as_list: list of str, the nested keys to access in parsed (e.g., ['settings', 'dt'])
# - section_condition: str or None, only set if self.type in ['plasmon', 'molecule'] matches or None for always
# - data_types: datatype or tuple
param_defs = [
    # Settings (always required)
    ('dt', ['settings', 'dt'], False, None, None, None, (int, float)),
    ('t_end', ['settings', 't_end'], False, None, None, None, (int, float)),

    # Plasmon params
    ('plasmon_dict', ['plasmon'], True, "has_plasmon", None, 'plasmon', dict),

    # Plasmon simulation params
    ('plasmon_simulation_dict', ['plasmon', 'simulation'], True, 'has_simulation', None, 'plasmon', dict),
    ('plasmon_tolerance_field_e', ['plasmon', 'simulation', "tolerance_field_e"], False, 'has_simulation', 1e-12, 'plasmon', (int, float)),
    ('plasmon_cell_length', ['plasmon', 'simulation', "cell_length"], False, 'has_simulation', 0.1, 'plasmon', (int, float)),
    ('plasmon_cell_volume', ['plasmon', 'simulation', "cell_volume"], False, 'has_simulation', None, 'plasmon', (int, float)),
    ('plasmon_pml_thickness', ['plasmon', 'simulation', "pml_thickness"], False, 'has_simulation', 0.01, 'plasmon', (int, float)),
    ('plasmon_symmetries', ['plasmon', 'simulation', 'symmetries'], False, 'has_simulation', None, 'plasmon', list),
    ('plasmon_surrounding_material_index', ['plasmon', 'simulation', "surrounding_material_index"], False, 'has_simulation', 1.33, 'plasmon', (int, float)),
    
    # Plasmon source params
    ('plasmon_source_dict', ['plasmon', 'source'], True, "has_plasmon_source", None, 'plasmon', dict),
    ('plasmon_source_type', ['plasmon', 'source', 'type'], False, 'has_plasmon_source', None, 'plasmon', str),
    ('plasmon_source_center', ['plasmon', 'source', 'center'], False, 'has_plasmon_source', None, 'plasmon', list),
    ('plasmon_source_size', ['plasmon', 'source', 'size'], False, 'has_plasmon_source', None, 'plasmon', list),
    ('plasmon_source_component', ['plasmon', 'source', 'component'], False, 'has_plasmon_source', None, 'plasmon', str),
    ('plasmon_source_amplitude', ['plasmon', 'source', 'amplitude'], False, 'has_plasmon_source', 1, 'plasmon', (int, float)),
    ('plasmon_source_is_integrated', ['plasmon', 'source', 'is_integrated'], False, 'has_plasmon_source', True, 'plasmon', bool),
    ('plasmon_source_additional_parameters', ['plasmon', 'source', 'additional_parameters'], False, 'has_plasmon_source', None, 'plasmon', dict),

    # Nanoparticle params
    ('nanoparticle_dict', ['plasmon', 'nanoparticle'], True, "has_nanoparticle", None, 'plasmon', dict),
    ('nanoparticle_material', ['plasmon', 'nanoparticle', 'material'], False, 'has_nanoparticle', None, 'plasmon', str),
    ('nanoparticle_radius', ['plasmon', 'nanoparticle', 'radius'], False, 'has_nanoparticle', None, 'plasmon', (int, float)),
    ('nanoparticle_center', ['plasmon', 'nanoparticle', 'center'], False, 'has_nanoparticle', [0,0,0], 'plasmon', list),

    # Images params
    ('images_dict', ['plasmon', 'images'], True, "has_images", None, 'plasmon', dict),
    ('images_timesteps_between', ['plasmon', 'images', 'timesteps_between'], False, 'has_images', None, 'plasmon', int),
    ('images_additional_parameters', ['plasmon', 'images', 'additional_parameters'], False, 'has_images', ["-Zc dkbluered", "-S 10"], 'plasmon', list),
    ('images_dir_name', ['plasmon', 'images', 'dir_name'], False, 'has_images', f"plasmol-{datetime.now().strftime('%m%d%Y_%H%M%S')}", 'plasmon', str),
    ('images_make_gif', ['plasmon', 'images', 'make_gif'], False, 'has_images', True, 'plasmon', bool),

    # Molecule position param
    ('plasmol_molecule_position', ['plasmon', 'molecule_position'], True, "has_molecule_position", None, 'plasmon', list),

    # Molecule params
    ('molecule_dict', ['molecule'], True, "has_molecule", None, 'molecule', dict),
    ('molecule_geometry', ['molecule', 'geometry'], False, 'has_molecule', None, 'molecule', list),
    ('molecule_geometry_units', ['molecule', 'geometry_units'], False, 'has_molecule', None, 'molecule', str),
    ('molecule_basis', ['molecule', 'basis'], False, 'has_molecule', None, 'molecule', str),
    ('molecule_charge', ['molecule', 'charge'], False, 'has_molecule', None, 'molecule', int),
    ('molecule_spin', ['molecule', 'spin'], False, 'has_molecule', None, 'molecule', int),
    ('molecule_xc', ['molecule', 'xc'], False, 'has_molecule', None, 'molecule', str),
    ('molecule_lrc_parameter', ['molecule', 'lrc_parameter'], False, 'has_molecule', None, 'molecule', float),
    ('molecule_propagator_str', ['molecule', 'propagator', "type"], False, 'has_molecule', 'magnus2', 'molecule', str),
    ('molecule_pc_convergence', ['molecule', 'propagator', "pc_convergence"], False, 'has_molecule', 1e-12, 'molecule', (int, float)),
    ('molecule_max_iterations', ['molecule', 'propagator', "max_iterations"], False, 'has_molecule', 200, 'molecule', int),
    ('molecule_hermiticity_tolerance', ['molecule', 'hermiticity_tolerance'], False, 'has_molecule', 1e-12, 'molecule', (int, float)),

    # Source params (molecule section)
    ('molecule_source_dict', ['molecule', 'source'], True, "has_molecule_source", None, 'molecule', dict),
    ('molecule_source_type', ['molecule', 'source', 'type'], False, 'has_molecule_source', None, 'molecule', str),
    ('molecule_source_intensity', ['molecule', 'source', 'intensity'], False, 'has_molecule_source', None, 'molecule', (int, float)),
    ('molecule_source_peak_time', ['molecule', 'source', 'peak_time'], False, 'has_molecule_source', None, 'molecule', (int, float)),
    ('molecule_source_width_steps', ['molecule', 'source', 'width_steps'], False, 'has_molecule_source', None, 'molecule', int),
    ('molecule_source_component', ['molecule', 'source', 'component'], False, 'has_molecule_source', None, 'molecule', str),
    ('molecule_source_additional_parameters', ['molecule', 'source', 'additional_parameters'], False, 'has_molecule_source', None, 'molecule', dict),

    # Fourier params runs three sims at once, one per axis
    ('fourier_dict', ['molecule', 'modifiers', 'fourier'], True, "has_fourier", None, 'molecule', dict),
    ('fourier_gamma', ['molecule', 'modifiers', 'fourier', 'gamma'], False, 'has_fourier', 0.01, 'molecule', (int, float)),
    ('fourier_npz_filepath', ['molecule', 'modifiers', 'fourier', 'npz_filepath'], False, 'has_fourier', None, 'molecule', str),
    ('fourier_spectrum_filepath', ['molecule', 'modifiers', 'fourier', 'spectrum_filepath'], False, 'has_fourier', None, 'molecule', str),
    ('fourier_damping_gamma', ['molecule', 'modifiers', 'fourier', 'damping_gamma'], False, 'has_fourier', None, 'molecule', (int, float)),

    # Lopata Broadening params
    ('broadening_dict', ['molecule', 'modifiers', 'broadening'], True, "has_broadening", None, 'molecule', dict),
    ('broadening_type', ['molecule', 'modifiers', 'broadening', "type"], False, 'has_broadening', None, 'molecule', str),
    ('broadening_gam0', ['molecule', 'modifiers', 'broadening', "gam0"], False, 'has_broadening', 1.0, 'molecule', (int, float)),
    ('broadening_xi', ['molecule', 'modifiers', 'broadening', "xi"], False, 'has_broadening', 0.5, 'molecule', (int, float)),
    ('broadening_eps0', ['molecule', 'modifiers', 'broadening', "eps0"], False, 'has_broadening', 0.05, 'molecule', (int, float)),
    ('broadening_clamp', ['molecule', 'modifiers', 'broadening', "clamp"], False, 'has_broadening', 100, 'molecule', (int, float)),
    
    # Comparison mode params
    ('comparison_dict', ['molecule', 'modifiers', 'comparison'], True, "has_comparison", None, 'molecule', dict),
    ('comparison_bases', ['molecule', 'modifiers', 'comparison', 'bases'], False, 'has_comparison', None, 'molecule', list),
    ('comparison_xcs', ['molecule', 'modifiers', 'comparison', 'xcs'], False, 'has_comparison', None, 'molecule', list),
    ('comparison_lrc_parameters', ['molecule', 'modifiers', 'comparison', 'lrc_parameters'], False, 'has_comparison', None, 'molecule', dict),
    ('comparison_num_virtual', ['molecule', 'modifiers', 'comparison', 'num_virtual'], False, 'has_comparison', None, 'molecule', int),
    ('comparison_num_occupied', ['molecule', 'modifiers', 'comparison', 'num_occupied'], False, 'has_comparison', None, 'molecule', int),
    ('comparison_y_min', ['molecule', 'modifiers', 'comparison', 'y_min'], False, 'has_comparison', None, 'molecule', (int, float)),
    ('comparison_y_max', ['molecule', 'modifiers', 'comparison', 'y_max'], False, 'has_comparison', None, 'molecule', (int, float)),
    ('comparison_index_min', ['molecule', 'modifiers', 'comparison', 'index_min'], False, 'has_comparison', None, 'molecule', (int, float)),
    ('comparison_index_max', ['molecule', 'modifiers', 'comparison', 'index_max'], False, 'has_comparison', None, 'molecule', (int, float)),
    ('comparison_dir_name', ['molecule', 'modifiers', 'comparison', 'dir_name'], False, 'has_comparison', f"img-{datetime.now().strftime('%m%d%Y_%H%M%S')}", 'molecule', str),

    # Checkpointing params
    ('checkpoint_dict', ['molecule', 'files', 'checkpoint'], True, "has_checkpoint", None, 'molecule', dict),
    ('checkpoint_filepath', ['molecule', 'files', 'checkpoint', 'filepath'], False, 'has_checkpoint', None, 'molecule', str),
    ('checkpoint_snapshot_frequency', ['molecule', 'files', 'checkpoint', 'frequency'], False, 'has_checkpoint', None, 'molecule', int),

    # Files
    ('field_e_filepath', ['molecule', 'files', 'field_e_filepath'], False, None, "field_e.csv", 'molecule', str),
    ('field_p_filepath', ['molecule', 'files', 'field_p_filepath'], False, None, "field_p.csv", 'molecule', str),
    ('spectra_e_vs_p_filepath', ['molecule', 'files', 'spectra_e_vs_p_filepath'], False, None, f"field_e_vs_p-{datetime.now().strftime('%m%d%Y_%H%M%S')}.png", 'molecule', str),
]
