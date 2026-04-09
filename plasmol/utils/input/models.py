from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class SettingsModel(BaseModel):
    """Global simulation settings (always required)."""
    dt: float = Field(
        ...,
        gt=0,
        description="Time step in atomic units [Units: a.u.] [Required]",
    )
    t_end: float = Field(
        ...,
        gt=0,
        description="End time in atomic units [Units: a.u.] [Required]",
    )

    model_config = ConfigDict(extra="forbid")


class PlasmonSimulationModel(BaseModel):
    """Plasmon (Meep) simulation parameters."""
    tolerance_field_e: float = Field(
        1e-12,
        gt=0,
        description="Minimum |E| before quantum propagation is triggered [Units: a.u.] [Default: 1e-12]",
    )
    cell_length: Optional[float] = Field(
        None,
        gt=0,
        description="Length of the simulation cell (used if cell_volume not provided) [Units: μm] [Default: 0.1]",
    )
    cell_volume: Optional[list[float]] = Field(
        None,
        min_length=3,
        max_length=3,
        description="Simulation cell volume (overrides cell_length if provided) [Units: μm] [Optional]",
    )
    pml_thickness: float = Field(
        0.01,
        gt=0,
        description="Thickness of the PML absorbing boundary layers [Units: μm] [Default: 0.01]",
    )
    symmetries: Optional[list] = Field(
        None,
        description="Symmetry operations (axis followed by phase) e.g. ['Y', 1, 'Z', -1] [Units: none] [Optional]",
    )
    surrounding_material_index: float = Field(
        1.33,
        ge=1.0,
        description="Refractive index of the surrounding medium [Units: none] [Default: 1.33]",
    )

    model_config = ConfigDict(extra="forbid")


class PlasmonSourceModel(BaseModel):
    """Source definition for the plasmon (Meep) simulation."""
    type: str = Field(
        ...,
        description="Type of source ('continuous', 'gaussian', or 'custom') [Units: none] [Required]",
    )
    center: list[float] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Center coordinates of the source [Units: μm] [Required]",
    )
    size: list[float] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Size of the source volume [Units: μm] [Required]",
    )
    component: Literal["x", "y", "z"] = Field(
        ...,
        description="Electric field component the source acts on [Units: none] [Required]",
    )
    amplitude: float = Field(
        1.0,
        description="Overall complex amplitude multiplying the source [Units: none] [Default: 1.0]",
    )
    is_integrated: bool = Field(
        True,
        description="Whether the source is integrated over time (dipole moment) [Units: none] [Default: True]",
    )
    additional_parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Type-specific parameters (frequency, wavelength, width, etc.) [Units: varies] [Default: {}]",
    )

    model_config = ConfigDict(extra="forbid")


class PlasmonNanoparticleModel(BaseModel):
    """Nanoparticle geometry (currently only spheres are supported)."""
    material: str = Field(
        ...,
        description="Material name from meep.materials (e.g. 'Au_JC_visible') [Units: none] [Required]",
    )
    radius: float = Field(
        ...,
        gt=0,
        description="Radius of the spherical nanoparticle [Units: μm] [Required]",
    )
    center: list[float] = Field(
        [0, 0, 0],
        min_length=3,
        max_length=3,
        description="Center coordinates of the nanoparticle [Units: μm] [Default: [0, 0, 0]]",
    )

    model_config = ConfigDict(extra="forbid")


class PlasmonImagesModel(BaseModel):
    """Output image / GIF settings for the Meep simulation."""
    timesteps_between: int = Field(
        ...,
        gt=0,
        description="Number of Meep timesteps between PNG frame outputs [Units: steps] [Required]",
    )
    additional_parameters: list[str] = Field(
        default=["-Zc dkbluered", "-S 10"],
        description="Additional arguments passed to Meep's output_png (color scale, etc.) [Units: none] [Default: ['-Zc dkbluered', '-S 10']]",
    )
    dir_name: str = Field(
        default_factory=lambda: f"plasmol-{datetime.now().strftime('%m%d%Y_%H%M%S')}",
        description="Directory name where PNG frames will be saved [Units: none] [Default: auto-generated]",
    )
    make_gif: bool = Field(
        True,
        description="Automatically create animated GIF from the PNG frames after simulation [Units: none] [Default: True]",
    )

    model_config = ConfigDict(extra="forbid")


class PlasmonModel(BaseModel):
    """Top-level plasmon (classical Meep) section."""
    simulation: PlasmonSimulationModel
    source: Optional[PlasmonSourceModel] = None
    nanoparticle: Optional[PlasmonNanoparticleModel] = None
    images: Optional[PlasmonImagesModel] = None
    molecule_position: Optional[list[float]] = Field(
        None,
        min_length=3,
        max_length=3,
        description="Position of the quantum molecule inside the Meep cell [Units: μm] [Optional]",
    )

    model_config = ConfigDict(extra="forbid")


class MoleculeGeometryEntry(BaseModel):
    atom: str = Field(..., description="Atomic symbol (e.g. 'O', 'H') [Units: none] [Required]")
    coord: list[float] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Cartesian coordinates of the atom [Units: geometry_units] [Required]",
    )


class MoleculePropagatorModel(BaseModel):
    type: Literal["step", "rk4", "magnus2"] = Field(
        "magnus2",
        description="Time-propagation algorithm [Units: none] [Default: magnus2]",
    )
    pc_convergence: float = Field(
        1e-12,
        gt=0,
        description="Predictor-corrector convergence threshold (Magnus2 only) [Units: a.u.] [Default: 1e-12]",
    )
    max_iterations: int = Field(
        200,
        gt=0,
        description="Maximum predictor-corrector iterations (Magnus2 only) [Units: none] [Default: 200]",
    )


class BroadeningModifierModel(BaseModel):
    type: Literal["static", "dynamic"] = Field(
        ...,
        description="Type of broadening (static or dynamic) [Units: none] [Required]",
    )
    gam0: float = Field(
        1.0,
        gt=0,
        description="Base broadening strength [Units: a.u.] [Default: 1.0]",
    )
    xi: float = Field(
        0.5,
        ge=0,
        description="Energy-dependent broadening exponent [Units: none] [Default: 0.5]",
    )
    eps0: float = Field(
        0.0477,
        ge=0,
        description="Reference energy for broadening [Units: a.u.] [Default: 0.0477]",
    )
    clamp: float = Field(
        100,
        gt=0,
        description="Maximum allowed broadening value [Units: a.u.] [Default: 100]",
    )


class FourierModifierModel(BaseModel):
    gamma: float = Field(
        ...,
        gt=0,
        description="Broadening factor for Fourier transformed spectrum [Units: a.u.] [Required]",
    )
    npz_filepath: str = Field(
        ...,
        description="File path for npz file containing imaginary absorption and frequencies [Units: none] [Required]",
    )
    spectrum_filepath: str = Field(
        ...,
        description="Output file path for the absorption spectrum plot [Units: none] [Required]",
    )
    damping_gamma: Optional[float] = Field(
        None,
        gt=0,
        description="Artificial damping applied to polarization field for better FFT resolution [Units: a.u.] [Optional]",
    )


class ComparisonModifierModel(BaseModel):
    bases: list[str] = Field(
        ...,
        description="List of basis sets to compare [Units: none] [Required]",
    )
    xcs: list[str] = Field(
        ...,
        description="List of exchange-correlation functionals to compare [Units: none] [Required]",
    )
    lrc_parameters: dict[str, float] = Field(
        default_factory=dict,
        description="Long-range correction parameters for RSH functionals [Units: none] [Default: {}]",
    )
    num_virtual: Optional[int] = Field(
        None,
        description="Number of virtual orbitals to show in MO plot [Units: none] [Optional]",
    )
    num_occupied: Optional[int] = Field(
        None,
        description="Number of occupied orbitals to show in MO plot [Units: none] [Optional]",
    )
    y_min: Optional[float] = Field(
        None,
        description="Minimum energy for MO plot (Hartree) [Units: Ha] [Optional]",
    )
    y_max: Optional[float] = Field(
        None,
        description="Maximum energy for MO plot (Hartree) [Units: Ha] [Optional]",
    )
    index_min: Optional[int] = Field(
        None,
        description="Lowest MO index to plot (1-indexed) [Units: none] [Optional]",
    )
    index_max: Optional[int] = Field(
        None,
        description="Highest MO index to plot (1-indexed) [Units: none] [Optional]",
    )
    dir_name: str = Field(
        default_factory=lambda: f"img-{datetime.now().strftime('%m%d%Y_%H%M%S')}",
        description="Directory name for comparison plots [Units: none] [Default: auto-generated]",
    )


class MoleculeModifiersModel(BaseModel):
    fourier: Optional[FourierModifierModel] = None
    broadening: Optional[BroadeningModifierModel] = None
    comparison: Optional[ComparisonModifierModel] = None


class MoleculeFilesCheckpointModel(BaseModel):
    frequency: int = Field(
        ...,
        gt=0,
        description="Number of timesteps between checkpoint saves [Units: steps] [Required]",
    )
    filepath: str = Field(
        ...,
        description="Path to the checkpoint .npz file [Units: none] [Required]",
    )


class MoleculeFilesModel(BaseModel):
    checkpoint: Optional[MoleculeFilesCheckpointModel] = None
    field_e_filepath: str = Field(
        "field_e.csv",
        description="File path for electric field at molecule position [Units: none] [Default: field_e.csv]",
    )
    field_p_filepath: str = Field(
        "field_p.csv",
        description="File path for induced polarization at molecule position [Units: none] [Default: field_p.csv]",
    )
    spectra_e_vs_p_filepath: Optional[str] = Field(
        None,
        description="File path for electric vs polarization field plot [Units: none] [Optional]",
    )


class MoleculeSourceModel(BaseModel):
    type: Literal["pulse", "kick"] = Field(
        ...,
        description="Shape of the external electric field source [Units: none] [Required]",
    )
    intensity: float = Field(
        ...,
        gt=0,
        description="Electric field intensity [Units: a.u.] [Required]",
    )
    peak_time: float = Field(
        ...,
        ge=0,
        description="Time at which the pulse/kick peaks [Units: a.u.] [Required]",
    )
    width_steps: int = Field(
        ...,
        gt=0,
        description="Width of the pulse in time steps [Units: steps] [Required]",
    )
    component: Literal["x", "y", "z"] = Field(
        ...,
        description="Direction of the electric field [Units: none] [Required]",
    )
    additional_parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters (wavelength or frequency for pulse) [Units: varies] [Default: {}]",
    )


class MoleculeModel(BaseModel):
    geometry: list[MoleculeGeometryEntry]
    geometry_units: Literal["angstrom", "bohr"] = Field(
        ...,
        description="Units of the geometry coordinates [Units: none] [Required]",
    )
    charge: int = Field(
        ...,
        description="Total molecular charge [Units: none] [Required]",
    )
    spin: int = Field(
        ...,
        description="Spin multiplicity minus one (0 = singlet) [Units: none] [Required]",
    )
    basis: str = Field(
        ...,
        description="Basis set name (e.g. '6-31g') [Units: none] [Required]",
    )
    xc: str = Field(
        ...,
        description="Exchange-correlation functional [Units: none] [Required]",
    )
    lrc_parameter: Optional[float] = Field(
        None,
        description="Long-range correction parameter (mu/omega) for RSH functionals [Units: a.u.] [Optional]",
    )
    propagator: MoleculePropagatorModel = Field(default_factory=MoleculePropagatorModel)
    hermiticity_tolerance: float = Field(
        1e-12,
        gt=0,
        description="Tolerance for checking Hermitian matrices [Units: a.u.] [Default: 1e-12]",
    )
    modifiers: Optional[MoleculeModifiersModel] = None
    source: Optional[MoleculeSourceModel] = None
    files: MoleculeFilesModel = Field(default_factory=MoleculeFilesModel)

    model_config = ConfigDict(extra="forbid")


class PlasMolInput(BaseModel):
    """Root model for the entire PlasMol input JSON."""
    settings: SettingsModel
    plasmon: Optional[PlasmonModel] = None
    molecule: Optional[MoleculeModel] = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_simulation_types(self) -> PlasMolInput:
        if not self.plasmon and not self.molecule:
            raise ValueError("At least one of 'plasmon' or 'molecule' section is required.")
        return self

def describe_parameters():
    """Pretty-print ALL parameters (fully expanded).
    - Type column shows the real base type (never 'Optional')
    - Hides any row where type ends with 'Model'
    - Default column is now clean (never shows "(required)" or "PydanticUndefined")
    - Required column shows the requirement status ("Yes"/"No")
    """
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        console = Console()
        use_rich = True
    except ImportError:
        use_rich = False
        print("Rich not installed → using clean plain text output.\n")

    from typing import get_origin, get_args, Optional, Union, List

    def is_required(field_info) -> bool:
        """Check if field is required (Field(...))."""
        return field_info.is_required()

    def get_default_str(field_info):
        """Return clean default string for the table.
        Fixes both reported issues:
          1. Required fields now show "—" (no more "(required)").
          2. default_factory fields (dict, lambda, etc.) no longer show PydanticUndefined.
        """
        if is_required(field_info):
            return "—"                                      # ← clean for required fields

        # Handle default_factory (the source of PydanticUndefined for non-required fields)
        if field_info.default_factory is not None:
            factory = field_info.default_factory
            if factory is dict:
                return "{}"
            if factory is list:
                return "[]"
            # lambda factories (e.g. datetime.now() for dir_name) or Model constructors
            try:
                val = factory()
                return str(val)
            except Exception:
                return "<auto-generated>"

        # Normal static default
        default = field_info.default
        if default is None:
            return "—"
        return str(default)

    def get_clean_type(annotation):
        """Return clean base type, stripping Optional/Union wrappers."""
        origin = get_origin(annotation)
        if origin is Union or origin is Optional:
            args = get_args(annotation)
            if len(args) == 2 and type(None) in args:
                annotation = next(a for a in args if a is not type(None))
        if hasattr(annotation, "__name__"):
            return annotation.__name__
        return str(annotation).replace("typing.", "").replace("<class '", "").replace("'>", "")

    def get_inner_model(annotation):
        """Safely extract the inner Pydantic model from Optional[list[Model]] etc."""
        origin = get_origin(annotation)
        if origin is Union or origin is Optional:
            args = get_args(annotation)
            annotation = next((a for a in args if a is not type(None)), annotation)

        # Fixed: original had duplicate "origin is list"
        if origin in (list, List):
            args = get_args(annotation)
            annotation = args[0] if args else annotation

        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return annotation
        return None

    def recurse_fields(model_class: type[BaseModel], prefix: str = "", table=None):
        for field_name, field_info in model_class.model_fields.items():
            full_name = f"{prefix}.{field_name}" if prefix else field_name

            field_type = get_clean_type(field_info.annotation)
            default_str = get_default_str(field_info)
            description = field_info.description or "No description"
            required_str = "Yes" if is_required(field_info) else "No"

            # Skip container rows that end with "Model"
            if field_type.endswith("Model"):
                inner_model = get_inner_model(field_info.annotation)
                if inner_model:
                    recurse_fields(inner_model, full_name, table)
                continue

            # Show the row
            if use_rich:
                table.add_row(full_name, field_type, default_str, description, required_str)
            else:
                print(f"  {full_name:<45} {field_type:<12} {default_str:<20} {description}   {required_str}")

            # Recurse into nested models
            inner_model = get_inner_model(field_info.annotation)
            if inner_model:
                recurse_fields(inner_model, full_name, table)

    if use_rich:
        table = Table(
            title="PlasMol — All Parameters (fully expanded)",
            box=box.ROUNDED,
            show_lines=True,
            title_style="bold cyan"
        )
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Type", style="magenta", justify="center")
        table.add_column("Default", style="yellow", justify="center")
        table.add_column("Description", style="green")
        table.add_column("Required", style="bright_red", justify="center")

        recurse_fields(PlasMolInput, table=table)
        console.print(table)
    else:
        print("PlasMol — All Parameters (fully expanded)\n")
        recurse_fields(PlasMolInput)