import sys
import numpy as np
import logging
import _pickle
import os
from plasmol.utils.csv import read_field_csv, init_csv
from argparse import Namespace
import fcntl
import time
from contextlib import contextmanager
from pathlib import Path
logger = logging.getLogger("main")


def _get_per_dir_checkpoint(base_checkpoint_filepath: str, direction: str) -> str:
    """Return hidden per-direction checkpoint path for a base name.
    E.g. base="checkpoint.npz", direction="x" -> ".checkpoint_x.npz"
         base="final-checkpoint.npz", direction="z" -> ".final-checkpoint_z.npz"
    """
    if not base_checkpoint_filepath:
        return None
    base = base_checkpoint_filepath[1:] if base_checkpoint_filepath.startswith(".") else base_checkpoint_filepath
    stem, ext = os.path.splitext(base)
    return f".{stem}_{direction}{ext}"

def _build_checkpoint_base(params, for_direction: str = None) -> dict:
    """Build the invariant + pre-populated optional keys for a checkpoint dict.
    Does not write anything. Used by init and by per-direction final writers.

    If for_direction is given (and is_fourier), only the checkpoint_time and
    related suffixed entries for *that* direction are pre-populated; foreign
    direction suffixed keys are omitted so that per-dir final files don't
    contain "empty" values for other directions that would stomp good data on merge.
    """
    input_file_path = params.input_file_path
    with open(input_file_path, "rb") as f:
        input_file_content = f.read()

    if hasattr(params, 'geometry_xyz_filepath') and params.geometry_xyz_filepath:
        xyz_file_path = params.geometry_xyz_filepath
        with open(xyz_file_path, "rb") as f:
            xyz_content = f.read()
    else:
        xyz_file_path = None
        xyz_content = None

    is_fourier = getattr(params, 'has_fourier', False)

    save_dict = {
        "params_dict":         dict(params.__dict__),
        "input_file_path":     input_file_path,
        "input_file_content":  input_file_content,
        "is_fourier":          is_fourier,
        "is_open_shell":       getattr(params, "molecule_spin", 0) != 0,
        "updated_after_init":  False,
        "xyz_file_path":       xyz_file_path,
        "xyz_content":         xyz_content,
    }

    for dir in params.xyz:
        suffix = f"_{dir}" if is_fourier else ""
        if for_direction is None or dir == for_direction:
            save_dict[f"checkpoint_time{suffix}"] = 0.0

    for key in OPTIONAL_KEYS:
        if key not in save_dict:
            save_dict[key] = None

    # For per-direction final files, drop suffixed keys belonging to other directions
    # so their 0/None values cannot overwrite good data from other per-dir files during merge.
    if for_direction and is_fourier:
        other_dirs = [dd for dd in params.xyz if dd != for_direction]
        keys_to_drop = []
        for k in list(save_dict.keys()):
            for od in other_dirs:
                if k.endswith(f"_{od}") or k.endswith(f"_{od}_content"):
                    keys_to_drop.append(k)
                    break
        for k in keys_to_drop:
            save_dict.pop(k, None)

    save_dict.pop('allow_pickle', None)
    return save_dict


# =============================================================================
# Exactly what must be present in every checkpoint (field contents optional for
# backward compatibility with old checkpoints).
# =============================================================================
REQUIRED_CHECKPOINT_KEYS = {
    "params_dict",
    "input_file_path",
    "input_file_content",
    "is_fourier",
    "updated_after_init",
    "xyz_file_path",
    "xyz_content",
}

OPTIONAL_KEYS = {
    "checkpoint_time",
    "C_orth_ndt", "F_orth_n12dt", 
    "field_e_content", "field_p_content", 
    "D_ao_0", "mo_coeff", 
    # Fourier-specific (3 directions × 2 fields)
    "field_e_x_content", "field_p_x_content",
    "field_e_y_content", "field_p_y_content", 
    "field_e_z_content", "field_p_z_content",
    "C_orth_ndt_x", "C_orth_ndt_y", "C_orth_ndt_z",
    "F_orth_n12dt_x", "F_orth_n12dt_y", "F_orth_n12dt_z",
    "D_ao_0_x", "D_ao_0_y", "D_ao_0_z", 
    "mo_coeff_x", "mo_coeff_y", "mo_coeff_z", 
    "checkpoint_time_x", "checkpoint_time_y", "checkpoint_time_z",
    'is_open_shell'
}


@contextmanager
def _checkpoint_lock(checkpoint_path: str, timeout: float = 45.0):
    """Exclusive file lock + atomic write protection for the checkpoint .npz.
    Prevents race conditions when multiple Fourier directions (x/y/z)
    call init/add/update simultaneously on macOS/Linux.
    """
    if not checkpoint_path:
        yield
        return

    lock_path = str(checkpoint_path) + ".lock"
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)

    lock_fd = None
    start = time.time()
    try:
        lock_fd = open(lock_path, "w")
        while True:
            try:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() - start > timeout:
                    raise TimeoutError(f"Could not acquire checkpoint lock after {timeout}s")
                time.sleep(0.05)
        yield
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
            except:
                pass
            lock_fd.close()


def _get_restored_filepath(original_path, include_dir=True, restored_text="_restored"):
    """Append '_restored' to the filename (before the extension)."""
    if not original_path:
        return None
    path = str(original_path)
    dir_name = os.path.dirname(path)
    base_name = os.path.basename(path)
    name, ext = os.path.splitext(base_name)
    restored_name = f"{name}{restored_text}{ext}"
    return os.path.join(dir_name, restored_name) if dir_name and include_dir else restored_name


def init_checkpoint(params, final_checkpoint_filepath=None):
    """Initialize the checkpoint file with the invariant data only.
    This should be called once at the start (before any Fourier directions run).
    Creates the .npz with params_dict, input_file_path, input_file_content,
    is_fourier, and all optional keys pre-initialized to None.
    """
    if final_checkpoint_filepath:
        checkpoint_path = final_checkpoint_filepath
    else:
        checkpoint_path = params.checkpoint_filepath

    checkpoint_path = "".join([".", checkpoint_path])

    # Load existing checkpoint
    if os.path.exists(checkpoint_path):
        try:
            _ = np.load(checkpoint_path, allow_pickle=True)
        except Exception as e:
            raise RuntimeError(f"Error occurred while checking existing checkpoint: {e}")
        finally:
            params.checkpoint_filepath = _get_restored_filepath(params.checkpoint_filepath, restored_text="_new")
            logger.warning(f"Previous checkpoint file exists, moving new one to {params.checkpoint_filepath}")
            checkpoint_path = params.checkpoint_filepath

    save_dict = _build_checkpoint_base(params)

    with _checkpoint_lock(checkpoint_path):
        np.savez(checkpoint_path, allow_pickle=True, **save_dict)

    logger.debug(f"Initialized checkpoint to {checkpoint_path}")


def add_field_e_checkpoint(params, field_e_filepath, final_checkpoint_filepath=None):
    """
    Add the electric field content to the checkpoint.
    For Fourier final checkpoints, each direction writes its own hidden per-dir file
    (no locking) so the parent can merge later without races.
    """
    if final_checkpoint_filepath:
        checkpoint_path = final_checkpoint_filepath
    else:
        checkpoint_path = params.checkpoint_filepath

    is_fourier = getattr(params, 'has_fourier', False) or bool(getattr(params, 'molecule_source_component', None))
    dir_component = getattr(params, 'molecule_source_component', None) if is_fourier else None

    # Per-direction hidden checkpoint (Fourier only): write directly to own file, no lock.
    # This covers both the regular checkpoint path and the final-checkpoint path.
    if dir_component:
        if final_checkpoint_filepath:
            per_path = _get_per_dir_checkpoint(final_checkpoint_filepath, dir_component)
            kind = "final"
        else:
            # regular checkpoint for this Fourier direction
            reg_base = params.checkpoint_filepath
            per_path = _get_per_dir_checkpoint(reg_base, dir_component)
            kind = "regular"
        save_dict = _build_checkpoint_base(params, for_direction=dir_component)
        with open(field_e_filepath, "rb") as f:
            save_dict[f"field_e_{dir_component}_content"] = f.read()
        save_dict.pop('allow_pickle', None)
        np.savez(per_path, allow_pickle=True, **save_dict)
        logger.debug(f"Wrote per-direction field_e {kind} checkpoint for {dir_component}: {per_path}")
        return

    # Non-Fourier path: original locked behavior on the (hidden) checkpoint.
    checkpoint_path = "".join([".", checkpoint_path])

    # === Locked read-modify-write ===
    with _checkpoint_lock(checkpoint_path):
        if os.path.exists(checkpoint_path):
            loaded = np.load(checkpoint_path, allow_pickle=True)
            save_dict = {key: loaded[key] for key in loaded.files}
            loaded.close()
        else:
            raise FileNotFoundError("Checkpoint file not found during update")
        
        is_fourier = save_dict["is_fourier"]
        dir_component = getattr(params, 'molecule_source_component') if is_fourier else None
        suffix = f"_{dir_component}" if is_fourier and dir_component else ""

        with open(field_e_filepath, "rb") as f:
            save_dict[f"field_e{suffix}_content"] = f.read()
    
        # In case 'allow_pickle=True' is in the save_dict
        save_dict.pop('allow_pickle', None)
        np.savez(checkpoint_path, allow_pickle=True, **save_dict)


def update_checkpoint(params, molecule, checkpoint_time, final_checkpoint_filepath=None):
    """
    Update only the dynamic/simulation-state values for the current direction.
    Loads the existing checkpoint first, merges the new data, then writes it back.
    In Fourier mode this guarantees that data for the other two directions is preserved.

    For Fourier *final* checkpoints (final_checkpoint_filepath provided + molecule_source_component),
    we bypass locking entirely: each direction writes its own hidden per-dir file
    (e.g. .final-checkpoint_x.npz). The parent Fourier driver merges them in a finally block.
    """
    if final_checkpoint_filepath:
        checkpoint_path = final_checkpoint_filepath
    else:
        checkpoint_path = params.checkpoint_filepath

    method = getattr(params, "propagator", getattr(params, "molecule_propagator_str", "")).lower()
    is_fourier = getattr(params, 'has_fourier', False)

    # Determine current direction (x/y/z) only in Fourier mode
    dir_component = getattr(params, 'molecule_source_component') if is_fourier else None

    # === Per-direction hidden (Fourier): no lock, each direction writes its own file ===
    # This now covers BOTH regular periodic checkpoints and final checkpoints for Fourier runs.
    if dir_component:
        if final_checkpoint_filepath:
            per_path = _get_per_dir_checkpoint(final_checkpoint_filepath, dir_component)
            kind = "final"
        else:
            per_path = _get_per_dir_checkpoint(params.checkpoint_filepath, dir_component)
            kind = "regular"

        if os.path.exists(per_path):
            loaded = np.load(per_path, allow_pickle=True)
            save_dict = {}
            for key in loaded.files:
                val = loaded[key]
                if isinstance(val, np.ndarray):
                    save_dict[key] = val.copy()
                elif hasattr(val, "copy"):
                    save_dict[key] = val.copy()
                else:
                    save_dict[key] = val
            loaded.close()
        else:
            save_dict = _build_checkpoint_base(params, for_direction=dir_component)

        # Ensure field_e content for this dir is present (add_field_e_checkpoint should have
        # written it for the first write, but be defensive).
        e_key = f"field_e_{dir_component}_content"
        if e_key not in save_dict or save_dict.get(e_key) is None:
            fe_path = getattr(params, f"field_e_{dir_component}_filepath", None)
            if fe_path and os.path.exists(fe_path):
                try:
                    with open(fe_path, "rb") as f:
                        save_dict[e_key] = f.read()
                except Exception as e:
                    logger.warning(f"Could not read field_e for per-dir {kind} checkpoint ({dir_component}): {e}")

        # === Embed the CSV files as exact raw bytes (direction-aware) ===
        dir_path = f"{dir_component}_dir" if is_fourier and dir_component else ''

        for name, filepath_attr in [("e", "field_e_filepath"), ("p", "field_p_filepath")]:
            base_filename = getattr(params, filepath_attr, f"field_{name}.csv").split(os.sep)[-1] or f"field_{name}.csv"
            filepath = os.path.join(dir_path, base_filename) if dir_path else getattr(params, f"field_{name}_filepath", None)
            content_key = f"field_{name}_{dir_component}_content" if is_fourier and dir_component else f"field_{name}_content"

            if filepath and os.path.exists(filepath):
                try:
                    with open(filepath, "rb") as f:
                        save_dict[content_key] = f.read()
                except Exception as e:
                    logger.error(f"Failed to read {filepath} for checkpoint: {e}")
                    raise RuntimeError(f"Cannot update checkpoint - unable to read {name} file") from e
            else:
                logger.warning(f"{name} file for {dir_component or 'non-fourier'} ('{filepath}') not found - keeping existing content")

        # === Update molecule / propagator state for the current direction ===
        suffix = f"_{dir_component}" if is_fourier and dir_component else ""

        save_dict[f"checkpoint_time{suffix}"] = checkpoint_time
        save_dict[f"D_ao_0{suffix}"] = molecule.D_ao_0
        save_dict[f"mo_coeff{suffix}"] = molecule.mf.mo_coeff
        if method == "step":
            save_dict[f"C_orth_ndt{suffix}"] = molecule.C_orth_ndt
        elif method == "magnus2":
            save_dict[f"F_orth_n12dt{suffix}"] = molecule.F_orth_n12dt
        save_dict["updated_after_init"] = True

        # Enforce core content
        missing = REQUIRED_CHECKPOINT_KEYS - set(save_dict.keys())
        if missing:
            raise RuntimeError(f"BUG: checkpoint is missing required keys: {missing}")
        
        save_dict.pop('allow_pickle', None)
        np.savez(per_path, allow_pickle=True, **save_dict)

        time_log_str = f"{'='*20} Updated per-dir {kind} checkpoint {per_path} at time = {checkpoint_time} (direction: {dir_component}) {'='*20}"
        logger.debug(time_log_str)
        return

    # === Original locked path for non-Fourier runs (and any other non-per-dir case) ===
    checkpoint_path = "".join([".", checkpoint_path])

    # === Locked read-modify-write (this was the crashing function) ===
    with _checkpoint_lock(checkpoint_path):
        if os.path.exists(checkpoint_path):
            loaded = np.load(checkpoint_path, allow_pickle=True)
            # Deep copy everything to avoid NumPy reference issues across processes
            save_dict = {}
            for key in loaded.files:
                val = loaded[key]
                if isinstance(val, np.ndarray):
                    save_dict[key] = val.copy()
                elif hasattr(val, "copy"):
                    save_dict[key] = val.copy()
                else:
                    save_dict[key] = val
            loaded.close()
        else:
            raise FileNotFoundError("Checkpoint file not found during update")
        
        # re-compute is_fourier/dir in case (kept for minimal diff in original branch)
        is_fourier = getattr(params, 'has_fourier', False)
        dir_component = getattr(params, 'molecule_source_component') if is_fourier else None

        # === Embed the CSV files as exact raw bytes (direction-aware) ===
        dir_path = f"{dir_component}_dir" if is_fourier and dir_component else ''

        for name, filepath_attr in [("e", "field_e_filepath"), ("p", "field_p_filepath")]:
            base_filename = getattr(params, filepath_attr, f"field_{name}.csv").split(os.sep)[-1] or f"field_{name}.csv"
            filepath = os.path.join(dir_path, base_filename) if dir_path else getattr(params, f"field_{name}_filepath", None)
            content_key = f"field_{name}_{dir_component}_content" if is_fourier and dir_component else f"field_{name}_content"

            if filepath and os.path.exists(filepath):
                try:
                    with open(filepath, "rb") as f:
                        save_dict[content_key] = f.read()
                except Exception as e:
                    logger.error(f"Failed to read {filepath} for checkpoint: {e}")
                    raise RuntimeError(f"Cannot update checkpoint - unable to read {name} file") from e
            else:
                logger.warning(f"{name} file for {dir_component or 'non-fourier'} ('{filepath}') not found - keeping existing content")

        # === Update molecule / propagator state for the current direction ===
        suffix = f"_{dir_component}" if is_fourier and dir_component else ""

        save_dict[f"checkpoint_time{suffix}"] = checkpoint_time
        save_dict[f"D_ao_0{suffix}"] = molecule.D_ao_0
        save_dict[f"mo_coeff{suffix}"] = molecule.mf.mo_coeff
        if method == "step":
            save_dict[f"C_orth_ndt{suffix}"] = molecule.C_orth_ndt
        elif method == "magnus2":
            save_dict[f"F_orth_n12dt{suffix}"] = molecule.F_orth_n12dt
        save_dict["updated_after_init"] = True

        # Enforce core content
        missing = REQUIRED_CHECKPOINT_KEYS - set(save_dict.keys())
        if missing:
            raise RuntimeError(f"BUG: checkpoint is missing required keys: {missing}")
        
        # In case 'allow_pickle=True' is in the save_dict
        save_dict.pop('allow_pickle', None)
            
        # remove "." at front of temp file name
        checkpoint_path = checkpoint_path[1:]
        np.savez(checkpoint_path, allow_pickle=True, **save_dict)

    time_log_str = f"{'='*20} Updated checkpoint file {checkpoint_path} at time = {checkpoint_time} "
    time_log_str += f"(direction: {dir_component}) " if is_fourier else ""
    time_log_str += f"{'='*20}"
    logger.debug(time_log_str)


def resume_from_checkpoint(args):
    """
    Load checkpoint and **mutate the passed params object in place**.
    Supports both 'checkpoint_filepath' and 'checkpoint' attributes from CLI/params.
    Safe to call multiple times (idempotent after first load).
    """
    fn = getattr(args, "checkpoint", None)
    if not fn:
        raise ValueError("args.checkpoint is not set - cannot resume")

    try:
        data = dict(np.load(fn, allow_pickle=True))
    except FileNotFoundError:
        logger.error(f"Checkpoint file {fn} not found.")
        raise
    except (_pickle.UnpicklingError, ValueError, OSError, KeyError) as e:
        logger.error(f"{fn} is not a valid checkpoint archive: {e}")
        raise
    logger.info(f"Loading checkpoint.")

    # === Construct restored input file ===
    restored_input_filepath = _get_restored_filepath(data["input_file_path"], False, restored_text="_restored")
    content = np.ndarray.item(data["input_file_content"])
    with open(restored_input_filepath, "w", encoding="utf-8") as f:
        f.write(content.decode("utf-8"))
    logger.debug(f"Input file reconstructed from checkpoint: {restored_input_filepath}")

    if (data["xyz_file_path"] != None):
        restored_xyz_filepath = _get_restored_filepath(data["xyz_file_path"], False, restored_text="_restored")
        content = np.ndarray.item(data["xyz_content"])
        with open(restored_xyz_filepath, "w", encoding="utf-8") as f:
            f.write(content.decode("utf-8"))
        logger.debug(f"Geometry xyz file reconstructed from checkpoint: {restored_xyz_filepath}")

    # === Validate core content ===
    loaded_keys = set(data.keys())
    missing = REQUIRED_CHECKPOINT_KEYS - loaded_keys
    extra = loaded_keys - REQUIRED_CHECKPOINT_KEYS - OPTIONAL_KEYS
    if missing:
        raise RuntimeError(f"Checkpoint is missing required keys: {missing}")
    if extra:
        logger.warning(f"Checkpoint contains unexpected extra keys (ignored): {extra}")

    is_fourier = bool(data.get("is_fourier", False))
    if is_fourier:
        logger.info("Fourier checkpoint detected (contains data for x/y/z directions).")

    # === Build Params dictionary ===
    saved_params_dict = data["params_dict"].item()
    params = Namespace(**saved_params_dict)
    params.resume_from_checkpoint = True

    # === Check Open/Closed Shell ===
    saved_open_shell = bool(data.get("is_open_shell", False))
    if saved_open_shell != (saved_params_dict.get("molecule_spin", 0) != 0):
        raise RuntimeError("Checkpoint open-shell flag inconsistent with molecule_spin")

    # === Restore CSV files ===
    if is_fourier:
        checkpoint_dirs = [d for d in params.xyz 
                           if float(data.get(f"checkpoint_time_{d}", 0.0)) > 1e-12]
        if len(checkpoint_dirs) < len(params.xyz):
            missing = set(params.xyz) - set(checkpoint_dirs)
            logger.info(f"No checkpoint data found for direction(s): {missing}")
        params.not_checkpointed_dirs = list(set(params.xyz) - set(checkpoint_dirs))
        name_set = ["field_e_x", "field_e_y", "field_e_z", "field_p_x", "field_p_y", "field_p_z"]
    else:
        name_set = ["field_e", "field_p"]

    for name in name_set:
        content_key = f"{name}_content"
        if content_key in data and data[content_key] is not None:
            original_filepath = getattr(params, f"{name}_filepath", None)
            if original_filepath:
                restored_filepath = _get_restored_filepath(original_filepath, restored_text="")
                try:
                    os.makedirs(os.path.dirname(restored_filepath) or ".", exist_ok=True)
                    if is_fourier and name in [f"field_p_{d}" for d in params.not_checkpointed_dirs]:
                        with open(restored_filepath, "wb") as f:
                            init_csv(restored_filepath, "Molecule's Polarizability Field intensity in atomic units")
                    else:
                        with open(restored_filepath, "wb") as f:
                            f.write(data[content_key])
                    setattr(params, f"{name}_filepath", restored_filepath)
                    logger.info(f"Restored {name} CSV from checkpoint → {restored_filepath}")
                except Exception as e:
                    logger.error(f"Failed to write restored {name} CSV: {e}")
                    raise
        else:
            logger.debug(f"No {name}_content in checkpoint (old checkpoint)")

    params.values_from_checkpoint = {}
    method = getattr(params, "propagator", getattr(params, "molecule_propagator_str", "")).lower()
    if is_fourier:
        for dir in params.xyz:
            params.values_from_checkpoint[f"checkpoint_time_{dir}"] = float(data[f"checkpoint_time_{dir}"])
            if dir in checkpoint_dirs:
                params.values_from_checkpoint[f"D_ao_0_{dir}"] = data[f"D_ao_0_{dir}"]
                params.values_from_checkpoint[f"mo_coeff_{dir}"] = data[f"mo_coeff_{dir}"]
                if method == "step":
                    params.values_from_checkpoint[f"C_orth_ndt_{dir}"] = data[f"C_orth_ndt_{dir}"]
                elif method == "magnus2":
                    params.values_from_checkpoint[f"F_orth_n12dt_{dir}"] = data[f"F_orth_n12dt_{dir}"]
    else:
        params.values_from_checkpoint["checkpoint_time"] = float(data["checkpoint_time"])
        params.values_from_checkpoint["D_ao_0"] = data["D_ao_0"]
        params.values_from_checkpoint["mo_coeff"] = data["mo_coeff"]
        if method == "step":
            params.values_from_checkpoint["C_orth_ndt"] = data["C_orth_ndt"]
        elif method == "magnus2":
            params.values_from_checkpoint["F_orth_n12dt"] = data["F_orth_n12dt"]

    # === Final validation ===
    try:
        if is_fourier:
            for dir in params.xyz:
                _ = read_field_csv(getattr(params, f"field_e_{dir}_filepath"))
            for dir in checkpoint_dirs:
                _ = read_field_csv(getattr(params, f"field_p_{dir}_filepath"))
        else:
            _ = read_field_csv(params.field_e_filepath)
            _ = read_field_csv(params.field_p_filepath)
        logger.debug("Successfully validated both field CSV files")
    except Exception as e:
        logger.error(f"Field CSV validation failed: {e}")
        raise RuntimeError("Checkpoint restore succeeded but field CSVs cannot be read") from e

    params.resumed_from_checkpoint = True
    if is_fourier:
        logger.info(f"Loaded fourier checkpoint with data at "
                    f"t_x={params.values_from_checkpoint['checkpoint_time_x']} au, "
                    f"t_y={params.values_from_checkpoint['checkpoint_time_y']} au, "
                    f"t_z={params.values_from_checkpoint['checkpoint_time_z']} au.")
    else:
        logger.info(f"Loaded checkpoint with data at t={params.values_from_checkpoint['checkpoint_time']} au.")

    # print all keys within values_from_checkpoint
    return params


def merge_per_direction_checkpoints(params, checkpoint_filepath):
    """General merge of per-direction hidden checkpoints into a single combined one.

    Works for both regular checkpoints (e.g. "checkpoint.npz" -> per-dir ".checkpoint_x.npz" etc.)
    and final checkpoints (e.g. "final-checkpoint.npz").

    The merge is careful to only pull direction-specific suffixed keys from the per-dir file
    that "owns" that direction. This prevents 0/None placeholder values (from each per-dir
    file's private base skeleton) from stomping good data from sibling per-dir files.

    Intended to be called from the parent Fourier driver's finally block so it runs even
    on crashes/partial failures. The combined result is written as both hidden (for cleanup)
    and visible (for resume / artifacts).
    """
    if not checkpoint_filepath:
        return

    directions = getattr(params, 'xyz', ['x', 'y', 'z'])
    merged = {}
    merged_any = False

    def _key_belongs_to_direction(key: str, direction: str) -> bool:
        """Return True for common keys or keys whose suffix matches this direction."""
        for d in ('x', 'y', 'z'):
            if key.endswith(f"_{d}") or key.endswith(f"_{d}_content"):
                return d == direction
        return True

    for d in directions:
        per_path = _get_per_dir_checkpoint(checkpoint_filepath, d)
        if not per_path or not os.path.exists(per_path):
            logger.info(f"No per-dir checkpoint found for direction {d} ({per_path})")
            continue
        try:
            loaded = np.load(per_path, allow_pickle=True)
            for k in loaded.files:
                if not _key_belongs_to_direction(k, d):
                    continue
                val = loaded[k]
                if isinstance(val, np.ndarray):
                    merged[k] = val.copy()
                elif hasattr(val, "copy"):
                    merged[k] = val.copy()
                else:
                    merged[k] = val
            loaded.close()
            merged_any = True
            logger.debug(f"Included per-dir checkpoint {per_path} for direction {d}")
        except Exception as e:
            logger.warning(f"Failed to load per-dir checkpoint {per_path}: {e}")

    if not merged_any:
        logger.warning(f"merge_per_direction_checkpoints: no per-direction files for base {checkpoint_filepath}; nothing to merge.")
        return

    merged.pop('allow_pickle', None)

    # Guarantee checkpoint_time_* for all dirs so resume code that does direct lookups succeeds.
    for d in directions:
        k = f"checkpoint_time_{d}"
        if k not in merged:
            merged[k] = 0.0

    # Write hidden + visible (consistent with prior behavior)
    base = checkpoint_filepath[1:] if checkpoint_filepath.startswith(".") else checkpoint_filepath
    hidden_path = "." + base
    np.savez(hidden_path, allow_pickle=True, **merged)

    visible_path = base
    np.savez(visible_path, allow_pickle=True, **merged)

    # Best-effort flag (final case)
    if "final" in base.lower():
        try:
            params.final_checkpoint_written_after_init = True
        except Exception:
            pass

    logger.info(f"Merged per-dir checkpoints for {base} -> {visible_path} (hidden: {hidden_path})")


def merge_final_checkpoints(params, final_checkpoint_filepath=None):
    """Backward-compatible wrapper for final checkpoints."""
    if final_checkpoint_filepath is None:
        final_checkpoint_filepath = getattr(params, 'final_checkpoint_filepath', None)
    merge_per_direction_checkpoints(params, final_checkpoint_filepath)


def cleanup_checkpoint(params):
    # Regular checkpoint hidden
    if hasattr(params, 'checkpoint_filepath') and params.checkpoint_filepath:
        checkpoint_path = "".join([".", params.checkpoint_filepath])
        if os.path.isfile(checkpoint_path):
            try:
                os.remove(checkpoint_path)
            except Exception:
                pass

    # Final checkpoint hidden (the combined one)
    final_base = getattr(params, 'final_checkpoint_filepath', None)
    if final_base:
        final_hidden = "." + (final_base[1:] if final_base.startswith(".") else final_base)
        if os.path.isfile(final_hidden):
            try:
                os.remove(final_hidden)
            except Exception:
                pass

        # Per-direction hidden finals (Fourier no-lock scheme)
        for d in getattr(params, 'xyz', ['x', 'y', 'z']):
            p = _get_per_dir_checkpoint(final_base, d)
            if p and os.path.isfile(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    # Regular checkpoint hidden (the combined one) + its per-direction hiddens
    reg_base = getattr(params, 'checkpoint_filepath', None)
    if reg_base:
        reg_hidden = "." + (reg_base[1:] if reg_base.startswith(".") else reg_base)
        if os.path.isfile(reg_hidden):
            try:
                os.remove(reg_hidden)
            except Exception:
                pass

        for d in getattr(params, 'xyz', ['x', 'y', 'z']):
            p = _get_per_dir_checkpoint(reg_base, d)
            if p and os.path.isfile(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    # Legacy lock files
    for filename in os.listdir("."):
        if filename.endswith(".lock"):
            try:
                os.remove(filename)
            except Exception:
                pass
