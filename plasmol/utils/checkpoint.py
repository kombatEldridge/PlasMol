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

logger = logging.getLogger("main")


# =============================================================================
# Exactly what must be present in every checkpoint (field contents optional for
# backward compatibility with old checkpoints).
# =============================================================================
REQUIRED_CHECKPOINT_KEYS = {
    "params_dict",
    "input_file_path",
    "input_file_content",
    "is_fourier"
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


def _get_restored_filepath(original_path, include_dir=True, restored_text="restored"):
    """Append '_restored' to the filename (before the extension)."""
    if not original_path:
        return None
    path = str(original_path)
    dir_name = os.path.dirname(path)
    base_name = os.path.basename(path)
    name, ext = os.path.splitext(base_name)
    restored_name = f"{name}_{restored_text}{ext}"
    return os.path.join(dir_name, restored_name) if dir_name and include_dir else restored_name


def init_checkpoint(params):
    """Initialize the checkpoint file with the invariant data only.
    This should be called once at the start (before any Fourier directions run).
    Creates the .npz with params_dict, input_file_path, input_file_content,
    is_fourier, and all optional keys pre-initialized to None.
    """
    if not getattr(params, "checkpoint_filepath", None):
        raise ValueError("params.checkpoint_filepath is not set - cannot write checkpoint")
    
    checkpoint_path = params.checkpoint_filepath

    # Load existing checkpoint
    if os.path.exists(checkpoint_path):
        try:
            _ = np.load(checkpoint_path, allow_pickle=True)
        except Exception as e:
            raise RuntimeError(f"Error occurred while checking existing checkpoint: {e}")
        finally:
            params.checkpoint_filepath = _get_restored_filepath(params.checkpoint_filepath, restored_text="new")
            logger.warning(f"Previous checkpoint file exists, moving new one to {params.checkpoint_filepath}")
            checkpoint_path = params.checkpoint_filepath

    input_file_path = params.input_file_path
    with open(input_file_path, "rb") as f:
        input_file_content = f.read()

    save_dict = {
        "params_dict":        dict(params.__dict__),
        "input_file_path":    input_file_path,
        "input_file_content": input_file_content,
        "is_fourier":         getattr(params, 'has_fourier', False),
    }

    for dir in params.xyz:
        suffix = f"_{dir}" if save_dict["is_fourier"] else ""
        save_dict[f"checkpoint_time{suffix}"] = 0.0

    # Pre-populate every optional key with None so the full structure exists
    # and later direction-specific updates never overwrite unrelated keys
    for key in OPTIONAL_KEYS:
        if key not in save_dict:
            save_dict[key] = None

    # === Locked + atomic write (prevents race condition) ===
    with _checkpoint_lock(checkpoint_path):
        np.savez(checkpoint_path, allow_pickle=True, **save_dict)

    logger.debug(f"Initialized checkpoint to {checkpoint_path} "
                 f"with keys: {list(save_dict.keys())}")


def add_field_e_checkpoint(params, field_e_filepath):
    """
    Add the electric field content to the checkpoint.
    """
    if not getattr(params, "checkpoint_filepath", None):
        raise ValueError("params.checkpoint_filepath is not set - cannot write checkpoint")

    checkpoint_path = params.checkpoint_filepath

    # === Locked read-modify-write ===
    with _checkpoint_lock(checkpoint_path):
        if os.path.exists(checkpoint_path):
            loaded = np.load(checkpoint_path, allow_pickle=True)
            save_dict = {key: loaded[key] for key in loaded.files}
            loaded.close()
        else:
            raise FileNotFoundError("Checkpoint file not found during update")
        
        is_fourier = save_dict["is_fourier"]
        dir_component = getattr(params, 'molecule_source_dict', {}).get('component') if is_fourier else None
        suffix = f"_{dir_component}" if is_fourier and dir_component else ""

        with open(field_e_filepath, "rb") as f:
            save_dict[f"field_e{suffix}_content"] = f.read()

        np.savez(checkpoint_path, allow_pickle=True, **save_dict)


def update_checkpoint(params, molecule, checkpoint_time):
    """
    Update only the dynamic/simulation-state values for the current direction.
    Loads the existing checkpoint first, merges the new data, then writes it back.
    In Fourier mode this guarantees that data for the other two directions is preserved.
    """
    if not getattr(params, "checkpoint_filepath", None):
        raise ValueError("params.checkpoint_filepath is not set - cannot write checkpoint")

    checkpoint_path = params.checkpoint_filepath

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
        
        method = getattr(params, "propagator", getattr(params, "molecule_propagator_str", "")).lower()
        is_fourier = getattr(params, 'has_fourier', False)

        # Determine current direction (x/y/z) only in Fourier mode
        dir_component = getattr(params, 'molecule_source_dict', {}).get('component') if is_fourier else None

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

        # Enforce core content
        missing = REQUIRED_CHECKPOINT_KEYS - set(save_dict.keys())
        if missing:
            raise RuntimeError(f"BUG: checkpoint is missing required keys: {missing}")

        np.savez(checkpoint_path, allow_pickle=True, **save_dict)

    time_log_str = f"{'='*40} Updated checkpoint file {checkpoint_path} at time = {checkpoint_time}"
    time_log_str += f"(direction: {dir_component}) " if is_fourier else ""
    time_log_str += f"{'='*40}"
    logger.info(time_log_str)


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
    restored_input_filepath = _get_restored_filepath(data["input_file_path"], False)
    content = np.ndarray.item(data["input_file_content"])
    with open(restored_input_filepath, "w", encoding="utf-8") as f:
        f.write(content.decode("utf-8"))
    logger.debug(f"Input file reconstructed from checkpoint: {restored_input_filepath}")

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

    # === Restore CSV files ===
    if is_fourier:
        # Fixed: do NOT modify list while iterating
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
                restored_filepath = _get_restored_filepath(original_filepath)
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

    params.checkpoint_dict = {}
    method = getattr(params, "propagator", getattr(params, "molecule_propagator_str", "")).lower()
    if is_fourier:
        for dir in params.xyz:
            params.checkpoint_dict[f"checkpoint_time_{dir}"] = float(data[f"checkpoint_time_{dir}"])
            if dir in checkpoint_dirs:
                params.checkpoint_dict[f"D_ao_0_{dir}"] = data[f"D_ao_0_{dir}"]
                params.checkpoint_dict[f"mo_coeff_{dir}"] = data[f"mo_coeff_{dir}"]
                if method == "step":
                    params.checkpoint_dict[f"C_orth_ndt_{dir}"] = data[f"C_orth_ndt_{dir}"]
                elif method == "magnus2":
                    params.checkpoint_dict[f"F_orth_n12dt_{dir}"] = data[f"F_orth_n12dt_{dir}"]
    else:
        params.checkpoint_dict["checkpoint_time"] = float(data["checkpoint_time"])
        params.checkpoint_dict["D_ao_0"] = data["D_ao_0"]
        params.checkpoint_dict["mo_coeff"] = data["mo_coeff"]
        if method == "step":
            params.checkpoint_dict["C_orth_ndt"] = data["C_orth_ndt"]
        elif method == "magnus2":
            params.checkpoint_dict["F_orth_n12dt"] = data["F_orth_n12dt"]

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
                    f"t_x={params.checkpoint_dict['checkpoint_time_x']} au, "
                    f"t_y={params.checkpoint_dict['checkpoint_time_y']} au, "
                    f"t_z={params.checkpoint_dict['checkpoint_time_z']} au.")
    else:
        logger.info(f"Loaded checkpoint with data at t={params.checkpoint_dict['checkpoint_time']} au.")
    return params
