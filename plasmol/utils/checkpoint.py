# utils/checkpoint.py
import numpy as np
import logging
import _pickle
import os
from plasmol.utils.csv import read_field_csv

logger = logging.getLogger("main")


# =============================================================================
# Exactly what must be present in every checkpoint (field contents optional for
# backward compatibility with old checkpoints).
# =============================================================================
REQUIRED_CHECKPOINT_KEYS = {
    "checkpoint_time",
    "D_ao_0",
    "mo_coeff",
    "params_dict",          # ← changed from "params" to a plain dict
}

OPTIONAL_KEYS = {"C_orth_ndt", "F_orth_n12dt", "field_e_content", "field_p_content"}


def _get_restored_filepath(original_path):
    """Append '_restored' to the filename (before the extension)."""
    if not original_path:
        return None
    path = str(original_path)
    dir_name = os.path.dirname(path)
    base_name = os.path.basename(path)
    name, ext = os.path.splitext(base_name)
    restored_name = f"{name}_restored{ext}"
    return os.path.join(dir_name, restored_name) if dir_name else restored_name


def update_checkpoint(params, molecule, checkpoint_time):
    """
    Save a checkpoint file containing the current state of the simulation.
    """
    if not getattr(params, "checkpoint_filepath", None):
        raise ValueError("params.checkpoint_filepath is not set - cannot write checkpoint")

    method = getattr(params, "propagator", getattr(params, "molecule_propagator_str", "")).lower()

    save_dict = {
        "checkpoint_time": checkpoint_time,
        "D_ao_0":          molecule.D_ao_0,
        "mo_coeff":        molecule.mf.mo_coeff,
        "params_dict":     dict(params.__dict__),   # ← plain dict, very reliable
        "field_e_content": None,
        "field_p_content": None,
    }

    # propagator-specific data
    if method == "step":
        save_dict["C_orth_ndt"] = molecule.C_orth_ndt
    elif method == "magnus2":
        save_dict["F_orth_n12dt"] = molecule.F_orth_n12dt

    # === Embed the two CSV files as exact raw bytes ===
    for name, filepath_attr in [("field_e", "field_e_filepath"),
                                ("field_p", "field_p_filepath")]:
        filepath = getattr(params, filepath_attr, None)
        content_key = f"{name}_content"
        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, "rb") as f:
                    save_dict[content_key] = f.read()
            except Exception as e:
                logger.error(f"Failed to read {filepath} for checkpoint: {e}")
                raise RuntimeError(f"Cannot create checkpoint - unable to read {name} file") from e
        else:
            logger.warning(f"{name} file '{filepath}' not found - saving None content")

    # Enforce core content
    missing = REQUIRED_CHECKPOINT_KEYS - set(save_dict.keys())
    if missing:
        raise RuntimeError(f"BUG: checkpoint is missing required keys: {missing}")

    np.savez(params.checkpoint_filepath, allow_pickle=True, **save_dict)
    logger.debug(f"Wrote checkpoint to {params.checkpoint_filepath} "
                 f"with keys: {list(save_dict.keys())}")


def resume_from_checkpoint(params):
    """
    Load checkpoint and mutate the passed params object in place.
    """
    fn = getattr(params, "checkpoint_filepath", None)
    if not fn:
        raise ValueError("params.checkpoint_filepath is not set - cannot resume")

    try:
        data = np.load(fn, allow_pickle=True)
    except FileNotFoundError:
        logger.error(f"Checkpoint file {fn} not found.")
        raise
    except (_pickle.UnpicklingError, ValueError, OSError, KeyError) as e:
        logger.error(f"{fn} is not a valid checkpoint archive: {e}")
        raise

    logger.debug(f"Loading checkpoint from {fn}")

    # === Validate core content ===
    loaded_keys = set(data.keys())
    missing = REQUIRED_CHECKPOINT_KEYS - loaded_keys
    extra = loaded_keys - REQUIRED_CHECKPOINT_KEYS - OPTIONAL_KEYS
    if missing:
        raise RuntimeError(f"Checkpoint is missing required keys: {missing}")
    if extra:
        logger.warning(f"Checkpoint contains unexpected extra keys (ignored): {extra}")

    # === Update the caller's params object in place (this was the bug) ===
    # np.savez wraps pickled objects in a 0-d ndarray on load; .item() unwraps it.
    saved_params_dict = data["params_dict"].item()
    for key, value in saved_params_dict.items():
        if key in params.__dict__:
            try:
                changed = params.__dict__[key] != value
                # numpy arrays return an element-wise array from !=; collapse to a single bool
                if hasattr(changed, "__iter__"):
                    changed = np.any(changed)
                if changed and not key == "resume_from_checkpoint":
                    logger.warning(f"Parameter '{key}': {params.__dict__[key]} != {value} (overwriting)")
            except Exception:
                pass  # if comparison itself fails, just overwrite silently
        params.__dict__[key] = value

    logger.debug("Params object has been updated in place from checkpoint.")

    # common simulation state
    params.checkpoint_dict = {
        "checkpoint_time": float(data["checkpoint_time"]),
        "D_ao_0":          data["D_ao_0"],
        "mo_coeff":        data["mo_coeff"],
    }

    # propagator-specific state
    method = getattr(params, "propagator", getattr(params, "molecule_propagator_str", "")).lower()
    if method == "step" and "C_orth_ndt" in data:
        params.checkpoint_dict["C_orth_ndt"] = data["C_orth_ndt"]
    elif method == "magnus2" and "F_orth_n12dt" in data:
        params.checkpoint_dict["F_orth_n12dt"] = data["F_orth_n12dt"]

    # === Restore CSV files (only if present) ===
    restored_any = False
    for name in ["field_e", "field_p"]:
        content_key = f"{name}_content"
        if content_key in data and data[content_key] is not None:
            original_filepath = getattr(params, f"{name}_filepath", None)
            if original_filepath:
                restored_filepath = _get_restored_filepath(original_filepath)
                try:
                    os.makedirs(os.path.dirname(restored_filepath) or ".", exist_ok=True)
                    with open(restored_filepath, "wb") as f:
                        f.write(data[content_key])
                    setattr(params, f"{name}_filepath", restored_filepath)
                    logger.info(f"Restored {name} CSV from checkpoint → {restored_filepath}")
                    restored_any = True
                except Exception as e:
                    logger.error(f"Failed to write restored {name} CSV: {e}")
                    raise
        else:
            logger.debug(f"No {name}_content in checkpoint (old checkpoint)")

    if not restored_any:
        logger.warning("This checkpoint does not contain embedded field CSV files (old-style checkpoint).")

    # === Final validation ===
    try:
        _ = read_field_csv(params.field_e_filepath)
        _ = read_field_csv(params.field_p_filepath)
        logger.debug("Successfully validated both field CSV files")
    except Exception as e:
        logger.error(f"Field CSV validation failed: {e}")
        raise RuntimeError("Checkpoint restore succeeded but field CSVs cannot be read") from e
    
    params.resume_from_checkpoint = True
    logger.info(f"Loaded checkpoint with data at t={params.checkpoint_dict['checkpoint_time']} au.")


def load_checkpoint_data(checkpoint_path):
    """
    Load data from checkpoint file and return the saved params dict and checkpoint state.
    This is used by PARAMS.__init__ to create a full params object without JSON parsing.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found.")

    try:
        data = np.load(checkpoint_path, allow_pickle=True)
    except FileNotFoundError:
        logger.error(f"Checkpoint file {checkpoint_path} not found.")
        raise
    except (_pickle.UnpicklingError, ValueError, OSError, KeyError) as e:
        logger.error(f"{checkpoint_path} is not a valid checkpoint archive: {e}")
        raise

    logger.debug(f"Loading checkpoint from {checkpoint_path}")
    logger.debug(f"Checkpoint contains {len(data.keys())} keys: {sorted(data.keys())}")

    # === Validate core content ===
    loaded_keys = set(data.keys())
    missing = REQUIRED_CHECKPOINT_KEYS - loaded_keys
    extra = loaded_keys - REQUIRED_CHECKPOINT_KEYS - OPTIONAL_KEYS
    if missing:
        raise RuntimeError(f"Checkpoint is missing required keys: {missing}")
    if extra:
        logger.warning(f"Checkpoint contains unexpected extra keys (ignored): {extra}")

    # np.savez wraps pickled objects in a 0-d ndarray on load; .item() unwraps it.
    saved_params_dict = data["params_dict"].item()
    logger.debug(f"Loaded {len(saved_params_dict)} parameters from checkpoint")
    logger.debug(f"Critical params present - dt: {saved_params_dict.get('dt')}, t_end: {saved_params_dict.get('t_end')}, "
                 f"checkpoint_filepath: {saved_params_dict.get('checkpoint_filepath')}, "
                 f"molecule_propagator_str: {saved_params_dict.get('molecule_propagator_str')}")

    # common simulation state
    checkpoint_dict = {
        "checkpoint_time": float(data["checkpoint_time"]),
        "D_ao_0":          data["D_ao_0"],
        "mo_coeff":        data["mo_coeff"],
    }

    # propagator-specific state
    method = saved_params_dict.get("propagator",
                                  saved_params_dict.get("molecule_propagator_str", "")).lower()
    if method == "step" and "C_orth_ndt" in data:
        checkpoint_dict["C_orth_ndt"] = data["C_orth_ndt"]
    elif method == "magnus2" and "F_orth_n12dt" in data:
        checkpoint_dict["F_orth_n12dt"] = data["F_orth_n12dt"]

    # === Restore CSV files (only if present) ===
    restored_any = False
    for name in ["field_e", "field_p"]:
        content_key = f"{name}_content"
        if content_key in data and data[content_key] is not None:
            original_filepath = saved_params_dict.get(f"{name}_filepath")
            if original_filepath:
                restored_filepath = _get_restored_filepath(original_filepath)
                try:
                    os.makedirs(os.path.dirname(restored_filepath) or ".", exist_ok=True)
                    with open(restored_filepath, "wb") as f:
                        f.write(data[content_key])
                    # Update the path in the saved dict
                    saved_params_dict[f"{name}_filepath"] = restored_filepath
                    logger.info(f"Restored {name} CSV from checkpoint → {restored_filepath}")
                    restored_any = True
                except Exception as e:
                    logger.error(f"Failed to write restored {name} CSV: {e}")
                    raise
        else:
            logger.debug(f"No {name}_content in checkpoint (old checkpoint)")

    if not restored_any:
        logger.warning("This checkpoint does not contain embedded field CSV files (old-style checkpoint).")

    # === Final validation ===
    try:
        _ = read_field_csv(saved_params_dict.get("field_e_filepath"))
        _ = read_field_csv(saved_params_dict.get("field_p_filepath"))
        logger.debug("Successfully validated both field CSV files")
    except Exception as e:
        logger.error(f"Field CSV validation failed: {e}")
        raise RuntimeError("Checkpoint restore succeeded but field CSVs cannot be read") from e

    logger.info(f"Loaded checkpoint with data at t={checkpoint_dict['checkpoint_time']} au.")

    return saved_params_dict, checkpoint_dict, data
