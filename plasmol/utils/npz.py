# NPZ write helper: persist arrays and log a header so users can unpack the file.
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger("main")

# Known archive keys → one-line meaning. Unknown keys still get dtype/shape.
_KEY_HELP = {
    "abs_imag": "Im[μ] or Im[μ/E] per Cartesian axis (object[3] of 1-D arrays)",
    "abs_real": "Re[μ] or Re[μ/E] per Cartesian axis (object[3] of 1-D arrays)",
    "freqs": "frequency grid (eV)",
    "deconvolved": "True if hybrid μ/E, False if kick Im[μ]",
    "params_dict": "pickled PARAMS.__dict__ at save time",
    "input_file_path": "path of the input JSON used for this run",
    "input_file_content": "raw input JSON bytes",
    "is_absorption": "True if directional absorption checkpoint",
    "is_fourier": "legacy name for is_absorption",
    "is_open_shell": "True if UKS (spin ≠ 0)",
    "updated_after_init": "True after the first propagator state write",
    "xyz_file_path": "geometry filename, if any",
    "xyz_content": "raw xyz / geometry bytes",
    "checkpoint_time": "simulation time of this snapshot (a.u.)",
    "D_ao_0": "AO density at t=0 (or current snapshot)",
    "mo_coeff": "MO coefficients",
    "C_orth_ndt": "step-propagator orthonormal C",
    "F_orth_n12dt": "magnus2 Fock at n+1/2",
    "field_e_content": "embedded field_e.csv bytes",
    "field_p_content": "embedded field_p.csv bytes",
    "core_hole_mo_occ_content": "embedded core-hole MO occupation CSV bytes",
}


def _help_for(name):
    if name in _KEY_HELP:
        return _KEY_HELP[name]
    if name.endswith("_content"):
        return "embedded file bytes"
    for stem in (
        "checkpoint_time",
        "D_ao_0",
        "mo_coeff",
        "C_orth_ndt",
        "F_orth_n12dt",
        "field_e",
        "field_p",
    ):
        if name.startswith(stem + "_") or name.startswith(stem):
            return _KEY_HELP.get(stem, "")
    return ""


def _nbytes_str(n):
    try:
        n = int(n)
    except (TypeError, ValueError):
        return ""
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f" {n / 1024:.1f} KiB"
    return f" {n / 1024 ** 2:.1f} MiB"


def _type_shape(value):
    """Return (kind, shape_label) without dumping values."""
    if value is None:
        return "None", ""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "bytes", _nbytes_str(len(value)).strip()
    if isinstance(value, np.ndarray):
        if value.dtype == object or getattr(value.dtype, "hasobject", False):
            if value.shape == ():
                return _type_shape(value.item())
            inner = []
            for item in value.ravel()[:3]:
                kind, shape = _type_shape(item)
                inner.append(f"{kind} {shape}".strip())
            extra = ", ..." if value.size > 3 else ""
            return "object", f"{value.shape} [{', '.join(inner)}{extra}]"
        if value.shape == ():
            return str(value.dtype), "()"
        return str(value.dtype), f"{value.shape}{_nbytes_str(value.nbytes)}"
    if isinstance(value, (list, tuple)):
        inner = []
        for item in list(value)[:3]:
            kind, shape = _type_shape(item)
            inner.append(f"{kind} {shape}".strip())
        extra = ", ..." if len(value) > 3 else ""
        return type(value).__name__, f"({len(value)},) [{', '.join(inner)}{extra}]"
    if isinstance(value, dict):
        return "dict", f"{len(value)} keys"
    if isinstance(value, (bool, np.bool_)):
        return "bool", str(bool(value))
    if isinstance(value, str):
        return "str", f"{len(value)} chars"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return type(value).__name__, ""
    return type(value).__name__, ""


def _needs_pickle(mapping):
    for value in mapping.values():
        if isinstance(value, (bytes, bytearray, memoryview, dict, list, tuple)):
            return True
        if isinstance(value, np.ndarray) and (
            value.dtype == object or getattr(value.dtype, "hasobject", False)
        ):
            return True
        if not isinstance(value, (np.ndarray, np.generic, bool, int, float, str, type(None))):
            return True
    return False


def _unpack_lines(path, names, needs_pickle):
    pickle_arg = ", allow_pickle=True" if needs_pickle else ""
    lines = [f"d = np.load({path!r}{pickle_arg})"]
    if 0 < len(names) <= 6:
        lhs = ", ".join(names)
        rhs = ", ".join(f"d[{n!r}]" for n in names)
        lines.append(f"{lhs} = {rhs}")
    elif names:
        lines.append("# keys listed above; d['<key>'] to unpack one array")
    return lines


def _checkpoint_time_label(mapping):
    """Human-readable snapshot time from a checkpoint mapping, or None."""
    if "checkpoint_time" in mapping and mapping["checkpoint_time"] is not None:
        try:
            t = float(np.asarray(mapping["checkpoint_time"]).reshape(-1)[0])
            return f"t={t:g} au"
        except (TypeError, ValueError):
            pass
    parts = []
    for axis in ("x", "y", "z"):
        key = f"checkpoint_time_{axis}"
        if key not in mapping or mapping[key] is None:
            continue
        try:
            t = float(np.asarray(mapping[key]).reshape(-1)[0])
        except (TypeError, ValueError):
            continue
        parts.append(f"t_{axis}={t:g} au")
    return ", ".join(parts) if parts else None


def describe_mapping(mapping, path, *, detail="full", log=None, existed=None):
    """Log a header for an in-memory mapping that was (or will be) written to path."""
    log = log or logger
    items = list(mapping.items())
    n = len(items)

    if detail == "quiet":
        verb = "updated" if existed else "written"
        when = _checkpoint_time_label(mapping)
        if when:
            log.debug(f"Checkpoint {verb}: {when} -> {path}")
        else:
            log.debug(f"Checkpoint {verb}: {path}")
        if log.isEnabledFor(logging.DEBUG):
            _log_key_table(items, log.debug)
        return

    log.info(f"Wrote NPZ '{path}' ({n} array{'' if n == 1 else 's'})")

    names = [k for k, _ in items]
    if detail == "summary":
        preview = ", ".join(names[:12])
        if len(names) > 12:
            preview += ", ..."
        log.info(f"  keys: {preview}")
        for line in _unpack_lines(str(path), names, _needs_pickle(mapping)):
            log.info(f"  {line}")
        if log.isEnabledFor(logging.DEBUG):
            _log_key_table(items, log.debug)
        return

    _log_key_table(items, log.info)
    log.info("  Unpack:")
    for line in _unpack_lines(str(path), names, _needs_pickle(mapping)):
        log.info(f"    {line}")


def _log_key_table(items, emit):
    width = max((len(str(k)) for k, _ in items), default=4)
    width = min(max(width, 8), 28)
    for name, value in items:
        kind, shape = _type_shape(value)
        meaning = _help_for(name)
        tail = f"  {meaning}" if meaning else ""
        emit(f"  {name:<{width}}  {kind:<12} {shape}{tail}")


def describe_npz(source, *, detail="full", log=None):
    """
    Log the contents of an NPZ file or a key→array mapping.

    ``source`` may be a filesystem path or a dict-like mapping.
    """
    if isinstance(source, (str, Path)):
        path = str(source)
        loaded = np.load(path, allow_pickle=True)
        try:
            mapping = {key: loaded[key] for key in loaded.files}
        finally:
            loaded.close()
    else:
        path = str(getattr(source, "filename", "<memory>"))
        mapping = dict(source)
    describe_mapping(mapping, path, detail=detail, log=log)
    return mapping


def save_npz(path, *, detail="full", **arrays):
    """
    Write ``path`` with ``np.savez`` and log a header (keys, dtypes, unpack).

    ``detail='full'`` (default) prints one line per key plus an unpack snippet
    — use for user-facing archives such as absorption spectra.

    ``detail='summary'`` prints the key list and unpack line at INFO (full
    table at DEBUG).

    ``detail='quiet'`` logs snapshot time and path at DEBUG
    (``written`` on first create, ``updated`` if the file already existed).
    """
    payload = dict(arrays)
    payload.pop("allow_pickle", None)
    existed = Path(path).exists()
    np.savez(path, **payload)
    describe_mapping(payload, path, detail=detail, existed=existed)
    return path


def main(argv=None):
    """``python -m plasmol.utils.npz file.npz [file2.npz ...]``"""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m plasmol.utils.npz FILE.npz [...]", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    status = 0
    for arg in argv:
        try:
            describe_npz(arg, detail="full")
        except Exception as exc:
            logger.error(f"{arg}: {exc}")
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
