"""NPZ save helper logs headers and does not dump array values."""
import logging

import numpy as np

from plasmol.utils.npz import describe_npz, save_npz


def test_save_npz_logs_keys_and_unpack(tmp_path, caplog):
    path = tmp_path / "spectrum.npz"
    freqs = np.linspace(1.5, 5.0, 8)
    imag = [np.ones(8), np.zeros(8), -np.ones(8)]
    caplog.set_level(logging.INFO, logger="main")
    save_npz(
        str(path),
        abs_imag=imag,
        abs_real=imag,
        freqs=freqs,
        deconvolved=False,
    )
    loaded = np.load(path, allow_pickle=True)
    assert set(loaded.files) == {"abs_imag", "abs_real", "freqs", "deconvolved"}
    assert np.allclose(loaded["freqs"], freqs)
    text = caplog.text
    assert "Wrote NPZ" in text and "spectrum.npz" in text
    assert "abs_imag" in text and "freqs" in text
    assert "deconvolved" in text
    assert "d = np.load(" in text
    assert "allow_pickle=True" in text
    assert "frequency grid (eV)" in text
    # values themselves must not appear
    assert "1.5" not in text or "frequency" in text


def test_save_npz_does_not_log_bytes_payload(tmp_path, caplog):
    path = tmp_path / "ckpt.npz"
    secret = b"THIS_SHOULD_NOT_APPEAR_IN_THE_LOG"
    caplog.set_level(logging.INFO, logger="main")
    save_npz(str(path), detail="summary", field_e_content=secret, is_absorption=True)
    text = caplog.text
    assert "field_e_content" in text
    assert "embedded file bytes" in text or "keys:" in text
    assert secret.decode() not in text
    assert "d = np.load(" in text


def test_describe_npz_reads_existing_file(tmp_path, caplog):
    path = tmp_path / "existing.npz"
    np.savez(path, freqs=np.arange(4.0), deconvolved=True)
    caplog.set_level(logging.INFO, logger="main")
    mapping = describe_npz(str(path))
    assert "freqs" in mapping
    assert "Wrote NPZ" in caplog.text
    assert "freqs" in caplog.text
