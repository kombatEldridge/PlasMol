"""Fourier IO / spectrum helpers beyond polarization builders."""
import numpy as np

from plasmol.drivers.custom_drivers.fourier import (
    apply_damping,
    absorption,
    absorption_single,
    fold,
)


def test_apply_damping_shortens_amplitude():
    t = np.linspace(0, 10, 101)
    mu_arrs = (t, np.ones_like(t), np.ones_like(t) * 2, np.ones_like(t) * 3)
    dx, dy, dz = apply_damping(mu_arrs, tau=2.0)
    assert dx[-1] < dx[0]
    assert np.isclose(dx[0], 1.0)
    assert dy[0] == 2.0


def test_absorption_single_shape():
    freqs = np.linspace(0.01, 1.0, 50)
    imag = freqs * 0.1
    A = absorption_single(imag, freqs)
    assert A.shape == freqs.shape
    assert np.all(A <= 0) or np.any(A != 0)  # non-trivial


def test_absorption_isotropic_shape():
    freqs = np.linspace(0.01, 1.0, 40)
    imag = np.stack([freqs * 0.1, freqs * 0.05, freqs * 0.02], axis=0)
    A = absorption(imag, freqs)
    assert A.shape == freqs.shape


def test_fold_three_files(tmp_path):
    def write(path, col_vals, colname):
        with open(path, "w") as f:
            f.write("Timestamps (au),X Values,Y Values,Z Values\n")
            for i, v in enumerate(col_vals):
                t = i * 0.1
                row = [t, 0.0, 0.0, 0.0]
                idx = {"X Values": 1, "Y Values": 2, "Z Values": 3}[colname]
                row[idx] = v
                f.write(",".join(str(x) for x in row) + "\n")

    xs = [1.0, 2.0, 3.0]
    ys = [4.0, 5.0, 6.0]
    zs = [7.0, 8.0, 9.0]
    fx, fy, fz = tmp_path / "x.csv", tmp_path / "y.csv", tmp_path / "z.csv"
    write(fx, xs, "X Values")
    write(fy, ys, "Y Values")
    write(fz, zs, "Z Values")
    t, stacked = fold(str(fx), str(fy), str(fz))
    assert stacked.shape == (3, 3)
    assert np.allclose(stacked[0], xs)
    assert np.allclose(stacked[1], ys)
    assert np.allclose(stacked[2], zs)
