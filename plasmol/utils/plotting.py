# utils/plotting.py
import logging
import re
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger("main")

_COMPONENTS = ('x', 'y', 'z')
_COMPONENT_COLS = {
    'x': 'X Values',
    'y': 'Y Values',
    'z': 'Z Values',
}
_COMPONENT_COLORS = {
    'x': '#1f77b4',
    'y': '#ff7f0e',
    'z': '#2ca02c',
}


def _load_field_csv(filepath, title):
    logging.debug(f"Reading CSV file: {filepath}")
    df = pd.read_csv(filepath, comment='#')
    ts_cols = [col for col in df.columns if col.startswith("Timestamps")]
    if not ts_cols:
        raise ValueError(f"No timestamp column found in {filepath}")
    ts_col = ts_cols[0]
    df = df.sort_values(by=ts_col, ascending=True)
    return {
        'df': df,
        'title': title,
        'timestamp_col': ts_col,
    }


def _save_or_show(fig, output_image_path):
    if output_image_path:
        save_path = f'{output_image_path}.png'
        fig.savefig(save_path, dpi=1000)
        logging.info(f"Matplotlib image written: {save_path}")
    else:
        plt.show()


def _plot_e_vs_p_by_component(fields, output_image_path=None):
    """Plot electric and polarization fields with one subplot per component."""
    e_field, p_field = (_load_field_csv(filepath, title) for filepath, title in fields)

    fig, axs = plt.subplots(3, 2, figsize=(12, 5), sharex='col', squeeze=False)

    for row, component in enumerate(_COMPONENTS):
        col_name = _COMPONENT_COLS[component]

        color = _COMPONENT_COLORS[component]

        ax_e = axs[row, 0]
        ax_e.plot(
            e_field['df'][e_field['timestamp_col']],
            e_field['df'][col_name],
            color=color,
        )
        ax_e.set_ylabel('Field Magnitude')
        if row == 0:
            ax_e.set_title(e_field['title'])
        if row == 2:
            ax_e.set_xlabel(e_field['timestamp_col'])

        ax_p = axs[row, 1]
        ax_p.plot(
            p_field['df'][p_field['timestamp_col']],
            p_field['df'][col_name],
            color=color,
        )
        ax_p.set_ylabel('Field Magnitude')
        if row == 0:
            ax_p.set_title(p_field['title'])
        if row == 2:
            ax_p.set_xlabel(p_field['timestamp_col'])

        ax_e.text(
            0.02, 0.95, f'{component}',
            transform=ax_e.transAxes,
            fontsize=10,
            fontweight='bold',
            color=color,
            va='top',
        )
        ax_p.text(
            0.02, 0.95, f'{component}',
            transform=ax_p.transAxes,
            fontsize=10,
            fontweight='bold',
            color=color,
            va='top',
        )

    plt.tight_layout()
    _save_or_show(fig, output_image_path)
    plt.close(fig)


def _plot_e_p_fields_combined(fields, output_image_path=None):
    """Plot each field with x, y, and z components on a single subplot."""
    data_list = []
    timestamp_col = None

    for filepath, title in fields:
        item = _load_field_csv(filepath, title)
        ts_col = item['timestamp_col']
        if timestamp_col is None:
            timestamp_col = ts_col
        elif ts_col != timestamp_col:
            logging.warning(f"Timestamp column mismatch in {filepath} (using first one)")
        data_list.append(item)

    n = len(data_list)
    fig, axs = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    axs = axs.flatten()

    for ax, item in zip(axs, data_list):
        df = item['df']
        ts_col = item['timestamp_col']
        title = item['title']

        for component in _COMPONENTS:
            ax.plot(
                df[ts_col],
                df[_COMPONENT_COLS[component]],
                label=component,
                color=_COMPONENT_COLORS[component],
                marker='o',
            )

        ax.set_title(title)
        ax.set_xlabel(ts_col)
        ax.set_ylabel('Field Magnitude')
        ax.legend()

    plt.tight_layout()
    _save_or_show(fig, output_image_path)
    plt.close(fig)


def plot_e_p_fields(fields, output_image_path=None, component_layout=None):
    """
    Plot one or more vector fields (X, Y, Z components) from CSV files.

    For the standard electric-vs-polarization pair, each directional component is
    shown in its own subplot: three E-field spectra on the left and three
    P-field spectra on the right, keeping the overall image size at 12x5 inches.

    Parameters:
    fields : list of tuples (filepath: str, title: str)
        List of (CSV filepath, subplot title) pairs.
    output_image_path : str, optional
        Base filename for the saved plot (will save as {output_image_path}.png).
    component_layout : bool, optional
        If True, split the two-field E-vs-P plot by component. Defaults to True
        when exactly two fields are provided.

    Returns:
    None
    """
    if not fields:
        raise ValueError("At least one field must be provided.")

    logging.getLogger('matplotlib').setLevel(logging.INFO)

    if component_layout is None:
        component_layout = len(fields) == 2

    if component_layout and len(fields) == 2:
        _plot_e_vs_p_by_component(fields, output_image_path)
    else:
        _plot_e_p_fields_combined(fields, output_image_path)


# Palette inspired by the multi-MO occupation traces (gold / teal / purple / …)
_MO_COLORS = (
    '#E0A800',  # gold
    '#2A9D8F',  # teal
    '#9B59B6',  # purple
    '#E76F51',  # coral
    '#264653',  # dark slate
    '#F4A261',  # sand
    '#1D3557',  # navy
    '#E63946',  # red
)


def plot_dch_mo_occupations(
    dch_mo_occ_filepath,
    output_image_path=None,
    indices=None,
    time_window=None,
    filter_by_amplitude=False,
    amplitude_threshold=0.6,
):
    """
    Plot time-dependent MO occupations from a DCH occupation CSV.

    Expects a file written by the DCH driver / ``molecule.get_mo_occupations``:
    comment lines starting with ``#``, a header row with a timestamp column and
    one column per watched MO (e.g. ``MO index 0``), then numeric data rows.

    Each MO series is overlaid on a single axes (same layout as typical
    core-hole occupation figures). Legend entries are the bare MO indices
    (e.g. ``0``, ``23``), laid out left-to-right and wrapping to the next
    line only when needed.

    Parameters:
    dch_mo_occ_filepath : str
        Path to the MO occupation CSV (``dch_mo_occ_filepath``).
    output_image_path : str, optional
        Base filename for the saved plot (saved as ``{output_image_path}.png``).
        If None, the figure is shown interactively.
    indices : list of int, optional
        0-based MO indices to include in the plot (matching the numbers in
        column headers such as ``MO index 23``). If None or empty, all MO
        columns in the file are plotted. Order of ``indices`` is preserved
        in the legend.
    time_window : tuple of float, optional
        ``(start, end)`` time window (same units as the CSV timestamp column).
        Data outside the window are dropped before amplitude filtering and plotting.
        If None, the full time range is used.
    filter_by_amplitude : bool, optional
        If True, keep only MOs whose peak-to-peak amplitude
        (``max - min`` over the plotted window) is strictly greater than
        ``amplitude_threshold``. Centering/mean of the series is ignored
        (matches Fig. 8 “> 0.2 e” style selection).
    amplitude_threshold : float, optional
        Amplitude cutoff used when ``filter_by_amplitude`` is True. Default 0.2.

    Returns:
    None
    """
    logging.getLogger('matplotlib').setLevel(logging.INFO)
    logger.debug(f"Reading DCH MO occupation CSV: {dch_mo_occ_filepath}")

    df = pd.read_csv(dch_mo_occ_filepath, comment='#')
    ts_cols = [col for col in df.columns if str(col).startswith("Timestamps")]
    if not ts_cols:
        raise ValueError(
            f"No timestamp column found in {dch_mo_occ_filepath}. "
            "Expected a column whose name starts with 'Timestamps'."
        )
    ts_col = ts_cols[0]
    mo_cols = [col for col in df.columns if col != ts_col]
    if not mo_cols:
        raise ValueError(
            f"No MO occupation columns found in {dch_mo_occ_filepath} "
            f"(only timestamp column '{ts_col}')."
        )

    if indices:
        # Map bare MO index (int) → CSV column name
        col_by_index = {}
        for col in mo_cols:
            try:
                col_by_index[int(_mo_index_label(col))] = col
            except (TypeError, ValueError):
                continue

        selected = []
        missing = []
        for idx in indices:
            try:
                key = int(idx)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Invalid MO index {idx!r}; expected an integer."
                ) from e
            if key in col_by_index:
                selected.append(col_by_index[key])
            else:
                missing.append(key)

        if missing:
            available = sorted(col_by_index.keys())
            raise ValueError(
                f"MO index(es) {missing} not found in {dch_mo_occ_filepath}. "
                f"Available indices: {available}."
            )
        mo_cols = selected
        logger.debug(f"Candidate MO indices: {list(indices)}")
    else:
        logger.debug(f"Candidate MO columns: all {len(mo_cols)}")

    df = df.sort_values(by=ts_col, ascending=True)
    if time_window is not None:
        t0, t1 = time_window
        df = df[(df[ts_col] >= t0) & (df[ts_col] <= t1)]
        if df.empty:
            raise ValueError(
                f"No data in time_window={time_window} for {dch_mo_occ_filepath}."
            )

    if filter_by_amplitude:
        kept = []
        for col in mo_cols:
            amp = float(df[col].max() - df[col].min())
            if amp > amplitude_threshold:
                kept.append(col)
                logger.debug(
                    f"Keeping MO {_mo_index_label(col)}: amplitude={amp:.4f} "
                    f"> {amplitude_threshold}"
                )
            else:
                logger.debug(
                    f"Dropping MO {_mo_index_label(col)}: amplitude={amp:.4f} "
                    f"<= {amplitude_threshold}"
                )
        if not kept:
            raise ValueError(
                f"No MO series with peak-to-peak amplitude > {amplitude_threshold} "
                f"in {dch_mo_occ_filepath}"
                + (f" within time_window={time_window}." if time_window else ".")
            )
        mo_cols = kept
        logger.info(
            f"Amplitude filter (>{amplitude_threshold}): plotting "
            f"{[_mo_index_label(c) for c in mo_cols]}"
        )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    handles = []
    labels = []
    for i, col in enumerate(mo_cols):
        color = _MO_COLORS[i % len(_MO_COLORS)]
        line, = ax.plot(
            df[ts_col],
            df[col],
            color=color,
            linewidth=1.2,
        )
        handles.append(line)
        labels.append(_mo_index_label(col))

    n_mo = len(labels)
    # Prefer one horizontal row; wrap after this many entries if many MOs are watched
    ncol = min(n_mo, 8)
    handles, labels = _legend_row_major(handles, labels, ncol)
    ax.set_xlabel(ts_col)
    ax.set_ylabel('hole occupation number')
    ax.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=ncol,
        frameon=True,
        fancybox=False,
        edgecolor='black',
        columnspacing=1.2,
        handlelength=1.5,
    )
    ax.axhline(0.0, color='0.7', linewidth=0.6, zorder=0)
    if len(df) > 0:
        ax.set_xlim(df[ts_col].iloc[0], df[ts_col].iloc[-1])

    plt.tight_layout()
    _save_or_show(fig, output_image_path)
    plt.close(fig)


def _mo_index_label(col):
    """Turn a CSV header like 'MO index 23' into the bare index string '23'."""
    match = re.search(r'(\d+)\s*$', str(col))
    if match:
        return match.group(1)
    return str(col).split()[-1]


def _legend_row_major(handles, labels, ncol):
    """
    Reorder (and pad) legend entries so matplotlib's column-major fill
    appears left-to-right, wrapping to the next line when ``ncol`` is full.
    """
    from matplotlib.lines import Line2D

    n = len(handles)
    if n == 0 or ncol <= 1:
        return handles, labels

    nrow = (n + ncol - 1) // ncol
    size = nrow * ncol
    # Invisible placeholders keep empty cells so a short last row stays left-aligned
    empty = Line2D([], [], linestyle='None', marker='None', label='')
    h_grid = [empty] * size
    l_grid = [''] * size
    for j in range(n):
        r, c = divmod(j, ncol)   # desired row-major position
        k = c * nrow + r         # matplotlib column-major slot
        h_grid[k] = handles[j]
        l_grid[k] = labels[j]
    return h_grid, l_grid