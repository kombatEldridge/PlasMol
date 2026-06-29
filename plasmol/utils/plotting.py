# utils/plotting.py
import logging
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


def _plot_fields_combined(fields, output_image_path=None):
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


def plot_fields(fields, output_image_path=None, component_layout=None):
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
        _plot_fields_combined(fields, output_image_path)