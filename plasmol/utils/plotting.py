# utils/plotting.py
import logging
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger("main")

def plot_fields(fields, output_image_path=None):
    """
    Plot one or more vector fields (X, Y, Z components) from CSV files.

    Each field gets its own subplot with a custom title.

    Parameters:
    fields : list of tuples (filepath: str, title: str)
        List of (CSV filepath, subplot title) pairs.
        Example:
        [
            (field_e_filepath, 'Incident Electric Field'),
            (field_p_filepath, "Molecule's Response")
        ]
    output_image_path : str, optional
        Base filename for the saved plot (will save as {output_image_path}.png).

    Returns:
    None
    """
    if not fields:
        raise ValueError("At least one field must be provided.")

    logging.getLogger('matplotlib').setLevel(logging.INFO)

    # Load and prepare all fields (sorting is now done in memory – no more mutating original CSVs)
    data_list = []
    timestamp_col = None

    for filepath, title in fields:
        logging.debug(f"Reading CSV file: {filepath}")

        df = pd.read_csv(filepath, comment='#')

        # Find the timestamp column
        ts_cols = [col for col in df.columns if col.startswith("Timestamps")]
        if not ts_cols:
            raise ValueError(f"No timestamp column found in {filepath}")
        ts_col = ts_cols[0]

        if timestamp_col is None:
            timestamp_col = ts_col
        elif ts_col != timestamp_col:
            logging.warning(f"Timestamp column mismatch in {filepath} (using first one)")

        # Sort by timestamp (in memory, non-destructive)
        df = df.sort_values(by=ts_col, ascending=True)

        data_list.append({
            'df': df,
            'title': title,
            'timestamp_col': ts_col
        })

    # Create subplots (works for 1, 2, 3+ fields automatically)
    n = len(data_list)
    fig, axs = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    axs = axs.flatten()

    for ax, item in zip(axs, data_list):
        df = item['df']
        ts_col = item['timestamp_col']
        title = item['title']

        timestamps = df[ts_col]
        x_values = df['X Values']
        y_values = df['Y Values']
        z_values = df['Z Values']

        ax.plot(timestamps, x_values, label='x', marker='o')
        ax.plot(timestamps, y_values, label='y', marker='o')
        ax.plot(timestamps, z_values, label='z', marker='o')

        ax.set_title(title)
        ax.set_xlabel(ts_col)
        ax.set_ylabel('Field Magnitude')
        ax.legend()

    plt.tight_layout()

    if output_image_path:
        save_path = f'{output_image_path}.png'
        plt.savefig(save_path, dpi=1000)
        logging.info(f"Matplotlib image written: {save_path}")
    else:
        plt.show()  # fallback if no path is given