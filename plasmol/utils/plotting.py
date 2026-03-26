# utils/plotting.py
import logging
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger("main")

# TODO: generalize this to take a field file and a title for that field multiple times 
def plot_fields(field_e_filepath, field_p_filepath=None, output_image_path=None):
    """
    Plot electric field and optionally polarization field from CSV files.

    Generates a plot with one or two subplots depending on input, saving it to a file.

    Parameters:
    field_e_filepath : str
        Path to the electric field CSV file.
    field_p_filepath : str, optional
        Path to the polarization field CSV file (default None).
    output_image_path : str, optional
        Filename for the plot image (default None).

    Returns:
    None
    """
    logging.getLogger('matplotlib').setLevel(logging.INFO)

    if field_p_filepath is not None:
        logging.debug(f"Reading CSV files: {field_e_filepath} and {field_p_filepath}")
    else:
        logging.debug(f"Reading CSV file: {field_e_filepath}")

    def sort_csv_by_first_column(filename):
        """
        Sort a CSV file by its first column (timestamps).

        Preserves comments and header while sorting data rows.

        Parameters:
        filename : str
            Path to the CSV file to sort.

        Returns:
        None
        """
        with open(filename, 'r') as file:
            lines = file.readlines()
        comments = [line for line in lines if line.startswith('#')]
        header = next(line for line in lines if not line.startswith('#'))
        data_lines = [line for line in lines if not line.startswith('#') and line != header]
        from io import StringIO
        data = pd.read_csv(StringIO(''.join(data_lines)))
        timestamp_cols = [col for col in data.columns if col.startswith("Timestamps")]
        data_sorted = data.sort_values(by=timestamp_cols[0])
        with open(filename, 'w') as file:
            file.writelines(comments)
            file.write(header)
            data_sorted.to_csv(file, index=False)

    sort_csv_by_first_column(field_e_filepath)
    data1 = pd.read_csv(field_e_filepath, comment='#')
    timestamp_cols = [col for col in data1.columns if col.startswith("Timestamps")]
    data1 = data1.sort_values(by=timestamp_cols[0], ascending=True)
    timestamps1 = data1[timestamp_cols[0]]
    x_values1 = data1['X Values']
    y_values1 = data1['Y Values']
    z_values1 = data1['Z Values']

    if field_p_filepath is not None:
        sort_csv_by_first_column(field_p_filepath)
        data2 = pd.read_csv(field_p_filepath, comment='#')
        data2 = data2.sort_values(by=timestamp_cols[0], ascending=True)
        timestamps2 = data2[timestamp_cols[0]]
        x_values2 = data2['X Values']
        y_values2 = data2['Y Values']
        z_values2 = data2['Z Values']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.plot(timestamps1, x_values1, label='x', marker='o')
        ax1.plot(timestamps1, y_values1, label='y', marker='o')
        ax1.plot(timestamps1, z_values1, label='z', marker='o')
        ax1.set_title('Incident Electric Field')
        ax1.set_xlabel(timestamp_cols[0])
        ax1.set_ylabel('Electric Field Magnitude')
        ax1.legend()

        ax2.plot(timestamps2, x_values2, label='x', marker='o')
        ax2.plot(timestamps2, y_values2, label='y', marker='o')
        ax2.plot(timestamps2, z_values2, label='z', marker='o')
        ax2.set_title("Molecule's Response")
        ax2.set_xlabel(timestamp_cols[0])
        ax2.set_ylabel('Polarization Field Magnitude')
        ax2.legend()
    else:
        fig, ax1 = plt.subplots(figsize=(7, 5))
        ax1.plot(timestamps1, x_values1, label='x', marker='o')
        ax1.plot(timestamps1, y_values1, label='y', marker='o')
        ax1.plot(timestamps1, z_values1, label='z', marker='o')
        ax1.set_title('Incident Electric Field')
        ax1.set_xlabel(timestamp_cols[0])
        ax1.set_ylabel('Electric Field Magnitude')
        ax1.legend()

    plt.tight_layout()
    plt.savefig(f'{output_image_path}.png', dpi=1000)
    logging.info(f"Matplotlib image written: {output_image_path}.png")


# def show_field_e_2field_p(field_eFileName, field_pFileName1=None, field_pFileName2=None, matplotlibLocationIMG=None, output_image_path=None):
#     """
#     Plot electric field and optionally polarization fields from CSV files.

#     Generates a plot with one or two subplots depending on input, saving it to a file.
#     When two field_p files are provided, plots the x-component from the first and y-component from the second on a single subplot.

#     Parameters:
#     field_eFileName : str
#         Path to the electric field CSV file.
#     field_pFileName1 : str, optional
#         Path to the first polarization field CSV file (default None).
#     field_pFileName2 : str, optional
#         Path to the second polarization field CSV file (default None).
#     matplotlibLocationIMG : str, optional
#         Directory to save the plot image (default None).
#     output_image_path : str, optional
#         Filename for the plot image (default None).

#     Returns:
#     None
#     """
#     logging.getLogger('matplotlib').setLevel(logging.INFO)

#     if field_pFileName1 is not None and field_pFileName2 is not None:
#         logging.debug(f"Reading CSV files: {field_eFileName}, {field_pFileName1} and {field_pFileName2}")
#     elif field_pFileName1 is not None:
#         logging.debug(f"Reading CSV files: {field_eFileName} and {field_pFileName1}")
#     else:
#         logging.debug(f"Reading CSV file: {field_eFileName}")

#     def sort_csv_by_first_column(filename):
#         """
#         Sort a CSV file by its first column (timestamps).

#         Preserves comments and header while sorting data rows.

#         Parameters:
#         filename : str
#             Path to the CSV file to sort.

#         Returns:
#         None
#         """
#         with open(filename, 'r') as file:
#             lines = file.readlines()
#         comments = [line for line in lines if line.startswith('#')]
#         header = next(line for line in lines if not line.startswith('#'))
#         data_lines = [line for line in lines if not line.startswith('#') and line != header]
#         from io import StringIO
#         data = pd.read_csv(StringIO(''.join(data_lines)))
#         timestamp_cols = [col for col in data.columns if col.startswith("Timestamps")]
#         data_sorted = data.sort_values(by=timestamp_cols[0])
#         with open(filename, 'w') as file:
#             file.writelines(comments)
#             file.write(header)
#             data_sorted.to_csv(file, index=False)

#     sort_csv_by_first_column(field_eFileName)
#     data1 = pd.read_csv(field_eFileName, comment='#')
#     timestamp_cols = [col for col in data1.columns if col.startswith("Timestamps")]
#     data1 = data1.sort_values(by=timestamp_cols[0], ascending=True)
#     timestamps1 = data1[timestamp_cols[0]]
#     x_values1 = data1['X Values']
#     y_values1 = data1['Y Values']
#     z_values1 = data1['Z Values']

#     has_p1 = field_pFileName1 is not None
#     has_p2 = field_pFileName2 is not None

#     if has_p1:
#         sort_csv_by_first_column(field_pFileName1)
#         data2 = pd.read_csv(field_pFileName1, comment='#')
#         data2 = data2.sort_values(by=timestamp_cols[0], ascending=True)
#         timestamps2 = data2[timestamp_cols[0]]
#         x_values2 = data2['X Values']
#         y_values2 = data2['Y Values']
#         z_values2 = data2['Z Values']

#     if has_p2:
#         sort_csv_by_first_column(field_pFileName2)
#         data3 = pd.read_csv(field_pFileName2, comment='#')
#         data3 = data3.sort_values(by=timestamp_cols[0], ascending=True)
#         timestamps3 = data3[timestamp_cols[0]]
#         x_values3 = data3['X Values']
#         y_values3 = data3['Y Values']
#         z_values3 = data3['Z Values']

#     if has_p1 or has_p2:
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
#         ax1.plot(timestamps1, x_values1, label='x', marker='o')
#         ax1.plot(timestamps1, y_values1, label='y', marker='o')
#         ax1.plot(timestamps1, z_values1, label='z', marker='o')
#         ax1.set_title('Incident Electric Field')
#         ax1.set_xlabel(timestamp_cols[0])
#         ax1.set_ylabel('Electric Field Magnitude')
#         ax1.legend()

#         if has_p1 and has_p2:
#             ax2.plot(timestamps2, z_values2, label='Without ABC', linewidth=1)
#             ax2.plot(timestamps3, z_values3, label='With ABC', linewidth=5)
#             ax2.set_title("Molecule's Response")
#             ax2.set_xlabel(timestamp_cols[0])
#             ax2.set_ylabel('Polarization Field Magnitude')
#             ax2.legend()
#         elif has_p1:
#             ax2.plot(timestamps2, x_values2, label='x', marker='o')
#             ax2.plot(timestamps2, y_values2, label='y', marker='o')
#             ax2.plot(timestamps2, z_values2, label='z', marker='o')
#             ax2.set_title("Molecule's Response")
#             ax2.set_xlabel(timestamp_cols[0])
#             ax2.set_ylabel('Polarization Field Magnitude')
#             ax2.legend()
#     else:
#         fig, ax1 = plt.subplots(figsize=(7, 5))
#         ax1.plot(timestamps1, x_values1, label='x', marker='o')
#         ax1.plot(timestamps1, y_values1, label='y', marker='o')
#         ax1.plot(timestamps1, z_values1, label='z', marker='o')
#         ax1.set_title('Incident Electric Field')
#         ax1.set_xlabel(timestamp_cols[0])
#         ax1.set_ylabel('Electric Field Magnitude')
#         ax1.legend()

#     plt.tight_layout()
#     if matplotlibLocationIMG is None:
#         if output_image_path is None:
#             plt.savefig('output.png', dpi=1000)
#             logging.info("Matplotlib image written: output.png")
#         else:
#             plt.savefig(f'{output_image_path}.png', dpi=1000)
#             logging.info(f"Matplotlib image written: {output_image_path}.png")
#     elif output_image_path is None:
#         plt.savefig(f'{matplotlibLocationIMG}.png', dpi=1000)
#         logging.info(f"Matplotlib image written: {matplotlibLocationIMG}.png")
#     else:
#         plt.savefig(f'{matplotlibLocationIMG}{output_image_path}.png', dpi=1000)
#         logging.info(f"Matplotlib image written: {matplotlibLocationIMG}{output_image_path}.png")