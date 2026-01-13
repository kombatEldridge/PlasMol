# utils/plotting.py
import logging
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger("main")

def show_eField_pField(eFieldFileName, pFieldFileName=None, matplotlibLocationIMG=None, matplotlibOutput=None):
    """
    Plot electric field and optionally polarization field from CSV files.

    Generates a plot with one or two subplots depending on input, saving it to a file.

    Parameters:
    eFieldFileName : str
        Path to the electric field CSV file.
    pFieldFileName : str, optional
        Path to the polarization field CSV file (default None).
    matplotlibLocationIMG : str, optional
        Directory to save the plot image (default None).
    matplotlibOutput : str, optional
        Filename for the plot image (default None).

    Returns:
    None
    """
    logging.getLogger('matplotlib').setLevel(logging.INFO)

    if pFieldFileName is not None:
        logging.debug(f"Reading CSV files: {eFieldFileName} and {pFieldFileName}")
    else:
        logging.debug(f"Reading CSV file: {eFieldFileName}")

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

    sort_csv_by_first_column(eFieldFileName)
    data1 = pd.read_csv(eFieldFileName, comment='#')
    timestamp_cols = [col for col in data1.columns if col.startswith("Timestamps")]
    data1 = data1.sort_values(by=timestamp_cols[0], ascending=True)
    timestamps1 = data1[timestamp_cols[0]]
    x_values1 = data1['X Values']
    y_values1 = data1['Y Values']
    z_values1 = data1['Z Values']

    if pFieldFileName is not None:
        sort_csv_by_first_column(pFieldFileName)
        data2 = pd.read_csv(pFieldFileName, comment='#')
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
    if matplotlibLocationIMG is None:
        if matplotlibOutput is None:
            plt.savefig('output.png', dpi=1000)
            logging.info("Matplotlib image written: output.png")
        else:
            plt.savefig(f'{matplotlibOutput}.png', dpi=1000)
            logging.info(f"Matplotlib image written: {matplotlibOutput}.png")
    elif matplotlibOutput is None:
        plt.savefig(f'{matplotlibLocationIMG}.png', dpi=1000)
        logging.info(f"Matplotlib image written: {matplotlibLocationIMG}.png")
    else:
        plt.savefig(f'{matplotlibLocationIMG}{matplotlibOutput}.png', dpi=1000)
        logging.info(f"Matplotlib image written: {matplotlibLocationIMG}{matplotlibOutput}.png")


# def show_eField_2pField(eFieldFileName, pFieldFileName1=None, pFieldFileName2=None, matplotlibLocationIMG=None, matplotlibOutput=None):
#     """
#     Plot electric field and optionally polarization fields from CSV files.

#     Generates a plot with one or two subplots depending on input, saving it to a file.
#     When two pField files are provided, plots the x-component from the first and y-component from the second on a single subplot.

#     Parameters:
#     eFieldFileName : str
#         Path to the electric field CSV file.
#     pFieldFileName1 : str, optional
#         Path to the first polarization field CSV file (default None).
#     pFieldFileName2 : str, optional
#         Path to the second polarization field CSV file (default None).
#     matplotlibLocationIMG : str, optional
#         Directory to save the plot image (default None).
#     matplotlibOutput : str, optional
#         Filename for the plot image (default None).

#     Returns:
#     None
#     """
#     logging.getLogger('matplotlib').setLevel(logging.INFO)

#     if pFieldFileName1 is not None and pFieldFileName2 is not None:
#         logging.debug(f"Reading CSV files: {eFieldFileName}, {pFieldFileName1} and {pFieldFileName2}")
#     elif pFieldFileName1 is not None:
#         logging.debug(f"Reading CSV files: {eFieldFileName} and {pFieldFileName1}")
#     else:
#         logging.debug(f"Reading CSV file: {eFieldFileName}")

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

#     sort_csv_by_first_column(eFieldFileName)
#     data1 = pd.read_csv(eFieldFileName, comment='#')
#     timestamp_cols = [col for col in data1.columns if col.startswith("Timestamps")]
#     data1 = data1.sort_values(by=timestamp_cols[0], ascending=True)
#     timestamps1 = data1[timestamp_cols[0]]
#     x_values1 = data1['X Values']
#     y_values1 = data1['Y Values']
#     z_values1 = data1['Z Values']

#     has_p1 = pFieldFileName1 is not None
#     has_p2 = pFieldFileName2 is not None

#     if has_p1:
#         sort_csv_by_first_column(pFieldFileName1)
#         data2 = pd.read_csv(pFieldFileName1, comment='#')
#         data2 = data2.sort_values(by=timestamp_cols[0], ascending=True)
#         timestamps2 = data2[timestamp_cols[0]]
#         x_values2 = data2['X Values']
#         y_values2 = data2['Y Values']
#         z_values2 = data2['Z Values']

#     if has_p2:
#         sort_csv_by_first_column(pFieldFileName2)
#         data3 = pd.read_csv(pFieldFileName2, comment='#')
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
#         if matplotlibOutput is None:
#             plt.savefig('output.png', dpi=1000)
#             logging.info("Matplotlib image written: output.png")
#         else:
#             plt.savefig(f'{matplotlibOutput}.png', dpi=1000)
#             logging.info(f"Matplotlib image written: {matplotlibOutput}.png")
#     elif matplotlibOutput is None:
#         plt.savefig(f'{matplotlibLocationIMG}.png', dpi=1000)
#         logging.info(f"Matplotlib image written: {matplotlibLocationIMG}.png")
#     else:
#         plt.savefig(f'{matplotlibLocationIMG}{matplotlibOutput}.png', dpi=1000)
#         logging.info(f"Matplotlib image written: {matplotlibLocationIMG}{matplotlibOutput}.png")