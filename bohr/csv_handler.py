import pandas as pd
import csv
import logging
import matplotlib.pyplot as plt
import os


def initCSV(filename, comment):
    with open(filename, 'w', newline='') as file:
        for line in comment.splitlines():
            file.write(f"# {line}\n")
        file.write("\n")
        writer = csv.writer(file)
        header = ['Timestamps (fs)']
        header.append('X Values')
        header.append('Y Values')
        header.append('Z Values')
        writer.writerow(header)


def updateCSV(filename, timestamp, x_value=None, y_value=None, z_value=None):
    file_exists = os.path.exists(filename)
    row = [timestamp]
    row.append(x_value if x_value is not None else 0)
    row.append(y_value if y_value is not None else 0)
    row.append(z_value if z_value is not None else 0)
    
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "X", "Y", "Z"])
        writer.writerow(row)


def show_eField_pField(eFieldFileName, matplotlibLocationIMG, matplotlibOutput, pFieldFileName=None):
    logging.getLogger('matplotlib').setLevel(logging.INFO)

    if pFieldFileName is not None:
        logging.debug(
            f"Reading CSV files: {eFieldFileName} and {pFieldFileName}")
    else:
        logging.debug(f"Reading CSV file: {eFieldFileName}")

    def sort_csv_by_first_column(filename):
        with open(filename, 'r') as file:
            lines = file.readlines()

        comments = [line for line in lines if line.startswith('#')]
        header = next(line for line in lines if not line.startswith('#'))
        data_lines = [line for line in lines if not line.startswith('#') and line != header]

        from io import StringIO
        data = pd.read_csv(StringIO(''.join(data_lines)))

        data_sorted = data.sort_values(by='Timestamps (fs)')

        with open(filename, 'w') as file:
            file.writelines(comments)
            file.write(header)
            data_sorted.to_csv(file, index=False)

    sort_csv_by_first_column(eFieldFileName)
    data1 = pd.read_csv(eFieldFileName, comment='#')
    data1 = data1.sort_values(by='Timestamps (fs)', ascending=True)
    timestamps1 = data1['Timestamps (fs)']
    x_values1 = data1['X Values']
    y_values1 = data1['Y Values']
    z_values1 = data1['Z Values']

    if pFieldFileName is not None:
        sort_csv_by_first_column(pFieldFileName)
        data2 = pd.read_csv(pFieldFileName, comment='#')
        data2 = data2.sort_values(by='Timestamps (fs)', ascending=True)
        timestamps2 = data2['Timestamps (fs)']
        x_values2 = data2['X Values']
        y_values2 = data2['Y Values']
        z_values2 = data2['Z Values']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        ax1.plot(timestamps1, x_values1, label='x', marker='o')
        ax1.plot(timestamps1, y_values1, label='y', marker='o')
        ax1.plot(timestamps1, z_values1, label='z', marker='o')

        ax1.set_title('Incident Electric Field')
        ax1.set_xlabel('Timestamps (fs)')
        ax1.set_ylabel('Electric Field Magnitude')
        ax1.legend()

        ax2.plot(timestamps2, x_values2, label='x', marker='o')
        ax2.plot(timestamps2, y_values2, label='y', marker='o')
        ax2.plot(timestamps2, z_values2, label='z', marker='o')
        ax2.set_title('Molecule\'s Response')
        ax2.set_xlabel('Timestamps (fs)')
        ax2.set_ylabel('Polarization Field Magnitude')
        ax2.legend()
    else:
        fig, ax1 = plt.subplots(figsize=(7, 5))

        ax1.plot(timestamps1, x_values1, label='x', marker='o')
        ax1.plot(timestamps1, y_values1, label='y', marker='o')
        ax1.plot(timestamps1, z_values1, label='z', marker='o')
        ax1.set_title('Incident Electric Field')
        ax1.set_xlabel('Timestamps (fs)')
        ax1.set_ylabel('Electric Field Magnitude')
        ax1.legend()

    plt.tight_layout()
    plt.savefig(f'{matplotlibLocationIMG}{matplotlibOutput}.png', dpi=1000)
    logging.debug(f"Matplotlib image written: {matplotlibLocationIMG}{matplotlibOutput}.png")


def show_eField_Energy(eFieldFileName, energyFileName, matplotlibLocationIMG, matplotlibOutput):
    logging.getLogger('matplotlib').setLevel(logging.INFO)

    logging.debug(f"Reading CSV files: {eFieldFileName} and {energyFileName}")

    def sort_csv_by_first_column(filename):
        with open(filename, 'r') as file:
            lines = file.readlines()

        comments = [line for line in lines if line.startswith('#')]
        header = next(line for line in lines if not line.startswith('#'))
        data_lines = [line for line in lines if not line.startswith('#') and line != header]

        from io import StringIO
        data = pd.read_csv(StringIO(''.join(data_lines)))

        data_sorted = data.sort_values(by='Timestamps (fs)')

        with open(filename, 'w') as file:
            file.writelines(comments)
            file.write(header)
            data_sorted.to_csv(file, index=False)

    sort_csv_by_first_column(eFieldFileName)
    data1 = pd.read_csv(eFieldFileName, comment='#')
    data1 = data1.sort_values(by='Timestamps (fs)', ascending=True)
    timestamps1 = data1['Timestamps (fs)']
    x_values1 = data1['X Values']
    y_values1 = data1['Y Values']
    z_values1 = data1['Z Values']

    sort_csv_by_first_column(energyFileName)
    data2 = pd.read_csv(energyFileName, comment='#')
    data2 = data2.sort_values(by='Timestamps (fs)', ascending=True)
    timestamps2 = data2['Timestamps (fs)']
    energy = data2['X Values']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.plot(timestamps1, x_values1, label='x', marker='o')
    ax1.plot(timestamps1, y_values1, label='y', marker='o')
    ax1.plot(timestamps1, z_values1, label='z', marker='o')

    ax1.set_title('Incident Electric Field')
    ax1.set_xlabel('Timestamps (fs)')
    ax1.set_ylabel('Electric Field Magnitude')
    ax1.legend()

    ax2.plot(timestamps2, energy, label='energy', marker='o')
    ax2.set_title('Molecule\'s Response')
    ax2.set_xlabel('Timestamps (fs)')
    ax2.set_ylabel('Energy (a.u.)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{matplotlibLocationIMG}{matplotlibOutput}.png', dpi=1000)
    logging.debug(f"Matplotlib image written: {matplotlibLocationIMG}{matplotlibOutput}.png")
