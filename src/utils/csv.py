# utils/csv.py
import os
import csv
import numpy as np

def initCSV(filename, comment):
    """
    Initialize a CSV file with a header and comment lines.

    Creates a new CSV file with comment lines prefixed by '#', followed by a header row:
    ['Timestamps (au)', 'X Values', 'Y Values', 'Z Values'].

    Parameters:
    filename : str
        Path to the CSV file to be created.
    comment : str
        Comment string to include at the beginning of the file, split into lines.

    Returns:
    None
    """
    with open(filename, 'w', newline='') as file:
        for line in comment.splitlines():
            file.write(f"# {line}\n")
        file.write("\n")
        writer = csv.writer(file)
        header = ['Timestamps (au)', 'X Values', 'Y Values', 'Z Values']
        writer.writerow(header)

def updateCSV(filename, timestamp, x_value=None, y_value=None, z_value=None):
    """
    Append a row of data to an existing CSV file.

    Adds a row with the timestamp and x, y, z values, defaulting to 0 if any value is not provided.
    Raises an error if the file does not exist (i.e., not initialized with initCSV).

    Parameters:
    filename : str
        Path to the CSV file.
    timestamp : float
        The timestamp for the data point in atomic units.
    x_value : float, optional
        Value for the x component, defaults to 0 if None.
    y_value : float, optional
        Value for the y component, defaults to 0 if None.
    z_value : float, optional
        Value for the z component, defaults to 0 if None.

    Returns:
    None
    """
    file_exists = os.path.exists(filename)
    row = [timestamp, x_value if x_value is not None else 0,
           y_value if y_value is not None else 0,
           z_value if z_value is not None else 0]
    
    if not file_exists:
        raise RuntimeError(f"{filename} hasn't been initialized yet. Call 'initCSV' before calling 'updateCSV'.")
    
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

def read_field_csv(file_path):
    """
    Read field values from a CSV file.

    Parses a CSV file with a header starting with 'Timestamps', returning four lists
    for time values and field components (x, y, z).

    Parameters:
    file_path : str
        Path to the CSV file.

    Returns:
    tuple
        A tuple of four lists: (time_values, x, y, z),
        each containing float values.
    """
    time_values, x, y, z = [], [], [], []
    with open(file_path, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row and row[0].startswith("Timestamps"):
                break
        for row in reader:
            if len(row) < 4:
                continue
            time_values.append(float(row[0]))
            x.append(float(row[1]))
            y.append(float(row[2]))
            z.append(float(row[3]))
    return time_values, x, y, z

def apply_damping(mu_arr, tau):
    """
    Apply damping to the polarizability array.

    Applies a damping factor to the polarizability values based on the provided parameters.
    The damping is applied as per the formula: mu_damped = mu * exp(-t/tau).
    Parameters:
    mu_arr : list of float
        The polarizability values to be damped.
    tau : float
        The damping time constant.

    Returns:
    list of float
        The damped polarizability values.
    """
    t = np.array(mu_arr[0])
    damped_mu_x = mu_arr[1] * np.exp(-t / tau)
    damped_mu_y = mu_arr[2] * np.exp(-t / tau)
    damped_mu_z = mu_arr[3] * np.exp(-t / tau)
    return damped_mu_x, damped_mu_y, damped_mu_z