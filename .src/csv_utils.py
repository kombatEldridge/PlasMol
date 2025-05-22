# csv_utils.py
import os
import csv
import pandas as pd

def initCSV(filename, comment, unit):
    """
    Initializes a CSV file with a header and comment lines.

    Parameters:
        filename (str): Path to the CSV file.
        comment (str): Comment to include at the beginning of the file.
    """
    with open(filename, 'w', newline='') as file:
        for line in comment.splitlines():
            file.write(f"# {line}\n")
        file.write("\n")
        writer = csv.writer(file)
        header = [f'Timestamps ({unit})', 'X Values', 'Y Values', 'Z Values']
        writer.writerow(header)

def updateCSV(filename, timestamp, x_value=None, y_value=None, z_value=None):
    """
    Appends a row of data to a CSV file.

    Parameters:
        filename (str): Path to the CSV file.
        timestamp (float): Time stamp.
        x_value (float, optional): Value for x component.
        y_value (float, optional): Value for y component.
        z_value (float, optional): Value for z component.
    """
    file_exists = os.path.exists(filename)
    row = [timestamp, x_value if x_value is not None else 0,
           y_value if y_value is not None else 0,
           z_value if z_value is not None else 0]
    
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            raise RuntimeError(f"{filename} hasn't been initialized yet. Call 'initCSV' before calling 'updateCSV'.")
        writer.writerow(row)

def read_electric_field_csv(file_path):
    """
    Reads electric field values from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        tuple: Four lists corresponding to time values, and electric field components (x, y, z).
    """
    time_values, electric_x, electric_y, electric_z = [], [], [], []
    with open(file_path, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row and row[0].startswith("Timestamps"):
                break
        for row in reader:
            if len(row) < 4:
                continue
            time_values.append(float(row[0]))
            electric_x.append(float(row[1]))
            electric_y.append(float(row[2]))
            electric_z.append(float(row[3]))
    return time_values, electric_x, electric_y, electric_z