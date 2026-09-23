import csv
import sys

# Check if the correct number of command-line arguments is provided
if len(sys.argv) != 4:
    print("Usage: python script.py <csv_file> <column> <output_file>")
    sys.exit(1)

# Get the CSV file path, column specifier, and output file from command-line arguments
csv_file = sys.argv[1]
column = sys.argv[2].lower()
output_file = sys.argv[3]

# Validate the column specifier
if column not in ['x', 'y', 'z']:
    print("Column must be 'x', 'y', or 'z'")
    sys.exit(1)

# Determine the header and column index based on the selected column
if column == 'x':
    col_index = 1  # Second column
elif column == 'y':
    col_index = 2  # Third column
elif column == 'z':
    col_index = 3  # Fourth column

# Open and read the CSV file
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the title line
    next(reader)  # Skip the header line
    
    # Open the output file for writing
    with open(output_file, 'w', newline='') as out_f:
        writer = csv.writer(out_f)
        
        # Process each data row
        for row in reader:
            if len(row) >= 4:  # Ensure there are at least 4 columns
                timestamp = row[0]  # First column: timestamps
                value = row[col_index]  # Selected column: X, Y, or Z
                writer.writerow([timestamp, value])  # Write to output file
            else:
                print(f"Warning: Skipping row with insufficient columns: {row}")

# Print a success message
print(f"Data saved to {output_file}")
