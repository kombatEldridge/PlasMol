import os
import pandas as pd
import matplotlib.pyplot as plt


def find_csv_files(directory="."):
    """Recursively find all CSV files in the directory and subdirectories."""
    csv_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

def display_files(files):
    """Display the CSV files as a numbered list."""
    print("Available CSV files:")
    for idx, file in enumerate(files, start=1):
        print(f"{idx}. {file}")
    print()

def get_user_selection(num_files):
    """Prompt the user for file selection."""
    selection = input(f"Enter the numbers of the files to plot (comma-separated, 1-{num_files}): ")
    try:
        selected_indices = [int(i) - 1 for i in selection.split(',')]
        if all(0 <= idx < num_files for idx in selected_indices):
            return selected_indices
        else:
            print("Invalid selection. Please enter valid numbers.")
            return get_user_selection(num_files)
    except ValueError:
        print("Invalid input. Please enter numbers only.")
        return get_user_selection(num_files)

def plot_csv_files(files):
    """Read and plot a CSV file."""
    plt.figure(figsize=(10, 6))

    for file in files:
        try:
            data = pd.read_csv(file, comment='#')
            
            if 'X Values' in data.columns:
                plt.plot(data['Timestamps (fs)'], data['X Values'], label="-".join([os.path.basename(file), "x"]), marker='o')
            if 'Y Values' in data.columns:
                plt.plot(data['Timestamps (fs)'], data['Y Values'], label="-".join([os.path.basename(file), "y"]), marker='o')
            if 'Z Values' in data.columns:
                plt.plot(data['Timestamps (fs)'], data['Z Values'], label="-".join([os.path.basename(file), "z"]), marker='o')
            
        except Exception as e:
            print(f"Error reading {file}: {e}")

    save = input("Should I save the plot? (y/n): ")
    if save == 'y':
        name = input("What shall I name the file? ")
        if name:
            file_name = name
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
            file_name = f"combinedCSV_{timestamp}.png"
        plt.savefig(file_name, dpi=500)

    if name:
        plt.title(name)
    else:
        plt.title("Combined Plot of Selected CSV Files")
    plt.xlabel("Timestamps (fs)")
    plt.ylabel("Electric Field Magnitude")
    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.show()

def main():
    csv_files = find_csv_files()
    if not csv_files:
        print("No CSV files found in the current directory or subdirectories.")
        return

    display_files(csv_files)
    selected_indices = get_user_selection(len(csv_files))
    selected_files = [csv_files[idx] for idx in selected_indices]

    plot_csv_files(selected_files)

if __name__ == "__main__":
    main()
