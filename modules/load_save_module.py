import os
import re
import glob
import pandas as pd


def loadData(filename: str, block_num: int) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Load ECG and EDA data from a file.

    """
    # Define the base path
    base_path = (
        f"datasets/CLAS_database/CLAS_database/CLAS/Participants/{filename}/by_block/"
    )

    # Find the ECG file dynamically
    ecg_files = glob.glob(os.path.join(base_path, f"{block_num}_ecg_*.csv"))
    ecg_signal_file = (
        ecg_files[0] if ecg_files else None
    )  # Select the first match if available

    # Find the EDA file dynamically
    eda_files = glob.glob(os.path.join(base_path, f"{block_num}_gsr_ppg_*.csv"))
    eda_signal_file = (
        eda_files[0] if eda_files else None
    )  # Select the first match if available

    # Check if file exists before proceeding
    if ecg_signal_file is None:
        print(f"Error: File '{block_num}_ecg_*.csv' not found.")
        return None, None

    if eda_signal_file is None:
        print(f"Error: File '{block_num}_gsr_ppg_*.csv' not found.")
        return None, None

    # Load ECG & EDA data
    print(f"Loading ECG data from {ecg_signal_file}")
    print(f"Loading EDA data from {eda_signal_file}")
    ecg_signal_full = pd.read_csv(ecg_signal_file)
    eda_signal_full = pd.read_csv(eda_signal_file)

    return ecg_signal_full, eda_signal_full


def saveData(data, signal_type, dest_dir1, dest_dir2, part_num, block_num):
    """
    Saves CSV file into one of two destination directories
    based on the starting number of the file name.

    Args:
        data (np.Array): array containing cleaned signal.
        signal_type (str): Type of signal being saved (ecg or eda).
        dest_dir1 (str): Path to the low load directory.
        dest_dir2 (str): Path to the high load directory.
        part_num (int): Participant number.
        block_num (int): Block number
    """
    setup1LowLoad = [1, 4, 7, 10, 15, 20, 25, 30, 32, 34, 36]
    highLoad = [2, 5, 8]
    setup2LowLoad = [1, 4, 7, 10, 12, 14, 16, 18, 23, 28, 33]

    num_pattern = r"(-?\d+)"  # Regex pattern to extract number from file name
    match = re.search(num_pattern, part_num)
    if match:
        part_num = int(match.group(1))
    else:
        print("Error: Unable to extract participant number.")
        return

    if part_num <= 11:
        if block_num in setup1LowLoad:
            dest_dir = dest_dir1
        elif block_num in highLoad:
            dest_dir = dest_dir2
        else:
            print("Block was skipped")
            return

    elif part_num >= 12:
        if block_num in setup2LowLoad:
            dest_dir = dest_dir1
        elif block_num in highLoad:
            dest_dir = dest_dir2
        else:
            print("Block was skipped")
            return

    if signal_type == "ecg":
        filename = f"{part_num}_{block_num}_ecg.csv"
    elif signal_type == "eda":
        filename = f"{part_num}_{block_num}_eda.csv"
    else:
        print("Error: Invalid signal type.")
        return

    pd.DataFrame(data).to_csv(f"{dest_dir}/{filename}", index=False, header=False)
    print(f"File saved to {dest_dir}/{filename}")
