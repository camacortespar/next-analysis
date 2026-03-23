#
# Data Management: A library for managing HDF5 data and simulation files.
#

from datetime import datetime
import h5py
import os
import pandas as pd
from typing import Callable


def h5_describer(file_path):

    # Function to visualize the structure of an HDF5 file
    def print_structure(name, obj):
        indent = '  ' * (name.count('/'))
        print(f"{indent}{name} ({type(obj).__name__})")
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}  Shape: {obj.shape}, Dtype: {obj.dtype}")

    with h5py.File(file_path, 'r') as f:
        f.visititems(print_structure)

def load_run_data(  
                    run_info, 
                    id=False, 
                    base_path="/lustre/ific.uv.es/prj/gl/neutrinos/users/ccortesp/NEXT-100/Sophronia/Alphas/", 
                    city="sophronia", 
                    trigger=2, 
                    key="/DST/Events", 
                    verbose=True
                ):
    """
    Loads HDF5 data for a specific run and returns a concatenated DataFrame.

    Parameters:
        run_info (dict or int): Run information containing "run_number" or the run ID directly if `id=True`.
        id (bool): If True, `run_info` is treated as the run ID. Default is False.
        base_path (str): Base directory for HDF5 files. Default is the specified path.
        city (str): Subdirectory name for the run files. Default is "sophronia".
        trigger (int or None): Trigger type (e.g., 2 for trg2). Default is 2. If None, no trigger is used.
        key (str): HDF5 dataset path to load. Default is "/DST/Events".
        verbose (bool): If True, prints debug messages. Default is True.

    Returns:
        dict: A dictionary with the run number as the key and the concatenated DataFrame as the value.
              Returns an empty DataFrame if no valid files are found.
    """
    # Determine run ID
    run_id = run_info if id else run_info.get("run_number", None)
    if run_id is None:
        raise ValueError("run_info must contain a 'run_number' key or be the run ID if `id=True`.")

    # Initialize storage for run data and valid file paths
    run_data = {}
    h5_files = []

    # Search for HDF5 files across LDCs (1 to 7)
    for ldc in range(1, 8):
        file_name = os.path.join(base_path, f"run_{run_id}_ldc{ldc}{'_trg2' if trigger == 2 else ''}_{city}.h5")
        if os.path.isfile(file_name):
            h5_files.append(file_name)
        elif verbose:
            print(f"Warning: File {file_name} does not exist, skipping...")

    # Return empty DataFrame if no files found
    if not h5_files:
        if verbose:
            print(f"Warning: No valid files found for run {run_id}.")
        run_data[run_id] = pd.DataFrame()
        return run_data

    # Load and concatenate data from valid files
    dataframes = []
    for file in h5_files:
        try:
            dataframes.append(pd.read_hdf(file, key=key))
        except Exception as e:
            if verbose:
                print(f"Error reading {file}: {e}")

    # Store concatenated DataFrame or empty DataFrame if no data loaded
    if dataframes:
        run_data[run_id] = pd.concat(dataframes, ignore_index=True)
        if verbose:
            print(f"{key}: Run {run_id} successfully loaded with data shape: {run_data[run_id].shape}")
    else:
        if verbose:
            print(f"Warning: No data loaded for run {run_id}.")
        run_data[run_id] = pd.DataFrame()

    return run_data

def filter_run_data(
                        run_info,
                        run_data, 
                        sel_criteria, 
                        id=False, 
                        verbose=True
                    ):
                    
    """
    Filters event-level data for a specific run based on custom criteria.

    Parameters:
        run_info (dict or int): Run information containing "run_number" or the run ID directly if `id=True`.
        run_data (dict of pd.DataFrame): Maps run numbers to their corresponding DataFrames.
        sel_criteria (callable): Function defining the filtering logic for events.        
        id (bool): If True, `run_info` is treated as the run ID. Default is False.
        verbose (bool): If True, prints debug messages. Default is True.

    Returns:
        dict: A dictionary with run numbers as keys and filtered DataFrames as values.
              Returns an empty DataFrame if filtering fails.
    """
    # Determine run ID
    run_id = run_info if id else run_info.get("run_number", None)
    if run_id is None:
        raise ValueError("run_info must contain a 'run_number' key or be the run ID if `id=True`.")

    # Initialize dictionary to store filtered data
    filtered_data = {}

    try:
        # Apply filtering criteria to the DataFrame grouped by 'event'
        filtered_data[run_id] = run_data[run_id].groupby('event').filter(sel_criteria)

        # Print success message if verbose is enabled
        if verbose:
            print(f"Run {run_id} filtered successfully. Data shape: {filtered_data[run_id].shape}")
    
    except Exception as e:
        # Handle errors and store an empty DataFrame for the run
        if verbose:
            print(f"Error filtering run {run_id}: {e}")
        filtered_data[run_id] = pd.DataFrame()

    return filtered_data

def merge_dataframes(
                file1,
                file2,
                output_file
            ):
    """
    Merge two .pkl files containing dictionaries of DataFrames into a single output .pkl file.

    Parameters:
        file1 (str): Path to the first .pkl file.
        file2 (str): Path to the second .pkl file.
        output_file (str): Path to save the merged .pkl file.

    Returns:
        dict: Merged dictionary of DataFrames.
    """
    # Load the .pkl files
    try:
        df1 = pd.read_pickle(file1)
        df2 = pd.read_pickle(file2)
    except Exception as e:
        print(f"Error loading .pkl files: {e}")
        return None

    # Validate that the files contain dictionaries of DataFrames
    if not isinstance(df1, dict) or not isinstance(df2, dict):
        print("Error: Files must contain dictionaries of DataFrames.")
        return None

    # Merge the dictionaries
    merged_data = {**df1, **df2}

    # Save the merged dictionary to a .pkl file
    try:
        with open(output_file, "wb") as f:
            pd.to_pickle(merged_data, f)
        print(f"Merged data saved to: {output_file}")
    except Exception as e:
        print(f"Error saving merged data: {e}")

    return merged_data

def save_dataframes(
                        data,
                        output_path,
                        group_path=""
                    ):
    """
    Saves DataFrames from a nested dictionary into an HDF5 file.

    Parameters:
        data (dict): Nested dictionary containing DataFrames or other dictionaries.
        output_path (str): Path to the HDF5 file.
        group_path (str): HDF5 group path for nested keys. Default is "".

    Returns:
        None
    """
    for city, structure in data.items():
        # Replace '/' in keys to avoid conflicts in HDF5 paths
        save_key = str(city).replace('/', '_SLASH_')

        # Construct the HDF5 key path
        key = f"{group_path}/{save_key}" if group_path else save_key

        if isinstance(structure, pd.DataFrame):
            # Save non-empty DataFrames to HDF5
            if not structure.empty:
                try:
                    structure.to_hdf(
                        output_path,
                        key=key,
                        mode='a',           # Append mode to avoid overwriting
                        complevel=5,        # Moderate compression level
                        complib='blosc',    # Efficient compression library
                        format='table'      # Table format for flexibility
                    )
                    print(f"    Saved DataFrame to key: '{key}' {structure.shape}")
                except Exception as e:
                    print(f"  Error saving DataFrame to key '{key}': {e}")
            else:
                print(f"  Empty DataFrame for key '{key}': not saved.")
        elif isinstance(structure, dict):
            # Recursively process nested dictionaries
            save_dataframes(structure, output_path, key)