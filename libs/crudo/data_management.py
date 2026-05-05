#
# Data Management: A library for managing HDF5 data and simulation files.
#

from . import utilities      as ut

from datetime import datetime
import h5py
import numpy as np
import os
import pandas as pd
from typing import Callable, List, Optional, Tuple, Union


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
        # print(f"Checking for file: {file_name}")
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

def apply_cut_and_update(
                            df_doro: pd.DataFrame,
                            df_soph: pd.DataFrame,
                            event_ids: Optional[Union[List[int], np.ndarray]] = None,
                            cut_mask: Optional[pd.Series] = None, 
                            df_for_mask: Optional[pd.DataFrame] = None
                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Selects events from Dorothea and Sophronia DataFrames based on a cut.

        This function provides two ways to select events:
        1. By providing a boolean mask (`cut_mask`) applied to a specified DataFrame (`df_for_mask`).
        2. By providing a direct list/array of event IDs to keep (`event_ids`).
        
        The function ensures that both Dorothea and Sophronia are filtered consistently
        to keep the same set of events.

        Args:
            df_doro (pd.DataFrame): The Dorothea DataFrame.
            df_soph (pd.DataFrame): The Sophronia DataFrame.
            event_ids (list or np.array, optional): An explicit list of event IDs to keep.
            cut_mask (pd.Series, optional): A boolean mask. Rows where the mask is True
                                            will be used to identify events to keep.
                                            Must be provided along with `df_for_mask`.
            df_for_mask (pd.DataFrame, optional): The DataFrame to which the `cut_mask`
                                                  should be applied. Required if `cut_mask` is used.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the updated (filtered)
                                               Dorothea and Sophronia DataFrames.

        Notes:
            - This function can be used with any pair of dataframes that share an 'event' column,
              as long as the appropriate mask or event IDs are provided. Not limited to Dorothea and Sophronia.
        """
        if event_ids is None and cut_mask is None:
            raise ValueError("Either `event_ids` or `cut_mask` must be provided.")
        if event_ids is not None and cut_mask is not None:
            raise ValueError("Provide either `event_ids` or `cut_mask`, not both.")
        if cut_mask is not None and df_for_mask is None:
            raise ValueError("`df_for_mask` must be provided when using `cut_mask`.")
        
        # Determine final event IDs to keep
        final_ids_to_keep = None
        if event_ids is not None:
            final_ids_to_keep = event_ids
        elif cut_mask is not None:
            final_ids_to_keep = df_for_mask.loc[cut_mask, 'event'].unique()
        
        df_doro_updated = df_doro[df_doro['event'].isin(final_ids_to_keep)].copy()
        df_soph_updated = df_soph[df_soph['event'].isin(final_ids_to_keep)].copy()

        return df_doro_updated, df_soph_updated


# Aggregated data management functions for loading, filtering, merging, and saving data

# def get_primary_pulse_info(df_doro: pd.DataFrame, reco_size_variable='old_reco_size') -> pd.DataFrame:
#     """
#     Aggregates Dorothea data to the event level.

#     For each event, it calculates:
#     - General event information (time, nS1, nS2).
#     - Properties of the S1 pulse with the maximum energy ('S1e').
#     - Properties of the S2 pulse with the maximum energy ('S2e').

#     Args:
#         df_doro (pd.DataFrame): The input Dorothea DataFrame (peak-level).

#     Returns:
#         pd.DataFrame: An event-level summary DataFrame.
#     """
#     # --- Input Validation --- #
#     required_columns = {
#         'event', 'time', reco_size_variable, 'nS1', 'nS2', 'S1e', 'S1e_corr', 'S1w', 'S1h', 'S1t',
#         'S2e', 'S2w', 'S2h', 'S2t', 'S2q'
#     }
#     missing_columns = required_columns - set(df_doro.columns)
#     if missing_columns:
#         raise ValueError(f"Input DataFrame is missing required columns: {sorted(list(missing_columns))}")

#     if df_doro.empty:
#         return pd.DataFrame()

#     # --- Event-Level Simple Aggregations ---
#     # These are properties that are constant per event or where a simple operation is sufficient
#     event_level_simple_agg = df_doro.groupby('event').agg(
#         time = ('time', 'mean'),
#         old_reco_size = (reco_size_variable, 'first'),
#         nS1  = ('nS1', 'first'),
#         nS2  = ('nS2', 'first')
#     )

#     # --- Identify Primary S1 and S2 Pulses ---
#     # Handle first events without S1
#     s1_valid_mask = df_doro['S1e'].notna()
#     if s1_valid_mask.any():
#         # .idxmax() finds the index label of the row with the maximum value for each group.
#         idx_primary_s1 = df_doro[s1_valid_mask].groupby('event')['S1e'].idxmax()
#         # Select the full rows of these primary pulses
#         primary_s1_df = df_doro.loc[idx_primary_s1].set_index('event')
#     else:
#         # If no valid S1 pulses, create an empty DataFrame
#         primary_s1_df = pd.DataFrame(index=event_level_simple_agg.index)
    
#     # S2 pulses are simpler
#     idx_primary_s2 = df_doro.groupby('event')['S2e'].idxmax()
#     primary_s2_df  = df_doro.loc[idx_primary_s2].set_index('event')

#     # --- Combine ALL Information ---
#     # Start with the simple aggregations
#     doro_info_df = event_level_simple_agg
#     # Add the S1 information
#     doro_info_df = doro_info_df.join(
#         primary_s1_df[['S1e', 'S1e_corr', 'S1w', 'S1h', 'S1t']].add_prefix('main_')
#     )
#     # Add the S2 information
#     doro_info_df = doro_info_df.join(
#         primary_s2_df[['S2e', 'S2w', 'S2h', 'S2t', 'S2q']].add_prefix('main_')
#     )
#     # Reset the index to make 'event' a column again
#     doro_info_df = doro_info_df.reset_index()

#     return doro_info_df

def get_primary_pulse_info(df_doro: pd.DataFrame, event_level_cols: List[str]) -> pd.DataFrame:
    """
    Aggregates Dorothea data to the event level in a controlled, scalable way.

    For each event, it calculates:
    - User-defined event-level properties by taking the 'first' value.
    - Properties of the S1 pulse with the maximum energy ('S1e').
    - Properties of the S2 pulse with the maximum energy ('S2e').

    Args:
        df_doro (pd.DataFrame): The input Dorothea DataFrame (peak-level).
        event_level_cols (list of str): List of column names to aggregate at the event level.

    Returns:
        pd.DataFrame: An event-level summary DataFrame.
    """
    # EVENT_LEVEL_COLS = event_level_cols
    S1_PULSE_COLS = ['S1e', 'S1e_corr', 'S1w', 'S1h', 'S1t']
    S2_PULSE_COLS = ['S2e', 'S2w', 'S2h', 'S2t', 'S2q']

    # --- Input Validation --- #
    required_columns = set(['event', 'time'] + event_level_cols + S1_PULSE_COLS + S2_PULSE_COLS)
    missing_columns = required_columns - set(df_doro.columns)
    if missing_columns:
        raise ValueError(f"Input DataFrame is missing required columns: {sorted(list(missing_columns))}")

    if df_doro.empty:
        return pd.DataFrame()

    # --- Event-Level Simple Aggregations ---
    # These are properties that are constant per event or where a simple operation is sufficient
    simple_agg_dict = {col: (col, 'first') for col in event_level_cols}
    simple_agg_dict['time'] = ('time', 'mean')  # Special case for 'time'
    event_level_simple_agg = df_doro.groupby('event').agg(**simple_agg_dict)

    # --- Identify Primary S1 and S2 Pulses ---
    # Handle first events without S1
    s1_valid_mask = df_doro['S1e'].notna()
    if s1_valid_mask.any():
        # .idxmax() finds the index label of the row with the maximum value for each group.
        idx_primary_s1 = df_doro[s1_valid_mask].groupby('event')['S1e'].idxmax()
        # Select the full rows of these primary pulses
        primary_s1_df = df_doro.loc[idx_primary_s1].set_index('event')
    else:
        # If no valid S1 pulses, create an empty DataFrame
        primary_s1_df = pd.DataFrame(index=event_level_simple_agg.index, columns=S1_PULSE_COLS)
    
    # S2 pulses are simpler
    idx_primary_s2 = df_doro.groupby('event')['S2e'].idxmax()
    primary_s2_df  = df_doro.loc[idx_primary_s2].set_index('event')

    # --- Combine ALL Information ---
    # Start with the simple aggregations
    doro_info_df = event_level_simple_agg
    # Add the S1 information
    doro_info_df = doro_info_df.join(primary_s1_df[S1_PULSE_COLS].add_prefix('main_'))
    # Add the S2 information
    doro_info_df = doro_info_df.join(primary_s2_df[S2_PULSE_COLS].add_prefix('main_'))
    # Reset the index to make 'event' a column again
    doro_info_df = doro_info_df.reset_index()

    return doro_info_df

def summarize_hits_to_event_level(df_hits: pd.DataFrame, energy_column='E_hit_mev', size_variable='event_size') -> pd.DataFrame:
    """
    Aggregates hit-level DataFrame to a final event-level summary.

    This function calculates various event-level properties including barycenters,
    spatial extent, total energy, and cluster multiplicity.

    Args:
        df_hits (pd.DataFrame): The input hit-level DataFrame.
                                Must have columns like 'event', 'X', 'Y', 'Z', 'energy', 'cluster'.

    Returns:
        pd.DataFrame: An event-level summary DataFrame with one row per event.
    """
    # --- Input Validation --- #
    required_columns = {'event', 'X', 'Y', 'Z', energy_column, 'cluster'}
    missing_columns = required_columns - set(df_hits.columns)
    if missing_columns:
        raise ValueError(f"Input DataFrame is missing required columns: {sorted(list(missing_columns))}")

    if df_hits.empty:
        return pd.DataFrame()
    
    # --- Preliminary Computations ---    
    # Pre-calculate R^2 to find R_max efficiently
    df_hits['R_sq'] = df_hits['X']**2 + df_hits['Y']**2
    # Pre-calculate weighted coordinates for barycenter calculation
    df_hits['X_w'] = df_hits['X'] * df_hits[energy_column]
    df_hits['Y_w'] = df_hits['Y'] * df_hits[energy_column]
    df_hits['Z_w'] = df_hits['Z'] * df_hits[energy_column]
    # Event size
    event_size_df = df_hits.groupby('event').size().rename(size_variable)

    # --- Event-Level Simple Aggregations ---
    soph_info_df = df_hits.groupby('event').agg(
        X_w_sum   = ('X_w', 'sum'),
        Y_w_sum   = ('Y_w', 'sum'),
        Z_w_sum   = ('Z_w', 'sum'),
        X_min     = ('X', 'min'),
        X_max     = ('X', 'max'),
        Y_min     = ('Y', 'min'),
        Y_max     = ('Y', 'max'),
        Z_min     = ('Z', 'min'),
        Z_max     = ('Z', 'max'),
        R_sq      = ('R_sq', 'max'),
        E_evt_mev  = (energy_column, 'sum'),        
        n_cluster = ('cluster', 'max')
    )
    
    # --- Post-Aggregation Computation ---    
    # Calculate R_max from the max of R_sq
    soph_info_df['R_max'] = np.sqrt(soph_info_df['R_sq'])
    # Calculate the barycenters
    # .replace(0, 1) prevents division by zero for events with zero energy.
    total_energy = soph_info_df['E_evt_mev'].replace(0, 1)
    soph_info_df['X_bary'] = soph_info_df['X_w_sum'] / total_energy
    soph_info_df['Y_bary'] = soph_info_df['Y_w_sum'] / total_energy
    soph_info_df['Z_bary'] = soph_info_df['Z_w_sum'] / total_energy
    # Adjust n_cluster to be number of clusters (max_label + 1)
    soph_info_df['n_cluster'] = soph_info_df['n_cluster'] + 1
    # Event size
    soph_info_df = soph_info_df.join(event_size_df)
    
    # Clean up and reorder columns for a nice output
    final_columns = [
        'X_bary', 'Y_bary', 'Z_bary', 'X_min', 'X_max', 'Y_min', 'Y_max', 'Z_min', 'Z_max', 'R_max', 
        'E_evt_mev', 'n_cluster', size_variable
    ]
    
    return soph_info_df[final_columns].reset_index()
    

def aggregate_to_event_peak_level(df_doro: pd.DataFrame, df_soph: pd.DataFrame, event_level_cols: List[str], energy_column='E_hit_mev') -> pd.DataFrame:
    """
    Aggregates hit-level data to event/peak-level summary data.
    This df_soph should be the FINAL clean hits dataframe after spurious hits treatment.
    """
    if df_soph.empty:
        return pd.DataFrame()

    # ----- Dorothea Info ----- #
    doro_info_df = get_primary_pulse_info(df_doro, event_level_cols)

    # ----- Sophronia Info ----- #
    # Event-level
    soph_event_info_df = summarize_hits_to_event_level(df_soph, energy_column=energy_column)
    # Peak-level
    soph_peak_info_df = df_soph.groupby(['event', 'npeak']).agg(E_peak_mev=(energy_column, 'sum')).reset_index()

    # ----- Merge Final DataFrame ----- #
    df_file = pd.merge(soph_peak_info_df, soph_event_info_df, on='event', how='left')
    df_file = pd.merge(df_file, doro_info_df, on='event', how='left')

    return df_file

def tag_particles(
                    df_peak_level: pd.DataFrame,
                    size_threshold: int,
                    s1_energy_threshold: float,
                    event_column: str = 'event'
                 ) -> pd.DataFrame:
    """
    Tags each peak in a DataFrame as 'electron' or 'alpha' based on its parent event's properties.

    The classification is done at the event-level and then mapped back to each peak.
    1. For events with nS1 = 1, classification is based on the S1 corrected energy.
    2. For events with nS1 = 0, classification is based on the event size (number of hits).
    3. Events with nS1 > 1 are tagged as 'unclassified'.

    Args:
        df_peak_level (pd.DataFrame): A DataFrame with one row per peak (event/peak level).
                                      Must contain 'event', 'nS1', 'reco_size',
                                      and 'main_S1e_corr' columns.
        size_threshold (int): The threshold on the number of hits for nS1=0 events.
        s1_energy_threshold (float): The threshold on S1 energy for nS1=1 events.

    Returns:
        pd.DataFrame: The input DataFrame with a new 'particle' column added.
    """

    if df_peak_level.empty:
        df_peak_level['particle'] = pd.Series(dtype='object')
        return df_peak_level
    
    # Event-level information
    event_summary = df_peak_level.groupby(event_column).agg(
                                                                nS1=('nS1', 'first'),
                                                                reco_size=('reco_size', 'first'),
                                                                main_S1e_corr=('main_S1e_corr', 'first')
    )

    # Masks of S1 multiplicity
    is_ns1_zero = (event_summary['nS1'] == 0)
    is_ns1_one  = (event_summary['nS1'] == 1)
    # is_ns1_multiple = (df_doro['nS1'] > 1)    # To explicitly handle this case

    # Masks for size and energy-based classification
    is_small_size = (event_summary['reco_size'] <= size_threshold)
    is_s1_low_energy = (event_summary['main_S1e_corr'] <= s1_energy_threshold)
    
    # Particle classification logic based on the defined conditions
    conditions = [
                    is_ns1_zero & is_small_size,        # Case 1: nS1=0 and small size event -> electron
                    is_ns1_zero & ~is_small_size,       # Case 2: nS1=0 and large size event -> alpha
                    is_ns1_one & is_s1_low_energy,      # Case 3: nS1=1 and low S1 energy    -> electron
                    is_ns1_one & ~is_s1_low_energy,     # Case 4: nS1=1 and high S1 energy   -> alpha
                 ]

    choices = ['electron', 'alpha', 'electron', 'alpha']

    # Map event-level classification back to peak-level DataFrame
    event_summary['particle'] = np.select(conditions, choices, default='unclassified')
    event_to_particle_map = event_summary['particle']

    df_peak_level['particle'] = df_peak_level[event_column].map(event_to_particle_map)
    
    return df_peak_level

def tag_event_by_detector_region(
                                    df_peak_level: pd.DataFrame,
                                    z_cut_low: float,
                                    z_cut_high: float,
                                    r_cut_high: float,
                                    event_column: str = 'event'
                                ) -> pd.Series:
    """
    Assigns a detector region tag to each event based on its full track extent.

    The classification is sequential and mutually exclusive, following this priority:
    1.  Anode (NO S1)
    2.  Anode (track crosses low-Z boundary)
    3.  Cathode (track crosses high-Z boundary)
    4.  Fiducial (fully contained in Z and R)
    5.  Tube (fully contained in Z, but outside R cut)
    6.  Unclassified (should not happen with this logic, but included for safety)

    Args:
        df_peak_level (pd.DataFrame): A DataFrame with one row per peak (event/peak level).. 
                                      Must contain columns like 'Z_min', 'Z_max', 'R_max', and 'nS1'.
        z_cut_low (float): The lower Z boundary for the fiducial volume.
        z_cut_high (float): The upper Z boundary for the fiducial volume.
        r_cut_high (float): The radial boundary for the fiducial volume.
        event_col (str): The name of the column representing the event ID. Default is 'event'.

    Returns:
        pd.DataFrame: The input DataFrame with a new 'region' column added.
    """
    if df_peak_level.empty:
        df_peak_level['region'] = pd.Series(dtype='object')
        return df_peak_level
    
    # Event-level information
    event_summary = df_peak_level.groupby(event_column).agg(
                                                                nS1=('nS1', 'first'),
                                                                Z_min=('Z_min', 'first'),
                                                                Z_max=('Z_max', 'first'),
                                                                R_max=('R_max', 'first'),
                                                            )

    # Base masks
    is_ns1_zero = (event_summary['nS1'] == 0)
    crosses_anode_z   = (event_summary['Z_min'] <= z_cut_low)
    crosses_cathode_z = (event_summary['Z_max'] >= z_cut_high)
    is_fully_z_contained = ((event_summary['Z_min'] > z_cut_low) & (event_summary['Z_max'] < z_cut_high))
    is_r_contained = (event_summary['R_max'] < r_cut_high)

    # Conditions and choices for np.select (priority order matters!)    
    conditions = [
                    is_ns1_zero,                                # 0. If no S1, it's Anode.
                    crosses_anode_z,                            # 1. If any part is in anode Z, it's Anode.
                    crosses_cathode_z,                          # 2. If any part is in cathode Z, it's Cathode.
                    is_fully_z_contained & is_r_contained,      # 3. If contained in Z and R, it's Fiducial.
                    is_fully_z_contained & ~is_r_contained      # 4. If contained in Z but not R, it's Tube.
                 ]
    choices = ['anode', 'anode', 'cathode', 'fiducial', 'tube']

    # Map event-level classification back to peak-level DataFrame
    event_summary['region'] = np.select(conditions, choices, default='unclassified')
    event_to_particle_map = event_summary['region']

    df_peak_level['region'] = df_peak_level[event_column].map(event_to_particle_map)
    
    return df_peak_level