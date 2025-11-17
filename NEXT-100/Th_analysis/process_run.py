#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script is designed to process calibration runs for the NEXT experiment, specifically for 238Th data. 
It automates the workflow of reading raw HDF5 files, applying processing steps, and saving the processed data into new HDF5 files.

This script performs the following high-level steps:

    1. Input Parsing
    2. File Discovery: Automatically identifies all relevant .h5 files in the specified input directory.
    3. Data Processing: Iterates through each file, applies the necessary calibration and processing steps.
    4. Output Generation: Saves the processed data into new .h5 files in the specified output directory.

Usage:
    python process_run.py <run_number> <ldc_number> <n_files>
"""

# ============================================================================
# ----- IMPORTS -----
# ============================================================================

import sys
sys.path.append('/lhome/ific/c/ccortesp/Analysis')

from libs import crudo
from libs import fit_functions as ff
from libs import plotting_tools as pt

import argparse
import csv
import glob
from invisible_cities.reco.corrections import read_maps, apply_all_correction
from invisible_cities.types.symbols import NormStrategy
from invisible_cities.core.core_functions import in_range
from joblib import Parallel, delayed 
import numpy as np
import os
import pandas as pd
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
from sklearn.neighbors import BallTree
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import NotFittedError
from typing import List, Callable, Tuple

# =============================================================================
# ----- CONFIGURATION & ARGUMENT DEFINITION -----
# =============================================================================

# DIRECTORIES, PATHS & FILES
DATA_DIR   = '/lustre/ific.uv.es/prj/gl/neutrinos/users/ccortesp/NEXT-100/Sophronia/Th_runs/'
OUTPUT_DIR = '/lustre/ific.uv.es/prj/gl/neutrinos/users/ccortesp/NEXT-100/Th_analysis/h5/'

SUMMARY_FILENAME = "summary_Th_processing.csv"      # Choose your name
SUMMARY_PATH = os.path.join('/lhome/ific/c/ccortesp/Analysis/NEXT-100/Th_analysis/txt/', SUMMARY_FILENAME)

MAP3D_FILENAME = '/lhome/ific/c/ccortesp/Analysis/NEXT-100/Th_analysis/combined_15546_15557.map3d'
TIME_FILENAME  = '/lhome/ific/c/ccortesp/Analysis/NEXT-100/Th_analysis/energy_scale_he.h5'

# KEYS
DORO_KEY = 'DST/Events'
SOPH_KEY = 'RECO/Events'

# COLUMNS TO USE
DORO_COLUMNS = ['event', 'time', 'nS1', 'nS2', 'S1h', 'S1e', 'S2e', 'DT', 'X', 'Y', 'Z']        # REVISAR SI SE PUEDEN REDUCIR MAS
SOPH_COLUMNS = ['event', 'time', 'npeak', 'X', 'Y', 'Z', 'Q', 'E']

# CUTFLOW
CUT_NAMES = ['IC', 'Cleaning', 'One_S1', 'One_S2', 'Electron_like', 'Processed']

# ANALYSIS PARAMETERS
# --- Drift Velocity --- #
V_DRIFT = 0.865     # in [mm/μs]

# --- Electron-like Events --- #
# Filtered using: S1e < m * DT + b
M_ELEC = 0.32
B_ELEC = 500

# --- Isolated/non-isolated Hits --- #
CLUSTER_CONFIG = {"distance": [16., 16., 4.], "nhit": 3}

def parse_arguments():
    """
    Parses command-line arguments.
    
    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Starting the processing of a Th calibrarion run...")
    
    # ----- Positional (Required) Arguments ----- #
    parser.add_argument("run_number",
                        type=int,
                        help="The run number to process (e.g., 15589).")
    
    parser.add_argument("ldc_number",
                        type=int,
                        choices=range(1, 8),        # Enforces that the value must be in this range
                        metavar="ldc_number[1-7]",  # Provides a hint in the help message
                        help="The LDC number, an integer from 1 to 7.")

    parser.add_argument("n_files",
                        type=str,
                        help="Number of files to process. Use an integer (e.g., 10) or 'All' to process all available files.")

    # If no arguments are provided, print the help message and exit
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    # Parse the arguments from the command line
    args = parser.parse_args()
    
    return args

# =============================================================================
# ----- HELPER FUNCTIONS -----
# =============================================================================

# --- DataFrame Update --- #
def update_dataframes(df_doro, df_soph, evt_ids):
    """
    Updates the Dorothea and Sophronia dataframes to keep only events in evt_ids.
    """
    df_doro_updated = df_doro[df_doro['event'].isin(evt_ids)].copy()
    df_soph_updated = df_soph[df_soph['event'].isin(evt_ids)].copy()
    return df_doro_updated, df_soph_updated

# --- Energy Correction --- #
def get_corr3d(fname):
    krmap = pd.read_hdf(fname, "/krmap")
    meta  = pd.read_hdf(fname, "/mapmeta")
    dtxy_map   = krmap.loc[:, list("zxy")].values
    factor_map = krmap.factor.values
    def corr(dt, x, y, method="nearest"):
        dtxy_data   = np.stack([dt, x, y], axis=1)
        factor_data = griddata(dtxy_map, factor_map, dtxy_data, method=method)
        return factor_data
    return corr
  
def smooth(y, n=4):
    m  = len(y)
    yf = np.flip(y)
    y  = np.concatenate([yf, y, yf])
    z  = np.ones(n) / n
    y  = np.convolve(y, z, mode="same")
    return y[m:2*m]

def get_corrt(fname, n=4):
    time_data = pd.read_hdf(fname, "/data")
    smoothed  = smooth(time_data.e0, n)
    corr      = smoothed.min() / smoothed
    time_correction = interp1d(time_data.time,  corr, "cubic", bounds_error=False, fill_value=(corr[0], corr[-1]))
    return time_correction

# --- Isolated Hits Identification --- #
def split_isolated_clusters_3D(distance: List[float], nhit: int) -> Callable:
    '''
    Tags hits into isolated (or non) using a 3D anisotropic algorithm.
    Does not re-weight any variable.
    '''
    def split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Edge case: If the group is already tiny, all hits are considered isolated
        if len(df) <= nhit: return df.iloc[:0], df
        # Normalise distances
        xyz = df[['X', 'Y', 'Z']].values / distance
        dist = np.sqrt(3)
        # Use NearestNeighbors to find neighbors within the specified radius
        try:
            nbrs = NearestNeighbors(radius=dist, algorithm='ball_tree').fit(xyz)
            neighbors = nbrs.radius_neighbors(xyz, return_distance=False)
            mask_non_iso = np.array([len(neigh) > nhit for neigh in neighbors])
            return df[mask_non_iso], df[~mask_non_iso]      # (non-isolated, isolated)
        except Exception as e:
            print(f"Error in NearestNeighbors: {e}")
            return df.iloc[:0], df.iloc[:0]
    return split

# --- Other Helpers --- #
def weighted_avg(series, weight):
    if weight.sum() == 0:                           # Avoid division by zero
        return np.nan
    return np.average(series, weights=weight)

def R_max_func(group_df):
    return np.sqrt(group_df['X']**2 + group_df['Y']**2).max()

# =============================================================================
# ----- PROCESSING -----
# =============================================================================

def process_file(filepath, cut_names):
    """
    Processes a single HDF5 file and returns its cutflow counts.

    Args:
        filepath (str): The full path to the input .h5 file.
        cut_names (list): The list of cut names to track.
    
    Returns:
        dict: A dictionary with event counts for each cut for this file.
              Returns a dictionary of zeros if processing fails.
    """
    filename = os.path.basename(filepath)
    print(f"→ Processing file: {filename}")
    
    # Initialize counts for this specific file
    local_cut_counts = {name: 0 for name in cut_names}
    
    try:
        # ----- Dorothea ----- #
        df_doro = pd.read_hdf(filepath, key=DORO_KEY)
        df_doro = df_doro.loc[:, DORO_COLUMNS]              # Keep only relevant columns

        # ----- Sophronia ----- #
        df_soph = pd.read_hdf(filepath, key=SOPH_KEY)
        df_soph = df_soph.loc[:, SOPH_COLUMNS]              # Keep only relevant columns
        df_soph.rename(columns={'Z': 'DT'}, inplace=True)   # Rename Z to DT for consistency
        local_cut_counts[cut_names[0]] = df_soph['event'].nunique()

        # ----- Energy Correction (Kr) ----- #
        corr3d = get_corr3d(MAP3D_FILENAME)
        corrt  = get_corrt(TIME_FILENAME)
        df_soph['E_corr'] = df_soph['E'] * corr3d(df_soph['DT'], df_soph['X'], df_soph['Y']) * corrt(df_soph['time'])

        # ----- Cleaning ----- #
        # NaN or negative energy to 0: hit-level
        df_soph['E_corr'] = np.where(
                                        pd.notna(df_soph['E_corr']) & (df_soph['E_corr'] > 0),      # Condition
                                        df_soph['E_corr'],                                          # Value if condition is True
                                        0                                                           # Value if condition is False   
                                    )

        # Drop events with negative Z: event-level
        df_soph['Z'] = df_soph['DT'] * V_DRIFT       # Compute real Z position: using the drift velocity
        Zpos_ids = df_soph.loc[df_soph['Z'] >= 0, 'event'].unique()
        df_doro, df_soph = update_dataframes(df_doro, df_soph, Zpos_ids)    # Updating dataframes
        local_cut_counts[cut_names[1]] = df_soph['event'].nunique()

        # ----- nS1 = 1 ----- #
        S1_ids = df_doro.loc[df_doro['nS1'] == 1, 'event'].unique()
        df_doro, df_soph = update_dataframes(df_doro, df_soph, S1_ids)      # Updating dataframes
        local_cut_counts[cut_names[2]] = df_soph['event'].nunique()

        # ----- nS2 = 1 ----- #
        S2_ids = df_doro.loc[df_doro['nS2'] == 1, 'event'].unique()
        df_doro, df_soph = update_dataframes(df_doro, df_soph, S2_ids)      # Updating dataframes
        local_cut_counts[cut_names[3]] = df_soph['event'].nunique()

        # ----- Electron-like Events ----- #
        elec_ids = df_doro.loc[df_doro['S1e'] < (M_ELEC * df_doro['DT'] + B_ELEC), 'event'].unique()
        df_doro, df_soph = update_dataframes(df_doro, df_soph, elec_ids)    # Updating dataframes
        local_cut_counts[cut_names[4]] = df_soph['event'].nunique()

        # ----- Handling Isolated Hits ----- #
        # STEP A: Tag hits as isolated/non-isolated
        splitter = split_isolated_clusters_3D(**CLUSTER_CONFIG)
        # Indices of isolated hits across all events
        isolated_hits_indices = []
        for _, group in df_soph.groupby(['event', 'npeak']):
            if group.empty: 
                continue
            _, isolated_df = splitter(group)
            if not isolated_df.empty:
                isolated_hits_indices.append(isolated_df.index)
        # Tagging
        df_soph['is_isolated'] = False      # Default to False
        if isolated_hits_indices:
            iso_indices = np.concatenate(isolated_hits_indices)
            df_soph.loc[iso_indices, 'is_isolated'] = True

        # STEP B: Redistribute energy from isolated hits to the non-isolated hits
        isolated_hits_df     = df_soph[df_soph['is_isolated']].copy()
        non_isolated_hits_df = df_soph[~df_soph['is_isolated']].copy()
        non_isolated_hits_df['E_final'] = non_isolated_hits_df['E_corr']        # Final energy column
        if not isolated_hits_df.empty and not non_isolated_hits_df.empty:

            # Z-range of the track for each event
            z_ranges = non_isolated_hits_df.groupby('event')['Z'].agg(['min', 'max']).rename(columns={'min': 'Z_min', 'max': 'Z_max'})
            # Correlate isolated hits with their event's Z-range
            isolated_in_z_range = isolated_hits_df.merge(z_ranges, on='event', how='left')
            isolated_in_z_range.dropna(subset=['Z_min', 'Z_max'], inplace=True)     # Discard hits from events without non-isolated hits

            # Geometric discrimnation: Z-cut
            z_cut_mask = isolated_in_z_range['Z'].between(isolated_in_z_range['Z_min'], isolated_in_z_range['Z_max'])
            kept_iso_hits = isolated_in_z_range[z_cut_mask].copy()

            # # Energy discrimnation: from the signal/background study ---> Martin?
            # # Apply min and max values for energy of the isolated hits to be kept
            # energy_min_signal = 50.0   # Lower bound of the positive peak
            # energy_max_signal = 1000.0
            # energy_cut_mask = kept_iso_hits['E'].between(energy_min_signal, energy_max_signal)
            # kept_iso_hits = kept_iso_hits[energy_cut_mask]

            if not kept_iso_hits.empty:
                # Perform redistribution: QUE SEA UNA FUNCIÓN!
                # Sum the energy of kept isolated hits for each event
                energy_to_add = kept_iso_hits.groupby('event')['E_corr'].sum().rename('E_iso_to_add')

                # Now let's work on non_isolated hits
                non_isolated_hits_df = non_isolated_hits_df.merge(energy_to_add, on='event', how='left')
                non_isolated_hits_df['E_iso_to_add'].fillna(0, inplace=True)        # Events without useful isolated hits get 0 to add

                # Calculate the sum of non-iso energy per event to find proportions
                total_non_iso_energy = non_isolated_hits_df.groupby('event')['E_corr'].transform('sum').replace(0, 1)

                # Calculate proportional redistribution
                non_isolated_hits_df['E_final'] = non_isolated_hits_df['E_corr'] + \
                                                (non_isolated_hits_df['E_corr'] / total_non_iso_energy) * non_isolated_hits_df['E_iso_to_add']                
                # PRINTEAMOS INFO DE ESTO? No creo que nos interese, estará implementado pronto.
                
        # ----- Final Dataframe @ Event-level ----- #
        df_file = non_isolated_hits_df.groupby(['event', 'npeak'], as_index=False).agg(

                    # Time
                    time=('time', 'mean'),
                    # Weighted averages for X, Y, Z
                    X_bary=('X', lambda x: weighted_avg(x, non_isolated_hits_df.loc[x.index, 'E_final'])),
                    Y_bary=('Y', lambda y: weighted_avg(y, non_isolated_hits_df.loc[y.index, 'E_final'])),
                    Z_bary=('Z', lambda z: weighted_avg(z, non_isolated_hits_df.loc[z.index, 'E_final'])),
                    # Sum of Ec
                    E_final=('E_final', 'sum'),
                    # Min and max of Z
                    Z_min=('Z', 'min'),
                    Z_max=('Z', 'max'),
                    # Max R
                    # For R_max, the lambda needs the group DataFrame to access both X and Y
                    R_max=('X', lambda xy_group: R_max_func(non_isolated_hits_df.loc[xy_group.index]))

        )

        local_cut_counts[cut_names[5]] = df_file['event'].nunique()
            
    except Exception as e:
        print(f"   Failed to process file {filename}. Error: {e}", file=sys.stderr)
        # Return a dictionary of zeros on failure to not affect the final sum
        return pd.DataFrame(), pd.DataFrame(), {name: 0 for name in cut_names}

    return df_file, df_soph, local_cut_counts

# =============================================================================
# ----- MAIN -----
# =============================================================================

def main():
    """
    Música maestro! This is the main function that orchestrates the processing
    """
    # 1. Parse command-line arguments and set up paths
    args = parse_arguments()

    # Construct specific input/output directories based on run and LDC
    INPUT_DIR = os.path.join(DATA_DIR, f"{args.run_number}", f"ldc{args.ldc_number}")
    if not os.path.isdir(INPUT_DIR):
        print(f"   Error: Input directory '{INPUT_DIR}' does not exist.", file=sys.stderr)
        sys.exit(1)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_list = sorted(glob.glob(os.path.join(INPUT_DIR, "*.h5")))
    if not file_list:
        print(f"   Error: No .h5 files found in '{INPUT_DIR}'.", file=sys.stderr)
        sys.exit(1)

    # Determine the number of files to process
    max_files_to_process = None
    if args.n_files.lower() == 'all':
        max_files_to_process = None     # A value of None will signify no limit
    else:
        try:
            max_files_to_process = int(args.n_files)
            if max_files_to_process <= 0:
                print(f"   Error: Number of files must be a positive integer, not '{args.n_files}'.", file=sys.stderr)
                sys.exit(1)
        except ValueError:
            print(f"   Error: Invalid value for n_files. Expected a number or 'All', but got '{args.n_files}'.", file=sys.stderr)
            sys.exit(1)

    files_to_process = file_list[:max_files_to_process] if max_files_to_process is not None else file_list

    # Print configuration summary
    print("\n--- Analysis Configuration ---")
    print(f"Run Number      : {args.run_number}")
    print(f"LDC Number      : {args.ldc_number}")
    print(f"Files to Process: {'All' if max_files_to_process is None else len(files_to_process)}")
    print("------------------------------")
    print(f"Input Directory : {INPUT_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("------------------------------\n")

    # 2. Parallel processing of files
    n_cores = os.cpu_count() - 1 if os.cpu_count() > 1 else 1
    print(f"----- Starting parallel processing on {n_cores} cores")

    # The Parallel object manages the pool of worker processes.
    # `delayed(process_file)` creates a lightweight "promise" of a function call.
    results = Parallel(n_jobs=n_cores)(delayed(process_file)(fp, CUT_NAMES) for fp in files_to_process)
    print("----- Parallel processing finished")

    # 3. Aggregate results
    print("----- Aggregating results")
    all_processed_dfs = []
    all_reco_dfs = []
    total_cut_counts = {name: 0 for name in CUT_NAMES}

    # Unpack the results (dataframes, counts dict)
    for df_file, df_soph, local_counts in results:
        if not df_file.empty and not df_soph.empty:
            all_processed_dfs.append(df_file)
            all_reco_dfs.append(df_soph)
        for cut_name, count in local_counts.items():
            total_cut_counts[cut_name] += count

    # --- DEBUGGING: Check the contents of the lists before concatenation ---
    print(f"Aggregation complete. Found {len(all_processed_dfs)} non-empty event DataFrames.")
    print(f"Aggregation complete. Found {len(all_reco_dfs)} non-empty reco hits DataFrames.")

    # 4. Output
    print("----- Saving output files")
    # Combine all processed dataframes into one
    output_filepath = os.path.join(OUTPUT_DIR, f"processed_run_{args.run_number}_ldc{args.ldc_number}.h5")
    if args.n_files.lower() != 'all':
        output_filepath = output_filepath.replace('.h5', f'_n{len(files_to_process)}.h5')

    if all_processed_dfs or all_reco_dfs:
        print(f"Opening HDF5 store for writing: {output_filepath}")
        with pd.HDFStore(output_filepath, mode='w') as store:
            if all_processed_dfs:
                final_df = pd.concat(all_processed_dfs, ignore_index=True)
                store.put('Events', final_df, format='table')
                print(f" Concatenated 'Events' DataFrame shape: {final_df.shape}")
            if all_reco_dfs:
                reco_df = pd.concat(all_reco_dfs, ignore_index=True)
                store.put('Hits', reco_df, format='table')
                print(f" Concatenated 'Hits' DataFrame shape: {reco_df.shape}")
    else:
        print("No data was processed. No output file created.")

    # Summary file
    summary_file_exists = os.path.isfile(SUMMARY_PATH)
    header = ['run_number', 'ldc', 'n_files_processed'] + CUT_NAMES
    data_row = [args.run_number, args.ldc_number, args.n_files.lower()] + [total_cut_counts[name] for name in CUT_NAMES]

    try:
        with open(SUMMARY_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            if not summary_file_exists:
                writer.writerow(header)
            writer.writerow(data_row)
        print(f"You can find the event summary in: {SUMMARY_PATH}")
    except IOError as e:
        print(f"   Error writing to summary file: {e}", file=sys.stderr)

    print("\nY ya, eso es todo, eso es todo ♥")

if __name__ == "__main__":
    main()