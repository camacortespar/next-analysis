#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script is designed to process calibration runs for the NEXT experiment, specifically 238Th data. 
It automates the workflow of reading raw HDF5 files, applying processing steps, and saving the processed data into new HDF5 files.

This script performs the following high-level steps:

    1. Input Parsing
    2. File Discovery: Automatically identifies all relevant .h5 files in the specified input directory.
    3. Data Processing: Iterates through each file, applies the necessary calibration and processing steps.
    4. Output Generation: Saves the processed data into new .h5 files in the specified output directory.

Usage:
    python process_run.py <run_number> <ldc_number> <n_files>
"""

# ===================
# ----- IMPORTS -----
# ===================

import sys
sys.path.append('/lhome/ific/c/ccortesp/Analysis')

from libs import crudo

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

# ===============================================
# ----- CONFIGURATION & ARGUMENT DEFINITION -----
# ===============================================

# ---------------------------------------
# 1. DIRECTORIES, PATHS, FILENAMES & KEYS
# ---------------------------------------
# OUTPUT FILENAME TAG
VERSION_TAG = 'Th_zemrude'

# DIRECTORIES, PATHS & FILES
DATA_DIR   = '/lustre/ific.uv.es/prj/gl/neutrinos/users/ccortesp/NEXT-100/Sophronia/Th_runs/'
OUTPUT_DIR = '/lustre/ific.uv.es/prj/gl/neutrinos/users/ccortesp/NEXT-100/Th_analysis/h5/'
# Kr files: using Zemrude results
MAP3D_FILENAME = '/lhome/ific/c/ccortesp/Analysis/NEXT-100/Th_analysis/h5/combined_15546_15557_zemrude_map.h5'
TIME_FILENAME  = '/lhome/ific/c/ccortesp/Analysis/NEXT-100/Th_analysis/h5/energy_scale_15589_15589_he.h5'
# Summary file
SUMMARY_FILENAME = 'summary_' + VERSION_TAG + '.csv'    # Choose your name
SUMMARY_PATH = os.path.join('/lhome/ific/c/ccortesp/Analysis/NEXT-100/Th_analysis/txt/', SUMMARY_FILENAME)

# KEYS
DORO_KEY = 'DST/Events'
SOPH_KEY = 'RECO/Events'

# COLUMNS TO USE
DORO_COLUMNS = ['event', 'time', 'nS1', 'nS2', 'S1h', 'S1e', 'S2e', 'DT', 'X', 'Y', 'Z']
SOPH_COLUMNS = ['event', 'time', 'npeak', 'X', 'Y', 'Z', 'Q', 'E']
FINAL_SOPH_COLUMNS = ['event', 'time', 'npeak', 'X', 'Y', 'DT', 'Z', 'Q', 'E_hit_pe']

# CUTFLOW
CUT_NAMES = ['Reconstructed', 'Z_Positive', 'S1_Cut', 'Clean_Events']

# ------------------------
# 2. PROCESSING PARAMETERS
# ------------------------
# --- Drift Velocity --- #
V_DRIFT = 0.865     # Drift velocity in [mm/μs]

# --- S1 Signal Cuts ---
# Po-like events are filtered using: S1h >= m * S1e + b
M_NOPOLIKE = 0.17
B_NOPOLIKE = -56

# --- S1e Correction ---
DT_STOP = 1372.2543          # Cathode temporal position in [μs]
CV_FIT  = [0.57, 796.53]     # Fit values for S1e correction vs DT

# --- Spurious Hits ---
# Minimum neighbors hits to define a valid cluster
N_HITS = 5
# Clusterizer configuration
CLUSTER_CONFIG = {"distance": [16., 16., 4.], "nhit": N_HITS}

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

# =============================================================================
# ----- PROCESSING -----
# =============================================================================

def process_file(filepath, cut_names):
    """
    DESCRIPCIÓN.
    """
    filename = os.path.basename(filepath)
    print(f"→ Processing file: {filename}")
    
    # Initialize counts for this specific file
    local_evt_counter = {name: 0 for name in cut_names}
    
    try:
        # ----- Load Dorothea & Sophronia ----- #
        df_doro = pd.read_hdf(filepath, key=DORO_KEY).loc[:, DORO_COLUMNS]      # Keep only relevant columns
        df_soph = pd.read_hdf(filepath, key=SOPH_KEY).loc[:, SOPH_COLUMNS]      # Keep only relevant columns
        df_soph.rename(columns={'Z': 'DT'}, inplace=True)                       # Rename Z to DT for consistency
        df_soph['Z'] = df_soph['DT'] * V_DRIFT                                  # Compute real Z position: using the drift velocity
        reco_ids = df_soph['event'].unique()
        local_evt_counter[cut_names[0]] = len(reco_ids)

         # ----- Removing Z <= 0 ----- #
        events_with_negative_z_hits = df_soph.loc[df_soph['Z'] < 0, 'event'].unique()
        events_with_positive_z_hits = np.setdiff1d(reco_ids, events_with_negative_z_hits)
        df_doro, df_soph = bf.apply_cut_and_update(df_doro, df_soph, event_ids=events_with_positive_z_hits)
        local_evt_counter[cut_names[1]] = df_soph['event'].nunique()

        # ----- Energy Correction (Kr) ----- #
        corr3d = get_corr3d(MAP3D_FILENAME)
        corrt  = get_corrt(TIME_FILENAME)
        df_soph['E_corr'] = df_soph['E'] * corr3d(df_soph['DT'], df_soph['X'], df_soph['Y']) * corrt(df_soph['time'])
        # NaN or negative energy to 0: hit-level
        df_soph['E_corr'] = np.where(pd.notna(df_soph['E_corr']) & (df_soph['E_corr'] > 0), df_soph['E_corr'], 0)

        # ----- S1e Cut & Correction ----- #
        # nS1 <= 1 (NO-Polike)
        s1_mask = (df_doro['nS1'] == 0) | ((df_doro['nS1'] == 1) & (df_doro['S1h'] >= M_NOPOLIKE * df_doro['S1e'] + B_NOPOLIKE))
        df_doro, df_soph = bf.apply_cut_and_update(df_doro, df_soph, cut_mask=s1_mask, df_for_mask=df_doro)
        local_evt_counter[cut_names[2]] = df_soph['event'].nunique()
        # S1e Correction
        df_doro = crudo.correct_S1e(df_doro, CV_FIT, DT_STOP, output_column='S1e_corr')     # Based on alpha analysis

         # ----- Deal with Spurious Hits ----- #
        df_clean_hits = bf.deal_spurious_hits(df_soph, cluster_config=CLUSTER_CONFIG)
        non_isolated_ids = df_clean_hits['event'].unique()
        df_doro, df_soph = bf.apply_cut_and_update(df_doro, df_soph, event_ids=non_isolated_ids)
        local_evt_counter[cut_names[3]] = df_soph['event'].nunique()
                
        # ----- Data @ Event/Peak-Level ----- #
        # First, store original event size in Dorothea dataframe
        original_event_size_df = df_soph.groupby('event').size().rename('n_hits_original').reset_index()
        df_doro = df_doro.merge(original_event_size_df, on='event', how='left')
        # Now, store just the relevant columns in final Sophronia dataframe
        df_soph_final = df_clean_hits.loc[:, FINAL_SOPH_COLUMNS].copy()
        # Finally, aggregate to event-peak level
        # Dorothea event-level info
        doro_info_df = df_doro.groupby('event').agg(
                                                        time = ('time', 'mean'),
                                                        nS1  = ('nS1', 'first'),
                                                        nS2  = ('nS2', 'first'),
                                                        S1e_max = ('S1e', 'max'),
                                                        S1e_corr_max = ('S1e_corr', 'max'),
                                                        n_hits_original = ('n_hits_original', 'first')
        )
        # ----- Sophronia Info ----- #
        # Event-level
        soph_event_info_df = df_soph.groupby('event').agg(
                                                                X_bary = ('X', lambda x: bf.weighted_avg(x, df_soph.loc[x.index, 'E_hit_pe'])),
                                                                Y_bary = ('Y', lambda y: bf.weighted_avg(y, df_soph.loc[y.index, 'E_hit_pe'])),
                                                                Z_bary = ('Z', lambda z: bf.weighted_avg(z, df_soph.loc[z.index, 'E_hit_pe'])),
                                                                Z_min = ('Z', 'min'),
                                                                Z_max = ('Z', 'max'),
                                                                R_max = ('X', lambda g: bf.R_max_func(df_soph.loc[g.index])),
                                                                E_evt_pe = ('E_hit_pe', 'sum'),
                                                                n_hits = ('event', 'size')
        )
        # Peak-level
        soph_peak_info_df = df_soph.groupby(['event', 'npeak']).agg(
                                                                        E_peak_pe = ('E_hit_pe', 'sum'),
                                                                        n_hits_peak = ('event', 'size')
        ).reset_index()

        # ----- Merge Final DataFrame ----- #
        df_event_peak = pd.merge(soph_peak_info_df, soph_event_info_df, on='event', how='left')
        df_event_peak = pd.merge(df_event_peak, doro_info_df, on='event', how='left')
            
    except Exception as e:
        print(f"   Failed to process file {filename}. Error: {e}", file=sys.stderr)
        # Return a dictionary of zeros on failure to not affect the final sum
        return pd.DataFrame(), pd.DataFrame(), {name: 0 for name in cut_names}

    return df_event_peak, df_soph, local_evt_counter

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
    output_filepath = os.path.join(OUTPUT_DIR, f"processed_run_{args.run_number}_ldc{args.ldc_number}_{VERSION_TAG}.h5")
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