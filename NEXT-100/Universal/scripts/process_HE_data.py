#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script automates the pre-analysis workflow for high-energy data (238Th) in the NEXT experiment.
It processes reconstructed data to prepare it for further analysis. The script performs the following high-level steps:

1. Input Parsing: Reads command-line arguments to determine the run number and processing options.
2. File Discovery: Identifies all relevant HDF5 files in the specified input directory.
3. Data Processing: Processes each file, applying corrections, cuts, and clustering. See function `process_file` for details.
4. Output Generation: Saves processed data into new HDF5 files in the specified output directory.
5. Summary Update: Optionally updates a summary CSV file with run-level statistics.

Usage:
    python process_HE_data.py <run_number> <ldc_number> <n_files> <kr_city> [--events-only]

Options:
    --events-only: If specified, only event-level data is saved, skipping hit-level data.

Note:
    Yes, this script is a mirror of the one used for low-background data processing, but adapted for the specific needs of high-energy calibration runs.
"""

# ============================================================================
# ----- IMPORTS -----
# ============================================================================

import sys
sys.path.append('/lhome/ific/c/ccortesp/Analysis')

from libs import crudo

import argparse
import csv
import glob
from invisible_cities.core.core_functions import in_range
from invisible_cities.reco.corrections import apply_all_correction, read_maps
from invisible_cities.types.symbols import NormMethod
from invisible_cities.types.symbols import NormStrategy
from joblib import delayed, Parallel 
import numpy as np
import os
import pandas as pd
from typing import Callable, List, Tuple

# =============================================================================
# ----- CONFIGURATION & ARGUMENT DEFINITION -----
# =============================================================================
# OUTPUT FILENAME TAG
VERSION_TAG = 'th_zemrude'

# DIRECTORIES, PATHS & FILES
# DATA_DIR   = '/lustre/ific.uv.es/prj/gl/neutrinos/users/ccortesp/NEXT-100/Sophronia/Th_runs/'
DATA_DIR   = '/lhome/ific/c/ccortesp/Analysis/NEXT-100/Th_analysis/h5/runs/'
ICAROS_DIR = '/lhome/ific/c/ccortesp/Analysis/NEXT-100/Th_analysis/h5/'
OUTPUT_DIR = '/lustre/ific.uv.es/prj/gl/neutrinos/users/ccortesp/NEXT-100/Th_analysis/h5/runs/'

SUMMARY_FILENAME = 'summary_' + VERSION_TAG + '_processed.csv'
SUMMARY_PATH = os.path.join('/lhome/ific/c/ccortesp/Analysis/NEXT-100/Th_analysis/txt/summaries/', SUMMARY_FILENAME)

# KEYS
DORO_KEY = 'DST/Events'
SOPH_KEY = 'RECO/Events'

# COLUMNS TO USE
DORO_COLUMNS = ['event', 'time', 'nS1', 'nS2', 'S1w', 'S1h', 'S1e', 'S1t', 'S2w', 'S2h', 'S2e', 'S2q', 'S2t', 'DT', 'X', 'Y', 'Z']
SOPH_COLUMNS = ['event', 'time', 'npeak', 'X', 'Y', 'Z', 'Q', 'E']
FINAL_SOPH_COLUMNS = ['event', 'time', 'npeak', 'X', 'Y', 'DT', 'Z', 'E_hit_mev', 'cluster']

EVENT_LEVEL_COLS = ['nS1', 'nS2', 'old_n_hits']

# CUTFLOW
CUT_NAMES = ['Reconstructed', 'Z_Positive', 'S1_Cut', 'Clean_Events']

# ---------------------
# PROCESSING PARAMETERS
# ---------------------
V_DRIFT = 0.865     # Drift velocity in [mm/μs]

# --- S1 Signal Cuts ---
# Po-like events are filtered using: S1h >= m * S1e + b
M_NOPOLIKE = 0.17
B_NOPOLIKE = -56

# --- S1e Correction ---
# Values from Radon analysis: S1e = m * DT + b
DT_CATH = 1350              # Cathode temporal position in [μs]
CV_FIT  = [0.57, 796.53]    # Fit values from S1e vs DT plot

# --- Hits Clusterizer ---
CLUSTERING_PARAMS = dict(eps = 1.8, min_samples = 5, scale_xy = 15.55, scale_z = 4.0)
CLUSTER_FUNCTION = crudo.tf.hits_clusterizer(CLUSTERING_PARAMS)

def parse_arguments():
    """
    Parses command-line arguments.
    
    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Starting the processing of a Thorium calibrarion run...")
    
    # ----- Positional (Required) Arguments ----- #
    parser.add_argument("run_number",
                        type=int,
                        help="The run number to process (e.g., 15589).")
    
    parser.add_argument("ldc_number",
                        type=int,
                        choices=range(1, 8),
                        metavar="ldc_number[1-7]",
                        help="The LDC number, an integer from 1 to 7.")

    parser.add_argument("n_files",
                        type=str,
                        help="Number of files to process. Use an integer (e.g., 10) or 'all' to process all available files.")

    parser.add_argument("kr_city",
                        type=str,
                        help="The Krypton map city to use for energy correction ('icaros', 'zemrude').")

    parser.add_argument("--events-only",
                        action='store_true',
                        help="If specified, only save the event-level data, not the hit-level data.")

    # If no arguments are provided, print the help message and exit
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    # Parse the arguments from the command line
    args = parser.parse_args()
    
    return args

# =============================================================================
# ----- PROCESSING -----
# =============================================================================

def process_file(filepath, kr_path, kr_city, cut_names=CUT_NAMES):
    """
    Processes a single file containing NEXT-100 data, applying a series of cuts and corrections 
    to prepare the data for further analysis. 
    Parameters:
    -----------
        filepath : str
            Path to the input HDF5 file containing Dorothea and Sophronia data.
        kr_path : str
            Path to the Krypton map file used for energy corrections.
        kr_city : str
            The city corresponding to the Krypton map (e.g., 'icaros', 'zemrude') to apply the correct energy correction.   
        cut_names : list of str, optional
            List of cut names to track the number of events passing each cut. Defaults to CUT_NAMES.
    Returns:
    --------
        df_event_peak : pandas.DataFrame
            Dataframe containing the processed data aggregated to the event-peak level.
        df_soph_final : pandas.DataFrame
            Dataframe containing the final hits-level data with relevant columns.
        local_evt_counter : dict
            Dictionary containing the count of events passing each cut.
    Notes:
    ------
    - The function performs the following steps:
        1. Loads Dorothea and Sophronia data from the input file.
        2. Computes the Z position using drift velocity and removes events with Z <= 0.
        3. Applies energy corrections using a Krypton map.
        4. Applies S1e cuts and corrections based on alpha analysis.
        5. Deals with spurious hits using a clustering function.
        6. Aggregates the data to event-peak level for further analysis.
    - If an error occurs during processing, the function returns empty dataframes and a dictionary of zeros 
      to ensure robustness.
    """
    filename = os.path.basename(filepath)
    print(f"→ Processing file: {filename}")
    
    # Initialize counts for this specific file
    local_evt_counter = {cut: 0 for cut in cut_names}
    
    try:
        # ----- Load Dorothea & Sophronia ----- #
        df_doro = pd.read_hdf(filepath, key=DORO_KEY).loc[:, DORO_COLUMNS]
        df_soph = pd.read_hdf(filepath, key=SOPH_KEY).loc[:, SOPH_COLUMNS]
        # Compute Z in [mm]
        df_soph.rename(columns={'Z': 'DT'}, inplace=True)                       # Rename Z to DT for consistency
        df_soph['Z'] = df_soph['DT'] * V_DRIFT                                  # Compute real Z position: using the drift velocity
        reco_ids = df_soph['event'].unique()
        local_evt_counter[cut_names[0]] = len(reco_ids)

        # ----- Removing Z <= 0 ----- #
        events_with_negative_z_hits = df_soph.loc[df_soph['Z'] < 0, 'event'].unique()
        events_with_positive_z_hits = np.setdiff1d(reco_ids, events_with_negative_z_hits)
        df_doro, df_soph = crudo.dm.apply_cut_and_update(df_doro, df_soph, event_ids=events_with_positive_z_hits)
        local_evt_counter[cut_names[1]] = df_soph['event'].nunique()

        # ----- Energy Correction ----- #
        df_soph = crudo.ef.correct_energy_by_kr_map( df_soph
                                                   , kr_path
                                                   , norm_method=NormMethod.median_anode
                                                   , city=kr_city
                                                   , mev_units=True
                                                   , output_col='Ec' )

        # ----- S1e Cut & Correction ----- #
        # nS1 <= 1 (NO-Polike)
        s1_mask = (df_doro['nS1'] == 0) | ((df_doro['nS1'] == 1) & (df_doro['S1h'] >= M_NOPOLIKE * df_doro['S1e'] + B_NOPOLIKE))
        df_doro, df_soph = crudo.dm.apply_cut_and_update(df_doro, df_soph, cut_mask=s1_mask, df_for_mask=df_doro)
        local_evt_counter[cut_names[2]] = df_soph['event'].nunique()
        # S1e Correction
        df_doro = crudo.ef.correct_S1e(df_doro, CV_FIT, DT_CATH, output_column='S1e_corr')     # Based on alpha analysis

        # ----- Deal with Spurious Hits ----- #
        df_clust_soph = CLUSTER_FUNCTION(df_soph)
        df_clean_soph = crudo.tf.deal_spurious_hits(df_clust_soph, energy_column='Ec', output_column='E_hit_mev')
        clean_evt_ids = df_clean_soph['event'].unique()
        df_doro, df_soph = crudo.dm.apply_cut_and_update(df_doro, df_soph, event_ids=clean_evt_ids) 
        local_evt_counter[cut_names[3]] = df_soph['event'].nunique()

        # ----- Data @ Event/Peak-Level ----- #
        # First, store original event size from Sophronia into Dorothea dataframe
        original_event_size_df = df_soph.groupby('event').size().rename('old_n_hits').reset_index()
        df_doro = df_doro.merge(original_event_size_df, on='event', how='left')
        # Now, store just the relevant columns in final Sophronia dataframe
        df_soph_final = df_clean_soph.loc[:, FINAL_SOPH_COLUMNS].copy()
        # Finally, aggregate to event-peak level
        df_event_peak = crudo.dm.aggregate_to_event_peak_level(df_doro, df_soph_final, event_level_cols=EVENT_LEVEL_COLS, energy_column='E_hit_mev')
        
    except Exception as e:
        print(f"   Failed to process file {filename}. Error: {e}", file=sys.stderr)
        # Return a dictionary of zeros on failure to not affect the final sum
        return pd.DataFrame(), pd.DataFrame(), {name: 0 for name in cut_names}

    return df_event_peak, df_soph_final, local_evt_counter

# =============================================================================
# ----- MAIN -----
# =============================================================================

def main():
    """
    Música maestro! This is the main function that orchestrates the processing
    """
    # 1. --- PARSE COMMAND-LINE ARGUMENTS
    #        LOAD CORRESPONDING KRYPTON MAP FOR ENERGY CORRECTION
    #        SET UP PATHS TO PROCESS
    args = parse_arguments()

    # Construct specific input/output directories based on run and LDC
    INPUT_DIR = os.path.join(DATA_DIR, f"{args.run_number}", f"ldc{args.ldc_number}")
    if not os.path.isdir(INPUT_DIR):
        print(f"   Error: Input directory '{INPUT_DIR}' does not exist.", file=sys.stderr)
        sys.exit(1)
    file_list = sorted(glob.glob(os.path.join(INPUT_DIR, "*.h5")))
    if not file_list:
        print(f"   Error: No .h5 files found in '{INPUT_DIR}'.", file=sys.stderr)
        sys.exit(1)

    # Determine the number of files to process
    max_files_to_process = None     # None means NO limit
    if args.n_files.lower() != 'all':
        try:
            max_files_to_process = int(args.n_files)
            if max_files_to_process <= 0:
                print(f"   Error: Number of files must be a positive integer, not '{args.n_files}'.", file=sys.stderr)
                sys.exit(1)
        except ValueError:
            print(f"   Error: Invalid value for n_files. Expected a number or 'all', but got '{args.n_files}'.", file=sys.stderr)
            sys.exit(1)
    files_to_process = file_list[:max_files_to_process] if max_files_to_process is not None else file_list

    # Find the Kr map
    kr_file = next((f 
                    for f in os.listdir(ICAROS_DIR) 
                    if (f'run_{args.run_number}' in f and
                        ((args.kr_city == 'icaros' and f.endswith('map.h5')) or 
                        (args.kr_city == 'zemrude' and f.endswith('zemrude.h5'))))), None)
    if not kr_file:
        raise FileNotFoundError(f"   Error: NO Kr map file found for run {args.run_number} in {ICAROS_DIR}")
        sys.exit(1)
    KR_PATH = os.path.join(ICAROS_DIR, kr_file)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Print configuration summary
    print("\n----- Processing Configuration -----")
    print(f"Run Number      : {args.run_number}")
    print(f"LDC Number      : {args.ldc_number}")
    print(f"Files to Process: {'All' if max_files_to_process is None else len(files_to_process)}")
    print(f"Kr Map City     : {args.kr_city}")
    print(f"Kr Map File     : {kr_file}")
    print("------------------------------------")
    print(f"Input Directory : {INPUT_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("------------------------------------")

    # 2. --- PARALLEL PROCESSING OF FILES
    n_cores = os.cpu_count() - 1 if os.cpu_count() > 1 else 1
    print(f"\n----- Starting parallel processing on {n_cores} cores")

    # The Parallel object manages the pool of worker processes.
    # `delayed(process_file)` creates a lightweight "promise" of a function call.
    results = Parallel(n_jobs=n_cores)(delayed(process_file)(fp, KR_PATH, kr_city=args.kr_city) for fp in files_to_process)
    print("----- Parallel processing finished")

    # 3. --- COMBINE RESULTS FROM ALL FILES
    print("\n----- Aggregating results")
    all_processed_dfs = []
    all_sophronia_dfs = []
    total_cut_counts = {name: 0 for name in CUT_NAMES}

    # Unpack the results (dataframes, counts dict)
    for df_file, df_soph, local_counts in results:
        if not df_file.empty and not df_soph.empty:
            all_processed_dfs.append(df_file)
            # Only keep hit-level data if --events-only flag is not set
            if not args.events_only and not df_soph.empty:
                all_sophronia_dfs.append(df_soph)
        for cut_name, count in local_counts.items():
            total_cut_counts[cut_name] += count

    # Concatenate dataframes
    if all_processed_dfs:
        run_event_df = pd.concat(all_processed_dfs, ignore_index=True)
    else:
        run_event_df = pd.DataFrame()
    # DEBUGGING: Check the contents of the lists before concatenation
    print(f"Event dataframe shape: {run_event_df.shape}")

    if not args.events_only:
        if all_sophronia_dfs:
            run_sophronia_df = pd.concat(all_sophronia_dfs, ignore_index=True)
        else:
            run_sophronia_df = pd.DataFrame()
        print(f"Hits dataframe shape: {run_sophronia_df.shape}")
    else:
        print("Hit-level data will NOT be saved due to --events-only flag. Skipping hit-level dataframe concatenation.")

    # 4. --- OUTPUT
    print("\n----- Saving output files")
    # npeak column in run_event_df is uint64, convert to int64
    for col in run_event_df.select_dtypes(include=['uint64']).columns:
        run_event_df[col] = run_event_df[col].astype('int64')
    
    # Combine all processed dataframes into one
    output_filename = f"processed_run_{args.run_number}_ldc{args.ldc_number}_{VERSION_TAG}"
    if args.n_files.lower() != 'all':
        output_filename += f"_n{len(files_to_process)}"
    if args.events_only:
        output_filename += "_events"
    output_filename += ".h5"
    output_filepath = os.path.join(OUTPUT_DIR, output_filename)

    print(f"Opening HDF5 store for writing: {output_filepath}")
    try:
        with pd.HDFStore(output_filepath, mode='w') as store:
            if not run_event_df.empty:
                store.put('Events', run_event_df, format='table', data_columns=True)
            if not args.events_only and not run_sophronia_df.empty:
                store.put('Hits', run_sophronia_df, format='table', data_columns=True)
        print("HDF5 saving complete.")
    except Exception as e:
        print(f"   Error writing to HDF5 file: {e}", file=sys.stderr)

    # Summary file: just when --events-only is set
    if args.events_only:
        print("\n----- Updating summary file")
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
    else:
        print("\n----- Skipping summary file update since --events-only flag is not set.")
        
    print("\nY ya, eso es todo, eso es todo ♥")

if __name__ == "__main__":
    main()