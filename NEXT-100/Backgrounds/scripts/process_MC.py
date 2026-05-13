#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script automates the pre-analysis workflow for simulation files in the NEXT experiment.
It is mainly intended for processing background MC production, but it also can process bb2nu and bb0nu MC files.
The script performs the following high-level steps:

1. Input Parsing: Reads command-line arguments to determine the processing options.
2. File Discovery: Identifies all relevant HDF5 files in the specified input directory.
3. Data Processing: Processes each file, applying corrections, cuts, and clustering. See function `process_file` for details.
4. Output Generation: Saves processed information into new HDF5 files in the specified output directory.
5. Summary Update: Optionally updates a summary CSV file with isotope-level statistics.

Usage:
    python process_MC.py <process_type> <isotope>

Notes:
    In difference with data processing, this script only stores at event/peak level, not at hit level.
"""

# ============================================================================
# ----- IMPORTS -----
# ============================================================================

import sys
sys.path.append('/lhome/ific/c/ccortesp/Analysis')

from libs import crudo

import argparse
from datetime import datetime
import glob
from joblib import delayed, Parallel
import numpy as np
import os
import pandas as pd
from pathlib import Path
from typing import Callable, List, Tuple

# =============================================================================
# ----- CONFIGURATION & ARGUMENT DEFINITION -----
# =============================================================================
DATE = datetime.now().strftime('%d%m%Y')    # It helps to keep track of when the files were processed

# DIRECTORIES
OUTPUT_DIR  = '/lustre/ific.uv.es/prj/gl/neutrinos/users/ccortesp/NEXT-100/Backgrounds/h5/mc/'
SUMMARY_DIR = '/lhome/ific/c/ccortesp/Analysis/NEXT-100/Backgrounds/txt/summaries/'

# KEYS
MC_CONFIG_KEY = '/MC/configuration'
TRUE_INFO_KEY = '/MC/particles'
DORO_KEY = '/DST/Events'
SOPH_KEY = '/RECO/Events'

# COLUMNS TO USE
DORO_COLUMNS = ['event', 'time', 'nS1', 'nS2', 'S1w', 'S1h', 'S1e', 'S1t', 'S2w', 'S2h', 'S2e', 'S2q', 'S2t', 'DT', 'X', 'Y', 'Z']
SOPH_COLUMNS = ['event', 'time', 'npeak', 'Xpeak','X', 'Y', 'Z', 'Q', 'Ec']
FINAL_SOPH_COLUMNS = ['event', 'time', 'npeak', 'X', 'Y', 'Z', 'E_hit_mev', 'cluster']

EVENT_LEVEL_COLS = ['nS1', 'nS2', 'isotope', 'volume', 'double_e', 'old_n_hits']

# CUTFLOW
CUT_NAMES = ['Generated', 'Interacting', 'Saved', 'Reconstructed', 'Strong_S2', 'S1_Cut', 'Clean_Events']

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
DT_CATH = 1350               # Cathode temporal position in [μs]
CV_FIT  = [0.57, 796.53]     # Fit values from S1e vs DT plot

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
    parser = argparse.ArgumentParser(description="Starting the processing of a MC production...")
    
    # ----- Positional (Required) Arguments ----- #
    parser.add_argument("process_type",
                        type=str,
                        choices=['radiogenics_hpr', 'radiogenics_lpr', 'calibration_hpr', 'calibration_lpr',
                                 'bb2nu_hpr', 'bb2nu_lpr', 'bb0nu_hpr', 'bb0nu_lpr'],
                        help="The type of MC process to analyze (e.g., 'radiogenics_hpr', 'bb2nu_hpr').")

    parser.add_argument("isotope",
                        type=str,
                        help="The isotope to process (e.g., 'Bi214', 'Co60', 'K40', 'Tl208', 'Tl214' or 'Xe136').")

    # If no arguments are provided, print the help message and exit
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    # Parse the arguments from the command line
    args = parser.parse_args()
    
    return args

# # =============================================================================
# # ----- PROCESSING -----
# # =============================================================================

def process_mc_file(filepath, isotope, cut_names=CUT_NAMES):
    """
    Processes a single file containing NEXT-100 MC reconstructed data, applying a series of cuts, 
    corrections, and clustering to prepare the data for further analysis.

    Parameters:
    -----------
        filepath : str
            Path to the input HDF5 file containing Nexus, Dorothea and Sophronia information.
        isotope : str
            The isotope being processed (e.g., 'Bi214', 'Co60', etc.).
        cut_names : list of str, optional
            List of cut names to track the number of events passing each cut. Defaults to CUT_NAMES.

    Returns:
    --------
        df_event_peak : pandas.DataFrame
            Dataframe containing the processed data aggregated to the event-peak level.
        local_evt_counter : dict
            Dictionary containing the count of events passing each cut.

    Notes:
    ------
    - The function performs the following steps:
        1. Loads Dorothea and Sophronia data from the input file.
        2. Extracts Monte Carlo information such as isotope, volume, and double-electron track flags.
        3. Applies data cleaning, including removing weak S2 peaks and spurious hits.
        4. Applies energy corrections and S1e cuts based on predefined parameters.
        5. Aggregates the data to event-peak level for further analysis.
    - If an error occurs during processing, the function returns an empty dataframe and a dictionary of zeros 
      to ensure robustness.
    """
    filepath = Path(filepath)
    # print(f"→ Processing file: {filepath}")

    # Initialize counts for this specific file
    local_evt_counter = {cut: 0 for cut in cut_names}
    
    try:
        # ----- Load Dataframes ----- #
        df_nexus = pd.read_hdf(filepath, key=MC_CONFIG_KEY)
        df_true  = pd.read_hdf(filepath, key=TRUE_INFO_KEY)
        df_doro  = pd.read_hdf(filepath, key=DORO_KEY).loc[:, DORO_COLUMNS]
        df_soph  = pd.read_hdf(filepath, key=SOPH_KEY).loc[:, SOPH_COLUMNS]
    
        # Counters
        mc_config = df_nexus.set_index('param_key')['param_value']
        local_evt_counter[cut_names[0]] = int(mc_config.get('num_events', 0))
        local_evt_counter[cut_names[1]] = int(mc_config.get('interacting_events', 0))
        local_evt_counter[cut_names[2]] = int(mc_config.get('saved_events', 0))
        local_evt_counter[cut_names[3]] = df_soph['event'].nunique()

        # ----- Monte Carlo Information ----- #
        # Isotope and volume
        volume  = filepath.parts[-3]
        df_doro['volume']   = volume
        df_doro['isotope']  = isotope
        df_doro['double_e'] = True
        if isotope != 'Xe136':
            pair_prod_evt_ids = df_true[(df_true['creator_proc'] == 'conv') & (df_true['initial_volume'] == 'ACTIVE')].event_id.unique()
            df_doro['double_e'] = df_doro['event'].isin(pair_prod_evt_ids)
    
        # ----- Data Cleaning ----- #
        # Remove weak S2 peaks in Dorothea and Sophronia
        df_doro = df_doro[df_doro['S2q'] >= 0].copy()
        df_soph = df_soph[df_soph['Xpeak'] >= -5000].copy()
        events_with_strong_S2_in_soph = df_soph['event'].unique()
        df_doro, df_soph = crudo.dm.apply_cut_and_update(df_doro, df_soph, event_ids=events_with_strong_S2_in_soph)
        local_evt_counter[cut_names[4]] = df_soph['event'].nunique()

        # ----- Energy Correction ----- #
        # Just set hits with NaN or negative energy to 0
        df_soph['Ec'] = np.where(pd.notna(df_soph['Ec']) & (df_soph['Ec'] > 0), df_soph['Ec'], 0)

        # ----- S1e Cut & Correction ----- #
        # nS1 <= 1 (NO-Polike)
        s1_mask = (df_doro['nS1'] == 0) | ((df_doro['nS1'] == 1) & (df_doro['S1h'] >= M_NOPOLIKE * df_doro['S1e'] + B_NOPOLIKE))
        df_doro, df_soph = crudo.dm.apply_cut_and_update(df_doro, df_soph, cut_mask=s1_mask, df_for_mask=df_doro)
        local_evt_counter[cut_names[5]] = df_soph['event'].nunique()
        # S1e Correction
        df_doro = crudo.ef.correct_S1e(df_doro, CV_FIT, DT_CATH, output_column='S1e_corr')     # Based on alpha analysis

        # ----- Deal with Spurious Hits ----- #
        df_clust_soph = CLUSTER_FUNCTION(df_soph)    # Applying hits_clusterizer
        df_clean_soph = crudo.tf.deal_spurious_hits(df_clust_soph, energy_column='Ec', output_column='E_hit_mev')
        clean_evt_ids = df_clean_soph['event'].unique()
        df_doro, df_soph = crudo.dm.apply_cut_and_update(df_doro, df_soph, event_ids=clean_evt_ids) 
        local_evt_counter[cut_names[6]] = df_soph['event'].nunique()

        # ----- Data @ Event/Peak-Level ----- #
        # First, store original event size from Sophronia into Dorothea dataframe
        original_event_size_df = df_soph.groupby('event').size().rename('old_n_hits').reset_index()
        df_doro = df_doro.merge(original_event_size_df, on='event', how='left')
        # Now, store just the relevant columns in final Sophronia dataframe
        df_soph_final = df_clean_soph.loc[:, FINAL_SOPH_COLUMNS].copy()
        # Finally, aggregate to event-peak level
        df_event_peak = crudo.dm.aggregate_to_event_peak_level(df_doro, df_soph_final, event_level_cols=EVENT_LEVEL_COLS, energy_column='E_hit_mev')

    except Exception as e:
        print(f"   Failed to process file {filepath}. Error: {e}", file=sys.stderr)
        # Return a dictionary of zeros on failure to not affect the final sum
        return pd.DataFrame(), {name: 0 for name in cut_names}

    return df_event_peak, local_evt_counter

# =============================================================================
# ----- MAIN -----
# =============================================================================

def main():
    """
    Música maestro! This is the main function that orchestrates the processing
    """
    # 1. --- PARSE COMMAND-LINE ARGUMENTS
    #        SET UP PATHS TO PROCESS
    args = parse_arguments()
    PROCESS_TYPE = args.process_type
    ISOTOPE = args.isotope
    print("\n----- Processing Configuration -----")
    print(f"Date: {DATE}")
    print(f"Process Type: {PROCESS_TYPE}")
    print(f"Isotopes: {ISOTOPE}")
    print("------------------------------------")

    # Outputs
    output_filename = 'processed_mc_' + PROCESS_TYPE + '_'
    if ISOTOPE != 'Xe136':  output_filename += ISOTOPE + '_'
    output_filename += DATE + '.h5'
    OUTPUT_FILEPATH = os.path.join(OUTPUT_DIR, output_filename)
    
    summary_filename = 'summary_' + PROCESS_TYPE + '_processed.csv'
    SUMMARY_PATH = os.path.join(SUMMARY_DIR, summary_filename)

    # Files to process
    MC_DIR = '/lustre/ific.uv.es/prj/gl/neutrinos/NEXT/MC/NEXT100/'
    MC_PATHS = []

    if PROCESS_TYPE == 'radiogenics_hpr': MC_DIR += 'Radiogenics/HPR/IC_v2.3.1/NEXUS_v7_10_01/'
    if PROCESS_TYPE == 'radiogenics_lpr': MC_DIR += 'Radiogenics/LPR/IC_v2.3.1/NEXUS_v7_09_00/'
    if PROCESS_TYPE == 'calibration_lpr': MC_DIR += 'Calibration/LPR/IC_v2.3.1/NEXUS_v7_09_00/'
    if PROCESS_TYPE == 'bb2nu_hpr':       MC_DIR += 'bb2nu/HPR/IC_v2.3.1/NEXUS_v7_10_01/bb2nu/'
    if PROCESS_TYPE == 'bb0nu_hpr':       MC_DIR += 'bb0nu/HPR/IC_v2.3.1/NEXUS_v7_10_01/bb0nu/'

    if ISOTOPE != 'Xe136':  MC_DIR += ISOTOPE + '/'
    # Get all subfolders in MC_DIR
    VOLUMES = [folder for folder in os.listdir(MC_DIR) if os.path.isdir(os.path.join(MC_DIR, folder))]
    for volume in sorted(VOLUMES):
        # Get all sophronia files in the volume subfolder
        sophronia_files = sorted(glob.glob(os.path.join(MC_DIR, volume, 'sophronia', '*.sophronia.h5')))
        # print(f"  Found {len(sophronia_files)} sophronia files in {volume}")
        if sophronia_files:
            MC_PATHS.extend(sophronia_files)
        else:
            print(f"  {ISOTOPE}/{volume}: No Sophronia files found")
    MC_PATHS = sorted(MC_PATHS)
    print(f"Total files to process: {len(MC_PATHS)}")
    print("------------------------------------")

    # 2. --- PARALLEL PROCESSING OF FILES
    n_cores = os.cpu_count() - 1 if os.cpu_count() > 1 else 1
    print(f"\n----- Starting parallel processing on {n_cores} cores")

    results = Parallel(n_jobs=n_cores)(delayed(process_mc_file)(h5_path, isotope=ISOTOPE) for h5_path in MC_PATHS)
    print("----- Parallel processing finished")

    # 2. --- COMBINE RESULTS FROM ALL FILES
    print("\n----- Aggregating results")
    all_processed_dfs = []
    total_cut_counts = {name: 0 for name in CUT_NAMES} 

    # Unpack the results (dataframes, counts dict)
    valid_results = [r for r in results if r is not None]       # Filter out None results
    for df_file, local_counts in valid_results:
        # Append the processed dataframe if it's not empty
        if not df_file.empty:
            all_processed_dfs.append(df_file)
        # Aggregate the counts for each cut
        for cut_name, count in local_counts.items():
            total_cut_counts[cut_name] += count

    # Concatenate dataframes
    final_mc_df = pd.concat(all_processed_dfs, ignore_index=True) if all_processed_dfs else pd.DataFrame()
    # DEBUGGING: Check the contents of the lists before concatenation
    print(f"Event dataframe shape: {final_mc_df.shape}")

    # 4. --- OUTPUT
    print("\n----- Saving output files")
    # npeak column in run_event_df is uint64, convert to int64
    for col in final_mc_df.select_dtypes(include=['uint64']).columns:
        final_mc_df[col] = final_mc_df[col].astype('int64')

    # Processed file
    print(f"Opening HDF5 store for writing: {OUTPUT_FILEPATH}")
    try:
        with pd.HDFStore(OUTPUT_FILEPATH, mode='w') as store:
            if not final_mc_df.empty:
                store.put('Events', final_mc_df, format='table', data_columns=True)
        print("HDF5 saving complete.")
    except Exception as e:
        print(f"   Error writing to HDF5 file: {e}", file=sys.stderr)

    # Summary file
    print("\n----- Updating summary file")
    summary_data = {'Isotope': [args.isotope]}
    # Add the cut counts
    for name in total_cut_counts.keys():
        summary_data[name] = [total_cut_counts.get(name, 0)]
    summary_row_df = pd.DataFrame(summary_data)
    # Append to the CSV file
    print(f"Appending summary to: {SUMMARY_PATH}")
    try:
        summary_row_df.to_csv(
                                SUMMARY_PATH,
                                mode='a',
                                header=not os.path.exists(SUMMARY_PATH),
                                index='Isotope',
                            )
        print("Summary file updated.")
    except IOError as e:
        print(f"   Error writing to summary file: {e}", file=sys.stderr)

    print('\nY ya, eso es todo, eso es todo ♥')

if __name__ == "__main__":
    main()