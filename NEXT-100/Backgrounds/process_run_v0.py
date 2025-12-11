#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script performs the pre-analysis workflow for low background data (trigger 2).
The process involves:

- Loading of reconstructed data.
- Correction of energy using Kr maps and HE scale-
- Data cleaning to mitigate artifacts or anomalies arising from the NEXT reconstruction process.
- Application of cuts on S1 and S2 signals. 
- Differentiation and isolation of alpha and electron populations.
- Cleaning of isolated clusters of hits.
- Generation of dataframes organized by analysis level (hit and event) and particle type.
- Application of selection criteria to separate populations into detector sub-volumes of interest.
- Storage of the processed and categorized information into a single output file per run for subsequent analysis phases.

The script automates the workflow of reading raw HDF5 files, applying processing steps, and saving the processed data into new HDF5 files,
with the following high-level steps:

    1. Input Parsing
    2. File Discovery: Automatically identifies all relevant .h5 files in the specified input directory.
    3. Data Processing: Iterates through each file, applies the necessary processing steps.
    4. Output Generation: Saves the processed data into new .h5 files in the specified output directory.

Usage:
    python process_run.py <run_number>
"""

# ============================================================================
# ----- IMPORTS -----
# ============================================================================

import sys
sys.path.append('/lhome/ific/c/ccortesp/Analysis/')

from libs import bckg_functions as bf
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

# OUTPUT FILENAME TAG
# This tag will be added to the output HDF5 filename to version the analysis.
# Avoids overwriting previous results and helps keep track of different cut configurations.
# Example tags: 'other', 'p2_nhit5', 'nhit5_Qthres7' (this is basically p1_nhit5)
VERSION_TAG = 'p2_nhit5_HEscale'

# DIRECTORIES, PATHS & FILES
DATA_DIR   = '/lustre/ific.uv.es/prj/gl/neutrinos/users/ccortesp/NEXT-100/Sophronia/Low_background/'
ICAROS_DIR = '/lustre/ific.uv.es/prj/gl/neutrinos/users/ccortesp/NEXT-100/Icaros/Low_background/'
OUTPUT_DIR = '/lustre/ific.uv.es/prj/gl/neutrinos/users/ccortesp/NEXT-100/Backgrounds/h5/runs/'

RUNS_INFO_PATH = os.path.join('/lhome/ific/c/ccortesp/Analysis/NEXT-100/Backgrounds/utilities/runs_information.csv')

SUMMARY_FILENAME = 'summary_' + VERSION_TAG +'.csv'     # Choose your name
SUMMARY_PATH = os.path.join('/lhome/ific/c/ccortesp/Analysis/NEXT-100/Backgrounds/txt/', SUMMARY_FILENAME)

# KEYS
DORO_KEY = 'DST/Events'
SOPH_KEY = 'RECO/Events'
EVT_KEY  = 'PROCESSED/Events'

# COLUMNS TO USE
DORO_COLUMNS = ['event', 'time', 'nS1', 'nS2', 'S1h', 'S1e', 'S2e', 'DT', 'X', 'Y', 'Z']        # REVISAR SI SE PUEDEN REDUCIR MAS
SOPH_COLUMNS = ['event', 'time', 'npeak', 'X', 'Y', 'Z', 'Q', 'E']

# CUTFLOWS
CUT_NAMES = ['Reconstructed', 'Clean', 'S1_Cut', 'S2_Cut', 'nElectron', 'nAlpha']

# ANALYSIS PARAMETERS
# ----------------------
V_DRIFT = 0.865     # Drift velocity in [mm/μs]

# --- S1 Signal Cuts ---
# Po-like events are filtered using: S1h >= m * S1e + b
M_NOPOLIKE = 0.17
B_NOPOLIKE = -56

# --- S1e Correction ---
DT_STOP = 1372.2543          # Cathode temporal position in [μs]
CV_FIT  = [0.57, 796.53]     # Fit values for S1e correction vs DT

# --- Alpha/Electron Separation Cut ---
# Events with total corrected energy above this threshold are classified as alphas.
ENERGY_THRESHOLD = 7.5e5       # in [PE]

# --- Electron Hit Cleaning Cut ---
# Minimum neighbors hits to define a valid cluster.
N_HITS = 5
# Minimum Q charge to activate a SiPM (P1 = 7 pe, P2 = 5 pe).
Q_THRESHOLD = 5        # in [pe]

# --- Alpha Hit Cleaning Cut ---
# In the alpha population, hits with charge below this value are removed.
Q_LIM_ALPHA = 100      # in [pe]

# --- HE Scale Correction ---
M_HE_SCALE = 4.80e-06
B_HE_SCALE = 0.0094

# --- Isolated/non-isolated Hits --- #
CLUSTER_CONFIG = {"distance": [16., 16., 4.], "nhit": N_HITS}

# --- Trigger 2 Efficiency Cut ---
# Events with event-level energy below this are removed to account for trigger efficiency.
TRG2_THRESHOLD = 0.5   # in [MeV]

# 3. DETECTOR REGIONS
# -------------------
# Geometric boundaries for event classification.
Z_LOW = 40          # in [mm]
Z_UP  = 1147        # in [mm]
R_UP  = 451.65      # in [mm]

def parse_arguments():
    """
    Parses command-line arguments.
    
    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Starting the processing of a low-background run...")
    
    # ----- Positional (Required) Arguments ----- #
    parser.add_argument("run_number",
                        type=int,
                        help="The run number to process (e.g., 15737).")

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

def process_file(filepath, kr_path, cut_names=CUT_NAMES):
    """
    DESCRIPCIÓN.
    """
    filename = os.path.basename(filepath)
    print(f"→ Processing file: {filename}")
    
    # Initialize counts for this specific file
    local_evt_counter = {cut: 0 for cut in cut_names}
    
    try:
        # ----- Load Dorothea & Sophronia ----- #
        df_doro = pd.read_hdf(filepath, key=DORO_KEY).loc[:, DORO_COLUMNS]      # Keep only relevant columns
        df_soph = pd.read_hdf(filepath, key=SOPH_KEY).loc[:, SOPH_COLUMNS]      # Keep only relevant columns
        df_soph.rename(columns={'Z': 'DT'}, inplace=True)                       # Rename Z to DT for consistency
        df_soph['Z'] = df_soph['DT'] * V_DRIFT                                  # Compute real Z position: using the drift velocity
        local_evt_counter[cut_names[0]] = df_soph['event'].nunique()

        # ----- Energy Correction ----- #
        df_soph = bf.correct_energy_by_map(df_soph, read_maps(kr_path))

        # ----- Basic Cuts & S1e Correction ----- #
        # Z >= 0
        current_event_ids = df_soph['event'].unique()
        events_with_negative_z_hits = df_soph.loc[df_soph['Z'] < 0, 'event'].unique()
        events_with_positive_z_hits = np.setdiff1d(current_event_ids, events_with_negative_z_hits)
        df_doro, df_soph = bf.apply_cut_and_update(df_doro, df_soph, event_ids=events_with_positive_z_hits)
        local_evt_counter[cut_names[1]] = df_soph['event'].nunique()
        # nS1 <= 1 (NO-Polike)
        s1_mask = (df_doro['nS1'] == 0) | ((df_doro['nS1'] == 1) & (df_doro['S1h'] >= M_NOPOLIKE * df_doro['S1e'] + B_NOPOLIKE))
        df_doro, df_soph = bf.apply_cut_and_update(df_doro, df_soph, cut_mask=s1_mask, df_for_mask=df_doro)
        local_evt_counter[cut_names[2]] = df_soph['event'].nunique()
        # nS2 = 1
        s2_mask = (df_doro['nS2'] == 1)
        df_doro, df_soph = bf.apply_cut_and_update(df_doro, df_soph, cut_mask=s2_mask, df_for_mask=df_doro)
        local_evt_counter[cut_names[3]] = df_soph['event'].nunique()
        # S1e Correction
        df_doro = crudo.correct_S1e(df_doro, CV_FIT, DT_STOP, output_column='S1e_corr')     # Based on alpha analysis

        # ----- Particle Tagging & Processing ----- #
        df_soph = bf.tag_particles(df_soph, energy_threshold=ENERGY_THRESHOLD)
        # Electrons
        df_electron = df_soph[df_soph['particle'] == 'electron'].copy()
        final_electron_df = bf.process_electrons(df_electron, cluster_config=CLUSTER_CONFIG)
        local_evt_counter[cut_names[4]]  = final_electron_df['event'].nunique()
        # Alphas
        df_alpha = df_soph[df_soph['particle'] == 'alpha'].copy()        
        final_alpha_df = bf.process_alphas(df_alpha, q_threshold=Q_LIM_ALPHA)
        local_evt_counter[cut_names[5]] = final_alpha_df['event'].nunique()
        # Merge final Sophronia dataframe
        df_soph_final = pd.concat([final_electron_df, final_alpha_df], ignore_index=True, sort=False)
        
        # ----- Data @ Event-Level ----- #
        df_event_final = bf.aggregate_to_event_level(df_soph_final, df_doro)
        # HE scale correction
        df_event_final = bf.energy_pe_to_mev(df_event_final, slope=M_HE_SCALE, intercept=B_HE_SCALE)

    except Exception as e:
        print(f"   Failed to process file {filename}. Error: {e}", file=sys.stderr)
        # Return a dictionary of zeros on failure to not affect the final sum
        return pd.DataFrame(), pd.DataFrame(), {name: 0 for name in cut_names}

    return df_event_final, df_soph_final, local_evt_counter

# =============================================================================
# ----- MAIN -----
# =============================================================================

def main():
    """
    Música maestro! This is the main function that orchestrates the processing
    """
    # 1. PARSE COMMAND-LINE ARGUMENTS
    #    EXTRACT RUN INFORMATION
    #    LOAD CORRESPONDING KRYPTON MAP FOR ENERGY CORRECTION
    #    SET UP PATHS TO PROCESS
    args = parse_arguments()
    print("\n----- Processing Configuration -----")
    print(f"Run Number: {args.run_number}")

    RUNS_INFO_DF = pd.read_csv(RUNS_INFO_PATH, index_col='run_number')
    RUNS_INFO_DF.columns = RUNS_INFO_DF.columns.str.strip()
    if args.run_number not in RUNS_INFO_DF.index:
        print(f"   Error: Run {args.run_number} not found in runs information.", file=sys.stderr)
        sys.exit(1)
    # Extract run information from the dataframe
    RUN_DURATION = RUNS_INFO_DF.at[args.run_number, 'duration']
    RUN_OK   = RUNS_INFO_DF.at[args.run_number, 'OK']
    RUN_LOST = RUNS_INFO_DF.at[args.run_number, 'LOST']
    print(f"Duration: {RUN_DURATION} s, OK: {RUN_OK}, LOST: {RUN_LOST}")

    kr_file = next((f for f in os.listdir(ICAROS_DIR) if f'run_{args.run_number}' in f and f.endswith('.map.h5')), None)
    if not kr_file:
        raise FileNotFoundError(f"   Error: NO Kr map file found for run {args.run_number} in {ICAROS_DIR}")
        sys.exit(1)
    KR_PATH = os.path.join(ICAROS_DIR, kr_file)
    print(f"Kr map file found: {kr_file}")

    # Construct specific list of files to process based on run number
    files_to_process = []
    # ----- LDC Loop ----- #
    for ldc in range(1, 8):
        # Load the HDF5 file
        h5_path = os.path.join(DATA_DIR, f'run_{args.run_number}_ldc{ldc}_trg2_sophronia.h5')
        if os.path.isfile(h5_path):
            files_to_process.append(h5_path)
    
    if not files_to_process:
        print(f"   Error: No .h5 files found to process.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("------------------------------------")
    print("Files to Process:")
    for fp in files_to_process:
        print(os.path.basename(fp))
    print(f"Output Directory: {OUTPUT_DIR}")
    print("------------------------------------")

    # 2. PARALLEL PROCESSING OF FILES
    n_cores = os.cpu_count() - 1 if os.cpu_count() > 1 else 1
    print(f"\n----- Starting parallel processing on {n_cores} cores")

    # The Parallel object manages the pool of worker processes.
    # `delayed(process_file)` creates a lightweight "promise" of a function call.
    results = Parallel(n_jobs=n_cores)(delayed(process_file)(fp, KR_PATH) for fp in files_to_process)
    print("----- Parallel processing finished")

    # 3. COMBINE RESULTS FROM ALL FILES
    print("\n----- Aggregating results")
    all_processed_dfs = []
    all_sophronia_dfs = []
    total_cut_counts = {name: 0 for name in CUT_NAMES}

    # Unpack the results (dataframes, counts dict)
    for df_file, df_soph, local_counts in results:
        if not df_file.empty and not df_soph.empty:
            all_processed_dfs.append(df_file)
            all_sophronia_dfs.append(df_soph)
        for cut_name, count in local_counts.items():
            total_cut_counts[cut_name] += count

    # Concatenate dataframes
    if all_processed_dfs:
        run_event_df = pd.concat(all_processed_dfs, ignore_index=True)
    else:
        run_event_df = pd.DataFrame()
    if all_sophronia_dfs:
        run_sophronia_df = pd.concat(all_sophronia_dfs, ignore_index=True)
    else:
        run_sophronia_df = pd.DataFrame()

    # DEBUGGING: Check the contents of the lists before concatenation
    print(f"Event dataframe shape: {run_event_df.shape}\nSophronia dataframe shape: {run_sophronia_df.shape}")

    # 4. PERFORM SELECTION
    print("\n----- Applying trigger02 cut and tagging by detector regions")
    # --- Trigger2 ---
    trg2_mask = (run_event_df['E_mev'] >= TRG2_THRESHOLD)
    run_event_df, run_sophronia_df = bf.apply_cut_and_update(run_event_df, run_sophronia_df, cut_mask=trg2_mask, df_for_mask=run_event_df)
    # Event counts by particle after trigger2 cut
    trg2_counts = run_event_df['particle'].value_counts()
    total_cut_counts['nElectron_Trg2'] = trg2_counts.get('electron', 0)
    total_cut_counts['nAlpha_Trg2']    = trg2_counts.get('alpha', 0)

    # --- Detector Regions Tagging ---
    event_region_tags = bf.tag_event_by_detector_region(run_event_df, z_cut_low=Z_LOW, z_cut_high=Z_UP, r_cut_high=R_UP)
    # Add region tags to both dataframes
    run_event_df['region'] = event_region_tags.values
    run_sophronia_df       = run_sophronia_df.merge(run_event_df[['event', 'region']], on='event', how='left')
    # Event counts by particle and region
    detector_region_counts = run_event_df.groupby(['particle', 'region'])['event'].nunique()
    for particle in ['electron', 'alpha']:
        for region in ['fiducial', 'tube', 'cathode', 'anode']:
            key = f"n{particle.capitalize()}_{region.capitalize()}"
            total_cut_counts[key] = detector_region_counts.get((particle, region), 0)

    # 4. OUTPUT
    print("\n----- Saving output files")
    # npeak column in run_event_df is uint64, convert to int64
    for col in run_event_df.select_dtypes(include=['uint64']).columns:
        run_event_df[col] = run_event_df[col].astype('int64')
    # Combine all processed dataframes into one
    output_filepath = os.path.join(OUTPUT_DIR, f"processed_run_{args.run_number}.h5")
    print(f"Opening HDF5 store for writing: {output_filepath}")
    try:
        with pd.HDFStore(output_filepath, mode='w') as store:
            if not run_event_df.empty:
                store.put('Events', run_event_df, format='table', data_columns=True)
            if not run_sophronia_df.empty:
                store.put('Hits', run_sophronia_df, format='table', data_columns=True)
        print("HDF5 saving complete.")
    except Exception as e:
        print(f"   Error writing to HDF5 file: {e}", file=sys.stderr)
    
    # Summary file
    summary_data = {
                        'Run_ID': [args.run_number],
                        'Duration': [RUN_DURATION],
                        'Date_CV': [round(run_event_df['time'].mean(), 4)],
                        'Date_Err': [round(run_event_df['time'].sem(), 4)],
                        'OK': [RUN_OK],
                        'LOST': [RUN_LOST]
                    }
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
                                index='Run_ID',
                             )
        print("Summary file updated.")
    except IOError as e:
        print(f"   Error writing to summary file: {e}", file=sys.stderr)

    print("\nY ya, eso es todo, eso es todo ♥")

if __name__ == "__main__":
    main()