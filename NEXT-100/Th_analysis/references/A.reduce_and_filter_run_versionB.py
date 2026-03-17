# =============================================================================
# reduce_raw_data.py
#
# Reads raw sophronia DST and RECO files for a full run.
# For each LDC in the run, it:
# 1. Applies event selection cuts based on the DST.
# 2. Corrects the hits from selected events using a map and rescues NaNs.
# 3. Tags each hit as isolated or non-isolated.
# 4. Saves a reduced DST and a clean, corrected, tagged RECO dataframe.
# =============================================================================

# THIS VERSIONB OF THE CODE IS NOT APPLYING MAP CORERCTIONS AS THE ORIGINAL CODE WAS.
# IS ONLY REDUCING THE SIZE OF THE FILES, AND THEYRE ACTUALLY SMALLER THAN WITH THE PREVIOUS VERSION

import os
import glob
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import List, Callable, Tuple
from  invisible_cities.io.dst_io            import load_dst
from  invisible_cities.reco.corrections     import read_maps, apply_all_correction
from  invisible_cities.types.symbols        import NormStrategy
from invisible_cities.core.core_functions     import in_range


# =============================================================================
# --- HELPER FUNCTIONS ---
# =============================================================================


def split_isolated_clusters_3D(distance: List[float], nhit: int) -> Callable:
    """Creates a function that splits hits using a 3D anisotropic algorithm."""
    dist = np.sqrt(3) # Normalized distance
    def split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if len(df) <= nhit: return df.iloc[:0], df
        xyz = df[['X', 'Y', 'Z']].values / distance
        try:
            nbrs = NearestNeighbors(radius=dist).fit(xyz)
            neighbors = nbrs.radius_neighbors(xyz, return_distance=False)
            mask_non_iso = np.array([len(neigh) > nhit for neigh in neighbors])
            return df[mask_non_iso], df[~mask_non_iso]
        except Exception:
            return df.iloc[:0], df.iloc[:0]
    return split

def get_selected_event_ids(kdst_df, cuts_config):
    """Applies Thallium selection cuts and returns a set of event IDs."""
    print("  - Applying selection cuts to kDST...")
    selection_df = kdst_df.copy()
    selection_df = selection_df[
        (selection_df.nS1 == 1) & 
        (selection_df.nS2 == 1) 
    ]
    selection_df = selection_df[selection_df.DT > 0]
    selection_df = selection_df[selection_df.S1e < (0.32 * selection_df.DT + 500)]
    selection_df = selection_df[selection_df.Z > 20]
    selection_df = selection_df[selection_df.Z < 1184.185]
    selection_df = selection_df[selection_df.R < 400]
    final_event_ids = set(selection_df['event'].unique())
    print(f"  - {len(final_event_ids)} events passed the selection.")
    return final_event_ids

# =============================================================================
# --- MAIN SCRIPT LOGIC ---
# =============================================================================

def main():
    # ----- 1. Configuration -----
    #BEFORE USING THIS SCRIPT AGAIN, NIT SHOULD BE CHANGED TO USE THE NEW MAP AND NOT ERASE THE E COLUMNS
    run_id = 15593
    max_ldc = 7   # The highest LDC number for this run to process


    # Columns to drop from RECO hits to save memory
    reco_columns_to_drop = ['Xpeak', 'Ypeak', 'nsipm', 'Xrms', 'Yrms', 'Qc', 'Ep', 'Ec']


    # Paths
    #base_dir = "/lhome/ific/v/villamil/kr_next100/HE/"
    base_dir = "/lhome/ific/v/villamil/kr_next100/HE"
    out_dir = f"/lustre/ific.uv.es/prj/gl/neutrinos/users/villamil/kr_next100/HE/"
    #map_file_path = f"/lustre/ific.uv.es/prj/gl/neutrinos/users/villamil/kr_next100/HE/{run_id}/map/run_{run_id}.v2.3.1.20250717.Kr.map.h5"
    output_dir_base = os.path.join(out_dir, f"{run_id}/sophronia/clean_df/")
    os.makedirs(output_dir_base, exist_ok=True)

    cuts_config = {"max_radius": 400.0, "min_z": 20.0, "max_z": 1184.185}
    cluster_config = {"distance": [16., 16., 4.], "nhit": 3}

    # ----- 3. Loop over LDCs -----
    for ldc in range(1, max_ldc + 1):
        print(f"\n{'='*15} Starting processing for Run {run_id}, LDC {ldc} {'='*15}")
        raw_sophronia_dir = os.path.join(base_dir, f"{run_id}/sophronia/ldc{ldc}/")
        
        # --- Step A: Load all raw data for the LDC ---
        raw_files = sorted(glob.glob(os.path.join(raw_sophronia_dir, "*.h5")))
        if not raw_files:
            print(f"No raw files found in {raw_sophronia_dir}. Skipping.")
            continue
        print(f"Loading {len(raw_files)} raw files for LDC {ldc}...")
        all_dsts = pd.concat([pd.read_hdf(f, 'DST/Events') for f in raw_files], ignore_index=True)
        all_reco_hits = pd.concat([pd.read_hdf(f, 'RECO/Events') for f in raw_files], ignore_index=True)

        
        print(f"  - Dropping {len(reco_columns_to_drop)} unused columns from RECO hits to save memory...")
        
        # The errors='ignore' flag is crucial. It prevents the script from
        # crashing if a column to be dropped doesn't exist in a particular file.
        all_reco_hits.drop(columns=reco_columns_to_drop, inplace=True, errors='ignore')


        # --- Step B: Apply Event Selection ---
        event_ids_to_keep = get_selected_event_ids(all_dsts, cuts_config)
        if not event_ids_to_keep:
            print("No events passed selection for this LDC. Skipping.")
            continue

        # --- Step C: Filter DST and RECO dataframes ---
        clean_dst = all_dsts[all_dsts['event'].isin(event_ids_to_keep)].copy()
        clean_hits = all_reco_hits[all_reco_hits['event'].isin(event_ids_to_keep)].copy()
        print(f"  - Reduced to {len(clean_hits)} hits.")

        
        # --- Step E: Add the 'is_isolated' flag ---
        print("Tagging hits as isolated/non-isolated...")
        splitter = split_isolated_clusters_3D(**cluster_config)
        list_of_iso_indices = []
        for _, group in clean_hits.groupby(['event', 'npeak']):
            if group.empty: continue
            _, iso = splitter(group)
            if not iso.empty:
                list_of_iso_indices.append(iso.index)
        
        if list_of_iso_indices:
            iso_indices = np.concatenate(list_of_iso_indices)
            clean_hits['is_isolated'] = False
            clean_hits.loc[iso_indices, 'is_isolated'] = True
            perc_iso = len(iso_indices) / len(clean_hits) * 100
            print(f"  - Tagged {len(iso_indices)} hits as isolated. ({perc_iso:.2f}%)")
        else:
            clean_hits['is_isolated'] = False
            print("  - No isolated hits found in this LDC.")

        # --- Step F: Save the final, clean files ---
        output_filepath = os.path.join(output_dir_base, f'run_{run_id}_ldc{ldc}_clean.h5')
        print(f"Saving final clean data to: {output_filepath}")

        clean_dst.to_hdf(output_filepath, key='DST/Events', mode='w', format='table')
        # Drop raw E after correction and save
        #clean_hits.drop(columns=['E'], inplace=True, errors='ignore') 
        clean_hits.to_hdf(output_filepath, key='RECO/Events', mode='a', format='table')

        print(f"--- Reduction complete for LDC {ldc}. ---")

if __name__ == "__main__":
    main()