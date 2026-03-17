# =============================================================================
# create_final_features.py
#
# Reads the "clean" reduced files for a full run.
# For each LDC in the run, it:
# 1. Applies a final cleaning logic (e.g., Z-range cut on isolated hits).
# 2. Calculates all high-level, event-wide features.
# 3. Saves a final, dense, analysis-ready dataframe.
# =============================================================================
import os
import glob
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import griddata

# =============================================================================
# --- HELPER FUNCTIONS ---
# =============================================================================
def calculate_barycenter(group):
    """Calculates charge-weighted barycenter and total E_corr."""
    total_charge = group['Q'].sum()
    if total_charge == 0:
        return pd.Series({'X_bary': np.nan, 'Y_bary': np.nan, 'Z_bary': np.nan})
    return pd.Series({
        'X_bary': (group['X'] * group['Q']).sum() / total_charge,
        'Y_bary': (group['Y'] * group['Q']).sum() / total_charge,
        'Z_bary': (group['Z'] * group['Q']).sum() / total_charge
    })
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
# --- MAIN SCRIPT LOGIC ---
# =============================================================================
def main():
    # ----- Configuration -----
    run_id = 15593
    max_ldc = 7

    # Paths
    base_dir = "/lustre/ific.uv.es/prj/gl/neutrinos/users/villamil/kr_next100/HE/"
    clean_data_dir = os.path.join(base_dir, f"{run_id}/sophronia/clean_df/")
    final_output_dir = os.path.join(base_dir, f"{run_id}/sophronia/final_dst/")
    os.makedirs(final_output_dir, exist_ok=True)

    time_filename  = "/lustre/ific.uv.es/prj/gl/neutrinos/users/villamil/kr_next100/HE/energy_scale_he.h5"
    map_file_path = f"/lustre/ific.uv.es/prj/gl/neutrinos/users/villamil/kr_next100/HE/combined_15546_15557.map3d"
    corr3d = get_corr3d(map_file_path)

    # ----- Loop over LDCs -----
    for ldc in range(1, max_ldc + 1):
        print(f"\n{'='*15} Starting Feature Creation for Run {run_id}, LDC {ldc} {'='*15}")

        # --- Step A: Load the clean data ---
        clean_filepath = os.path.join(clean_data_dir, f'run_{run_id}_ldc{ldc}_clean.h5')
        if not os.path.exists(clean_filepath):
            print(f"Clean data file not found: {clean_filepath}. Skipping.")
            continue

        print(f"Loading clean data: {os.path.basename(clean_filepath)}")
        dst_df = pd.read_hdf(clean_filepath, 'DST/Events')
        hits_df = pd.read_hdf(clean_filepath, 'RECO/Events')


        hm_s2s = hits_df.groupby('event').agg(npeaks_count=('npeak', 'nunique'))
        dst_df = dst_df.merge(hm_s2s, on='event', how='left')


        # --- Step C: Calculate all high-level features ---
        print("Calculating high-level event features...")
      
        #first thing, remove any hit with negative energy
        hits_df = hits_df[hits_df.E > 0]


        #apply geometrical correcitons
        corr_factors = corr3d(hits_df.Z, hits_df.X, hits_df.Y)
        hits_df['E_corr'] = hits_df.E * corr_factors
        
        #apply also time ev corrections
        corrt  = get_corrt (time_filename)
        hits_df['E_corr_time'] = hits_df.E_corr * corrt(hits_df.time)


        # --- Step B: Apply final hit cleaning via ENERGY REDISTRIBUTION ---
        print("Applying final hit cleaning logic (Z-range cut and Energy Redistribution)...")
        
        all_non_iso = hits_df[~hits_df['is_isolated']].copy()
        all_iso = hits_df[hits_df['is_isolated']].copy()
        
        # Determine Z-range from non-isolated hits
        if not all_non_iso.empty and not all_iso.empty:
            z_ranges = all_non_iso.groupby('event')['Z'].agg(['min', 'max']).rename(columns={'min': 'Z_min', 'max': 'Z_max'})
            iso_with_z_range = all_iso.merge(z_ranges, on='event', how='left')
            iso_with_z_range.dropna(subset=['Z_min', 'Z_max'], inplace=True)

            #==============================================
            #NEW MODIFICATION
            # here we're going to clean the hits by removing background from the signal/back study
            #so first remove noise from both dataframes
            all_non_iso = all_non_iso[all_non_iso.E > 1] 
            iso_with_z_range = iso_with_z_range[iso_with_z_range.E > 1]
            #==============================================
            
            # Find the isolated hits that PASS the Z-cut
            z_cut_mask = (iso_with_z_range['Z'] >= iso_with_z_range['Z_min']) & (iso_with_z_range['Z'] <= iso_with_z_range['Z_max'])
            kept_iso_hits = iso_with_z_range[z_cut_mask]

            #==============================================
            #SECOND PART OF THE MODIFICATION    
            #second, apply min and max values for energy of the IN z hits, according to the peak sig/back
            energy_min_signal = 50.0   # Lower bound of the positive peak
            energy_max_signal = 1000.0
            kept_iso_hits = kept_iso_hits[(kept_iso_hits.E > energy_min_signal) & (kept_iso_hits.E < energy_max_signal)]  
            #==============================================      
            
            #HERE WE KEEP THE ENERGIES ONLY WITH GEOMETRICAL CORRECITONS AND WITH TIME CORRECTIONS SEPARATED
            # Calculate the total energy of these kept hits for each event
            energy_to_redistribute = kept_iso_hits.groupby('event')['E_corr'].sum().rename('E_iso_to_add')
            energy_to_redistribute_te = kept_iso_hits.groupby('event')['E_corr_time'].sum().rename('E_iso_to_add_te')
            
            # Merge this energy into the non-isolated dataframe
            all_non_iso = all_non_iso.merge(energy_to_redistribute, on='event', how='left')
            all_non_iso['E_iso_to_add'].fillna(0, inplace=True)

            all_non_iso = all_non_iso.merge(energy_to_redistribute_te, on='event', how='left')
            all_non_iso['E_iso_to_add_te'].fillna(0, inplace=True)
            
            # Calculate the proportional redistribution
            # Using transform() is efficient for this group operation
            non_iso_energy_sum = all_non_iso.groupby('event')['E_corr'].transform('sum').replace(0, 1)
            non_iso_energy_sum_te = all_non_iso.groupby('event')['E_corr_time'].transform('sum').replace(0, 1) # Avoid division by zero
            
            # The final energy of a non-isolated hit is its own corrected energy
            # plus its proportional share of the kept isolated energy.
            all_non_iso['E_final'] = all_non_iso['E_corr'] + \
                                     (all_non_iso['E_corr'] / non_iso_energy_sum) * all_non_iso['E_iso_to_add']

            all_non_iso['E_final_te'] = all_non_iso['E_corr_time'] + \
                                     (all_non_iso['E_corr_time'] / non_iso_energy_sum_te) * all_non_iso['E_iso_to_add_te']
            # Print stats
            discarded_iso_count = len(all_iso) - len(kept_iso_hits)
            total_iso_count = len(all_iso) if len(all_iso) > 0 else 1
            perc_discarded = (discarded_iso_count / total_iso_count) * 100
            print(f"  - Kept and redistributed energy from {len(kept_iso_hits)} isolated hits.")
            print(f"  - Discarded {discarded_iso_count} isolated hits outside Z-range ({perc_discarded:.2f}% of isolated).")

        else:
            # If there are no isolated hits, the final energy is just the corrected energy
            all_non_iso['E_final'] = all_non_iso['E_corr']
            
        # The dataframe used for all subsequent feature creation is now all_non_iso
        # It contains the final, fully processed energy in the 'E_final' column
        final_hits = all_non_iso
        #final_hits["Z_corr"] = final_hits.Z * 0.865

        
        #Here we deal frist with the variables coming form the hit dataframe

        # Total corrected energy, total charge, hit count from the final clean sample
        final_hits["R"]  = np.sqrt( final_hits.X**2 + final_hits.Y**2 )


        event_agg = final_hits.groupby('event').agg(
            E_final=('E_final', 'sum'),
            E_final_te=('E_final_te', 'sum'),
            Rmax=('R', 'max'),
            Zmax=('Z', 'max'),
            Zmin=('Z', 'min'),
            #E_final_time=('E_corr_time', 'sum'),    
            Q_final=('Q', 'sum'),
            hits_final=('event', 'size')
        )

        #final_hits["Z_corr"] = final_hits.Z * 0.865 # 0.865 mm/mus is the drift velocity
        
        # Charge-weighted barycenter of the final clean sample
        barycenter_agg = final_hits.groupby('event').apply(calculate_barycenter)
        
        # Max/min positions from the final clean sample
        maxmin_agg = final_hits.groupby('event').agg(
            Xmax_final=('X', 'max'), Ymax_final=('Y', 'max'), Zmax_final=('Z', 'max'),
            Xmin_final=('X', 'min'), Ymin_final=('Y', 'min'), Zmin_final=('Z', 'min'),
        )
        maxmin_agg['Rmax_final'] = np.sqrt(maxmin_agg['Xmax_final']**2 + maxmin_agg['Ymax_final']**2)
        maxmin_agg['Zdiff_final'] = maxmin_agg['Zmax_final'] - maxmin_agg['Zmin_final']


        #Then we emrge those final event-level variables with the dst dataframe
        # --- Step D: Merge all features into a final dataframe ---
        print("Merging features into final dataframe...")
        # Start with the clean DST, which contains the original event info
        final_df = dst_df.copy()

        final_df.drop(columns=['Nsipm', 's1_peak', 's2_peak', 'nS1', 'nS2', 'Z', 'X', 'Y','Zrms', 'Phi', 'Xrms', 'Yrms'], inplace=True, errors='ignore')
        
        # Merge the new features, using suffixes to avoid column name collisions
        final_df = final_df.merge(event_agg, on='event', how='left')
        final_df = final_df.merge(barycenter_agg, on='event', how='left')
        final_df = final_df.merge(maxmin_agg, on='event', how='left')
        
        # Drop events that were lost during cleaning (if they had no non-iso hits)
        final_df.dropna(subset=['E_final'], inplace=True)
        print(f"  - Produced a final feature set for {len(final_df)} events.")

        # --- Step E: Save the final, dense dataframe ---
        output_filepath = os.path.join(final_output_dir, f'run_{run_id}_ldc{ldc}_final_features_with_TE_v2.h5')
        print(f"Saving final feature dataframe to: {output_filepath}")
        final_df.to_hdf(output_filepath, key='FinalFeatures', mode='w', format='table')

        print(f"--- Feature Creation Complete for LDC {ldc}. ---")

if __name__ == "__main__":
    main()