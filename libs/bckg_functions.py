import sys
sys.path.append('/lhome/ific/c/ccortesp/Analysis')

from invisible_cities.reco.corrections import apply_all_correction
from invisible_cities.types.symbols import NormStrategy
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import Callable, List, Optional, Tuple, Union

# =============================================================================
# ----- HELPER FUNCTIONS -----
# =============================================================================

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

def weighted_avg(series, weight):
    if weight.sum() == 0:   # Avoid division by zero
        return np.nan
    return np.average(series, weights=weight)

def R_max_func(group_df):
    return np.sqrt(group_df['X']**2 + group_df['Y']**2).max()

# =============================================================================
# ----- PROCESSING FUNCTIONS -----
# =============================================================================

def correct_energy_by_map(df: pd.DataFrame, cmap) -> pd.DataFrame:
    """
    Applies energy correction and cleans negative/NaN values.
    """
    corr_func = apply_all_correction(cmap, apply_temp=True, norm_strat=NormStrategy.max)
    x_vals, y_vals, z_vals, t_vals = df.X.values, df.Y.values, df.Z.values, df.time.values
    
    df['corr_factor'] = corr_func(x_vals, y_vals, z_vals, t_vals)
    df['E_corr'] = df['E'] * df['corr_factor']
    
    # NaN or negative energy to 0: hit-level
    df['E_corr'] = np.where(pd.notna(df['E_corr']) & (df['E_corr'] > 0), df['E_corr'], 0)
    
    return df

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

def tag_particles(df: pd.DataFrame, energy_threshold: float) -> pd.DataFrame:
    """
    Tags hits as 'electron' or 'alpha' based on total event energy.
    """
    event_energy = df.groupby('event')['E_corr'].transform('sum')
    df['particle'] = np.where(event_energy <= energy_threshold, 'electron', 'alpha')
    
    return df

def process_electrons(
                        df_electron: pd.DataFrame,
                        cluster_config: dict
                     ) -> pd.DataFrame:
    """
    Processes a DataFrame of electron hits to filter noise and conserve energy.

    The process involves three main steps:
    1.  Tags all hits as either "isolated" or "non-isolated" using a 3D clustering algorithm.
    2.  Identifies which "isolated" hits are physically plausible by checking if they fall
        within the Z-span of their corresponding non-isolated event track.
    3.  Redistributes the energy of these plausible isolated hits proportionally among the
        non-isolated hits of the same event.

    Args:
        df_electron (pd.DataFrame): DataFrame containing only hits from electron-like events.
        cluster_config (dict): A dictionary with parameters for the clustering algorithm,
                               e.g., {'distance': [16., 16., 4.], 'nhit': 5}.

    Returns:
        pd.DataFrame: A DataFrame containing only the final, processed non-isolated hits,
                      with an added 'E_final_pe' column containing the redistributed energy.
    """
    if df_electron.empty:
        return pd.DataFrame(columns=list(df_electron.columns) + ['E_hit_pe'])

    # STEP A: Tag all hits as isolated or non.isolated
    splitter = split_isolated_clusters_3D(**cluster_config)
    isolated_hits_indices = []
    for _, group in df_electron.groupby(['event', 'npeak']):
        if group.empty: continue
        _, isolated_df = splitter(group)
        if not isolated_df.empty:
            isolated_hits_indices.append(isolated_df.index)
    # Tagging
    df_electron['is_isolated'] = False
    if isolated_hits_indices:
        iso_indices = np.concatenate(isolated_hits_indices)
        df_electron.loc[iso_indices, 'is_isolated'] = True

    # STEP B: Identify "good" isolated hits
    non_isolated_hits_df = df_electron[~df_electron['is_isolated']].copy()
    isolated_hits_df     = df_electron[df_electron['is_isolated']].copy()
    if non_isolated_hits_df.empty:
        return pd.DataFrame(columns=list(df_electron.columns) + ['E_hit_pe'])

    # Find the Z-span for the main track of each event
    z_ranges = non_isolated_hits_df.groupby('event')['Z'].agg(['min', 'max']).rename(columns={'min': 'Z_min', 'max': 'Z_max'})
    # Determine the total energy from isolated hits that fall within their event's Z-span
    energy_to_add = (
                        isolated_hits_df
                        .merge(z_ranges, on='event', how='left')
                        .dropna(subset=['Z_min', 'Z_max'])
                        .query("Z >= Z_min and Z <= Z_max")
                        .groupby('event')['E_corr'].sum().rename('E_iso_to_add')
                    )
    
    # STEP C: Redistribute the "good" isolated energy to non-isolated hits
    if not energy_to_add.empty:
        non_isolated_hits_df = non_isolated_hits_df.merge(energy_to_add, on='event', how='left')
        non_isolated_hits_df['E_iso_to_add'].fillna(0, inplace=True)
        # Calculate the total energy of the main track for proportional scaling
        total_non_iso_energy = non_isolated_hits_df.groupby('event')['E_corr'].transform('sum').replace(0, 1)
        # Redistribution formula
        non_isolated_hits_df['E_hit_pe'] = (non_isolated_hits_df['E_corr'] + 
                                              (non_isolated_hits_df['E_corr'] / total_non_iso_energy) * non_isolated_hits_df['E_iso_to_add'])
    else:
        non_isolated_hits_df['E_hit_pe'] = non_isolated_hits_df['E_corr']

    # Drop additional columns
    non_isolated_hits_df.drop(columns=['is_isolated', 'E_iso_to_add'], inplace=True, errors='ignore')

    return non_isolated_hits_df

def process_alphas(df_alpha: pd.DataFrame, q_threshold) -> pd.DataFrame:
    """
    Processes alpha hits: applies a simple charge cut.
    """
    if df_alpha.empty:
        return pd.DataFrame(columns=list(df_alpha.columns) + ['E_hit_pe'])

    final_alpha_df = df_alpha[df_alpha['Q'] >= q_threshold].copy()
    # To ensure concatenation works, add the 'E_final_pe' column
    final_alpha_df['E_hit_pe'] = final_alpha_df['E_corr']
    return final_alpha_df


def aggregate_to_event_level(df_soph: pd.DataFrame, df_doro: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates hit-level data to event-level summary data.
    """
    if df_soph.empty:
        return pd.DataFrame()

    # ----- Dorothea Aggregation ----- #
    doro_agg_dict = {
                        'nS1': ('nS1', 'max'),
                        'nS2': ('nS2', 'max'),
                        'S1e': ('S1e', 'max'),
                        'S1e_corr': ('S1e_corr', 'max'),
                    }
    doro_agg_df = df_doro.groupby('event', as_index=False).agg(**doro_agg_dict)

    # ----- Sophronia Aggregation ----- #
    soph_agg_dict = {
                        'time': ('time', 'mean'),
                        'X_bary': ('X', lambda x: weighted_avg(x, df_soph.loc[x.index, 'E_hit_pe'])),
                        'Y_bary': ('Y', lambda y: weighted_avg(y, df_soph.loc[y.index, 'E_hit_pe'])),
                        'Z_bary': ('Z', lambda z: weighted_avg(z, df_soph.loc[z.index, 'E_hit_pe'])),
                        'E_pe': ('E_hit_pe', 'sum'),
                        'Z_min': ('Z', 'min'),
                        'Z_max': ('Z', 'max'),
                        'R_max': ('X', lambda g: R_max_func(df_soph.loc[g.index])),
                        'particle': ('particle', 'first')
                    }

    # Perform aggregation
    df_file = df_soph.groupby(['event', 'npeak'], as_index=False).agg(**soph_agg_dict)
    df_file = pd.merge(df_file, doro_agg_df, on='event', how='left')

    return df_file

def energy_pe_to_mev(df: pd.DataFrame, slope, intercept) -> pd.DataFrame:
    """
    Converts energy from photoelectrons (pe) to mega-electronvolts (MeV).
    Use HE energy scale factor.
    """
    df['E_mev'] = slope * df['E_pe'] + intercept
    return df

def tag_event_by_detector_region(
                                    df_event: pd.DataFrame,
                                    z_cut_low: float,
                                    z_cut_high: float,
                                    r_cut_high: float,
                                    event_col: str = 'event'
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
        df_event (pd.DataFrame): The event-level summary DataFrame. 
                                 Must contain columns like 'Z_min', 'Z_max', 'R_max', and 'nS1'.
        z_cut_low (float): The lower Z boundary for the fiducial volume.
        z_cut_high (float): The upper Z boundary for the fiducial volume.
        r_cut_high (float): The radial boundary for the fiducial volume.
        event_col (str): The name of the column representing the event ID. Default is 'event'.

    Returns:
        pd.Series: A pandas Series with the same index as `df_event`, containing
                   a string tag ('Fiducial', 'Tube', 'Anode', 'Cathode', 'Unclassified')
                   for each event.
    """
    # Base masks
    has_no_s1 = (df_event['nS1'] == 0)
    crosses_anode_z   = (df_event['Z_min'] < z_cut_low)
    crosses_cathode_z = (df_event['Z_max'] > z_cut_high)
    is_fully_z_contained = ((df_event['Z_min'] >= z_cut_low) & (df_event['Z_max'] <= z_cut_high))
    is_r_contained = (df_event['R_max'] <= r_cut_high)

    # Conditions and choices for np.select (priority order matters!)    
    conditions = [
                    has_no_s1,                                  # 1. Highest priority: nS1=0 is Anode.
                    crosses_anode_z,                            # 2. If any part is in anode Z, it's Anode.
                    crosses_cathode_z,                          # 3. If any part is in cathode Z, it's Cathode.
                    is_fully_z_contained & is_r_contained,      # 4. If contained in Z and R, it's Fiducial.
                    is_fully_z_contained & ~is_r_contained      # 5. If contained in Z but not R, it's Tube.
                 ]
    choices = ['anode', 'anode', 'cathode', 'fiducial', 'tube']

    # Apply classification
    region_tags = np.select(conditions, choices, default='unclassified')

    # Return a pandas Series with the specified event column as the index for easy mapping
    return pd.Series(region_tags, index=df_event[event_col])