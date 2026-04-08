#
# Topology Functions: A toolkit to treat/extract topological information from events/hits in NEXT.
#

from functools import partial
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from typing import Callable

# ----- Hit-level ----- #

# 'event_column' is the only difference wrt IC version, giving me more flexibility
def cluster_tagger(df_hits: pd.DataFrame, *, 
                   eps:float, min_samples:int, 
                   scale_xy:float, scale_z:float,
                   event_column:str='event') -> pd.DataFrame:
        """
        Applies DBSCAN clustering to hits on an event-by-event basis.

        This function processes a DataFrame of hits, groups them by event,
        scales their coordinates, and applies DBSCAN to identify spatial clusters.
        A 'cluster' column is added to the DataFrame with the resulting labels.

        Parameters
        ----------
        df_hits     : pd.DataFrame
            DataFrame containing hit information with columns 'X', 'Y', 'Z', and 'event'.
        eps         : float
            The maximum distance between two samples for one to be considered as in the
            neighborhood of the other. This is the most important DBSCAN parameter.
        min_samples : int
            The number of samples (or total weight) in a neighborhood for a point
            to be considered as a core point.
        scale_xy    : float
            Scale factor to apply to X and Y coordinates before clustering to account
            for different detector resolutions.
        scale_z     : float
            Scale factor to apply to the Z coordinate.

        Returns
        -------
        pd.DataFrame
            The input DataFrame with an added 'cluster' column indicating the
            cluster label for each hit (-1 for noise).
        """
        if df_hits.empty:
            return df_hits.assign(cluster=pd.Series(dtype=int))  

        # Pre-allocate array for cluster labels
        cluster_labels = np.full(len(df_hits), -9999, dtype=int)

        # Get values once (faster than repeatedly accessing DataFrame columns)
        coords = df_hits[['X', 'Y', 'Z']].to_numpy()
        events = df_hits[event_column].to_numpy()

        # Use np.unique to get sorted event IDs
        unique_events = np.unique(events)
        for event_id in unique_events:
            
            mask = (events == event_id)
            X = coords[mask].copy()

            # Scale
            X[:, :2] /= scale_xy
            X[:, 2]  /= scale_z

            # DBSCAN clustering
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
            cluster_labels[mask] = labels

        df_hits['cluster'] = cluster_labels

        return df_hits


def hits_clusterizer(clustering_params: dict) -> Callable:
    """"
    This function receives a configuration dictionary and returns a callable
    that will perform DBSCAN clustering on a DataFrame of hits. 

    Parameters
    ----------
    clustering_params : dict
        A dictionary containing the configuration for the clustering algorithm.
        Expected keys are:
        - 'eps'        : float, Epsilon value for DBSCAN.
        - 'min_samples': int,   Min Samples value for DBSCAN.
        - 'scale_xy'   : float, optional, scale factor for XY coordinates.
        - 'scale_z'    : float, optional, scale factor for Z coordinate.

    Returns
    -------
    Callable
        A function that takes a DataFrame of hits and returns the same DataFrame 
        with an added 'cluster' column, which are the clusters labels assigned by DBSCAN
        (-1 for noise).
    """
    eps         = clustering_params['eps']
    min_samples = clustering_params['min_samples']
    scale_xy    = clustering_params['scale_xy']
    scale_z     = clustering_params['scale_z']
    
    return partial(cluster_tagger,
                   eps=eps, min_samples=min_samples,
                   scale_xy=scale_xy, scale_z=scale_z)

# def hits_clusterizer( eps          : float
#                     , min_samples  : float
#                     , scale_xy     : float = 14.55
#                     , scale_z      : float = 3.7
#                     , event_column : str = 'event'
#                     ) -> Callable:
#     """
#     Cluster hits in 3D space for each event using DBSCAN.
#     The coordinates are scaled to account for detector geometry differences in samplig 
    
#     Parameters
#     ----------
#     eps         : float, Epsilon value for DBSCAN.
#     min_samples : int, Min Samples value for DBSCAN.
#     scale_xy    : float, scale factor for XY coordinates.
#     scale_z     : float, scale factor for Z coordinate.
    
#     Returns
#     -------
#     Callable
#     A function that takes a DataFrame of hits and returns the same DataFrame 
#     with an added 'cluster' column, which are the clusters labels assigned by DBSCAN
#     (-1 for noise).
#     """
#     def cluster_tagger(df_hits: pd.DataFrame) -> pd.DataFrame:
#         if df_hits.empty:
#             return df_hits.assign(cluster=pd.Series(dtype=int))  

#         # Pre-allocate array for cluster labels
#         cluster_labels = np.full(len(df_hits), -9999, dtype=int)

#         # Get values once (faster than repeatedly accessing DataFrame columns)
#         coords = df_hits[['X', 'Y', 'Z']].to_numpy()
#         events = df_hits[event_column].to_numpy()

#         # Use np.unique to get sorted event IDs
#         unique_events = np.unique(events)

#         for event_id in unique_events:
#             mask = (events == event_id)
#             X = coords[mask].copy()

#             # Scale
#             X[:, :2] /= scale_xy
#             X[:, 2]  /= scale_z

#             # DBSCAN clustering
#             labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
#             cluster_labels[mask] = labels

#         df_hits['cluster'] = cluster_labels

#         return df_hits
    
#     return cluster_tagger

def deal_spurious_hits(df_hits: pd.DataFrame, energy_column='E_corr_pe') -> pd.DataFrame:
    """
    Processes a DataFrame of cluterized hits (extra 'cluster' column) to filter noise and conserve energy.

    The process involves two main steps:
    1.  Identifies which "noise" hits are physically plausible by checking if they fall
        within the Z-span of their corresponding clustered/non-isolated event track.
    2.  Redistributes the energy of these plausible noisy/isolated hits proportionally among the
        clustered/non-isolated hits of the same event.

    Args:
        df_hits (pd.DataFrame): DataFrame at hit-level with a 'cluster' column.
        energy_column (str, optional): The name of the energy column to use for redistribution. Defaults to 'E_corr_pe'.

    Returns:
        pd.DataFrame: A DataFrame containing only the final, processed non-isolated hits,
                      with an added 'E_hit_pe' column containing the redistributed energy per hit.
    """
    if df_hits.empty:
        return pd.DataFrame(columns=list(df_hits.columns) + ['E_hit_pe'])

    # STEP A: Identify "good" noisy/isolated hits
    noise_mask = (df_hits['cluster'] == -1)
    non_isolated_hits_df = df_hits[~noise_mask].copy()      # Clustered/non-isolated hits
    isolated_hits_df     = df_hits[noise_mask].copy()       # Noisy/isolated hits

    if non_isolated_hits_df.empty:
        return pd.DataFrame(columns=list(df_soph.columns) + ['E_hit_pe'])

    # Find the Z-span for the main track of each event
    z_ranges = non_isolated_hits_df.groupby('event')['Z'].agg(['min', 'max']).rename(columns={'min': 'Z_min', 'max': 'Z_max'})
    # Determine the total energy from isolated hits that fall within their event's Z-span
    energy_to_add = (
                        isolated_hits_df
                        .merge(z_ranges, on='event', how='left')
                        .dropna(subset=['Z_min', 'Z_max'])
                        .query("Z >= Z_min and Z <= Z_max")
                        .groupby('event')[energy_column].sum().rename('E_iso_to_add')
                    )
    
    # STEP B: Redistribute the "good" isolated energy to clustered/non-isolated hits
    if not energy_to_add.empty:
        non_isolated_hits_df = non_isolated_hits_df.merge(energy_to_add, on='event', how='left')
        non_isolated_hits_df['E_iso_to_add'].fillna(0, inplace=True)
        # Calculate the total energy of the main track for proportional scaling
        total_non_iso_energy = non_isolated_hits_df.groupby('event')[ energy_column].transform('sum').replace(0, 1)
        # Redistribution formula
        non_isolated_hits_df['E_hit_pe'] = (non_isolated_hits_df[energy_column] + 
                                           (non_isolated_hits_df[energy_column] / total_non_iso_energy) * non_isolated_hits_df['E_iso_to_add'])
    else:
        non_isolated_hits_df['E_hit_pe'] = non_isolated_hits_df[energy_column]

    # Drop additional columns
    non_isolated_hits_df.drop(columns=['E_iso_to_add'], inplace=True, errors='ignore')

    return non_isolated_hits_df

# def deal_spurious_hits(
#                         df_soph: pd.DataFrame,
#                         cluster_config: dict
#                      ) -> pd.DataFrame:
#     """
#     Processes a DataFrame of hits to filter noise and conserve energy.

#     The process involves three main steps:
#     1.  Tags all hits as either "isolated" or "non-isolated" using a 3D clustering algorithm.
#     2.  Identifies which "isolated" hits are physically plausible by checking if they fall
#         within the Z-span of their corresponding non-isolated event track.
#     3.  Redistributes the energy of these plausible isolated hits proportionally among the
#         non-isolated hits of the same event.

#     Args:
#         df_soph (pd.DataFrame): DataFrame at hit-level.
#         cluster_config (dict): A dictionary with parameters for the clustering algorithm,
#                                e.g., {'distance': [16., 16., 4.], 'nhit': 5}.

#     Returns:
#         pd.DataFrame: A DataFrame containing only the final, processed non-isolated hits,
#                       with an added 'E_hit_pe' column containing the redistributed energy per hit.
#     """
#     if df_soph.empty:
#         return pd.DataFrame(columns=list(df_soph.columns) + ['E_hit_pe'])

#     # STEP A: Tag all hits as isolated or non.isolated
#     splitter = split_isolated_clusters_3D(**cluster_config)
#     isolated_hits_indices = []
#     for _, group in df_soph.groupby(['event', 'npeak']):
#         if group.empty: continue
#         _, isolated_df = splitter(group)
#         if not isolated_df.empty:
#             isolated_hits_indices.append(isolated_df.index)
#     # Tagging
#     df_soph['is_isolated'] = False
#     if isolated_hits_indices:
#         iso_indices = np.concatenate(isolated_hits_indices)
#         df_soph.loc[iso_indices, 'is_isolated'] = True

#     # STEP B: Identify "good" isolated hits
#     non_isolated_hits_df = df_soph[~df_soph['is_isolated']].copy()
#     isolated_hits_df     = df_soph[df_soph['is_isolated']].copy()
#     if non_isolated_hits_df.empty:
#         return pd.DataFrame(columns=list(df_soph.columns) + ['E_hit_pe'])

#     # Find the Z-span for the main track of each event
#     z_ranges = non_isolated_hits_df.groupby('event')['Z'].agg(['min', 'max']).rename(columns={'min': 'Z_min', 'max': 'Z_max'})
#     # Determine the total energy from isolated hits that fall within their event's Z-span
#     energy_to_add = (
#                         isolated_hits_df
#                         .merge(z_ranges, on='event', how='left')
#                         .dropna(subset=['Z_min', 'Z_max'])
#                         .query("Z >= Z_min and Z <= Z_max")
#                         .groupby('event')['E_corr'].sum().rename('E_iso_to_add')
#                     )
    
#     # STEP C: Redistribute the "good" isolated energy to non-isolated hits
#     if not energy_to_add.empty:
#         non_isolated_hits_df = non_isolated_hits_df.merge(energy_to_add, on='event', how='left')
#         non_isolated_hits_df['E_iso_to_add'].fillna(0, inplace=True)
#         # Calculate the total energy of the main track for proportional scaling
#         total_non_iso_energy = non_isolated_hits_df.groupby('event')['E_corr'].transform('sum').replace(0, 1)
#         # Redistribution formula
#         non_isolated_hits_df['E_hit_pe'] = (non_isolated_hits_df['E_corr'] + 
#                                               (non_isolated_hits_df['E_corr'] / total_non_iso_energy) * non_isolated_hits_df['E_iso_to_add'])
#     else:
#         non_isolated_hits_df['E_hit_pe'] = non_isolated_hits_df['E_corr']

#     # Drop additional columns
#     non_isolated_hits_df.drop(columns=['is_isolated', 'E_iso_to_add'], inplace=True, errors='ignore')

#     return non_isolated_hits_df

# The following functions have been part of the creation for deal_spurious_hits

def drop_isolated_clusters_2D(
                                distance=[15., 15.],
                                nhit=3,
                                variables=['Ec']
                              ):
    """
    Drops rogue/isolated hits (SiPMs) from a groupedby dataframe.

    Parameters
    ----------
    df      : GroupBy 'event' dataframe ---> for inner function

    Initialization parameters:
        distance  : Distance to check for other sensors. Usually equal to sensor pitch.
        variables : List with variables to be redistributed.

    Returns
    -------
    pass_df : hits after removing isolated clusters
    """
    dist = np.sqrt(distance[0] ** 2 + distance[1] ** 2)

    def drop_event(df : pd.DataFrame) -> pd.DataFrame:
        x       = df.X.values
        y       = df.Y.values
        xy      = np.column_stack((x,y))
        dr2     = cdist(xy, xy)                 # Compute the distance between all hits

        if not np.any(dr2>0):
            return df.iloc[:0]                  # Empty dataframe

        closest = np.apply_along_axis(lambda d: len(d[d < dist]), 1, dr2)       # Number of neighbours
        mask_xy = closest > nhit
        pass_df = df.loc[mask_xy, :].copy()
        # isol_df = df.loc[~mask_xy, :].copy()

        # Variable redistribution: new hit weighted
        with np.errstate(divide='ignore'):
            columns = pass_df.loc[:, variables]
            columns *= np.divide(df.loc[:,variables].sum().values, columns.sum())
            pass_df.loc[:, variables] = columns

        return pass_df #, isol_df

    return drop_event

def drop_isolated_clusters_3D(
                                distance=[16., 16., 4.],
                                nhit=3,
                                variables=['Ec']
                            ):
    '''
    Drops isolated clusters of hits (SiPMs).

    Parameters
    ----------
    df       : Groupby ('event' and 'npeak') dataframe

    Initialisation parameters:
        distance  : Distance to check for other sensors, equal to sensor pitch and z rebinning.
        nhits     : Number of hits to classify a cluster.
        variables : List of variables to be redistributed (generally the energies).

    Returns
    -------
    pass_df : hits after removing isolated clusters
    """
    '''
    def drop_event(df: pd.DataFrame) -> pd.DataFrame:

        if len(df) == 0:
            return df

        # Normalise distances and (x,y,z) array
        x   = df.X.values / distance[0]
        y   = df.Y.values / distance[1]
        z   = df.Z.values / distance[2]
        xyz = np.column_stack((x,y,z))

        # Normalised, so define distance sqrt(3)
        dist = np.sqrt(3)
        
        # Use NearestNeighbors to find neighbors within the specified radius
        try:
            nbrs = NearestNeighbors(radius=dist, algorithm='ball_tree').fit(xyz)
            neighbors = nbrs.radius_neighbors(xyz, return_distance=False)
            mask = np.array([len(neigh) > nhit for neigh in neighbors])
        except Exception as e:
            print(f"Error in NearestNeighbors: {{e}}")
            return df.iloc[:0]  # fallback: return empty

        pass_df = df.loc[mask].copy()

        if not pass_df.empty and variables:
            with np.errstate(divide='ignore', invalid='ignore'):
                columns = pass_df.loc[:, variables]
                scale = df[variables].sum().values / columns.sum().values
                columns *= scale
                pass_df.loc[:, variables] = columns

        return pass_df

    return drop_event

def drop_hits_under_Q_threshold(
                                    Q_threshold=7,
                                    variables=['Ec']
                                ):
    '''
    Drops hits (SiPMs) below of a certain charge threshold.

    Parameters
    ----------
    df       : Groupby ('event' and 'npeak') dataframe

    Initialisation parameters:
        Q_threshold : Threshold of SiPM charge.
        variables   : List of variables to be redistributed (generally the energies).

    Returns
    -------
    pass_df : hits after removing those under threshold
    """
    '''
    def drop_hits(df: pd.DataFrame) -> pd.DataFrame:

        if len(df) == 0:
            return df
        
        pass_df = df[df['Q'] >= Q_threshold].copy()

        if not pass_df.empty and variables:
            with np.errstate(divide='ignore', invalid='ignore'):
                columns = pass_df.loc[:, variables]
                scale = df[variables].sum().values / columns.sum().values
                columns *= scale
                pass_df.loc[:, variables] = columns        

        return pass_df

    return drop_hits



# ----- Event-level ----- #

def cathode_position(
                        run_info,
                        run_data,
                        id=False,
                        n_bins=80,
                        step_back=0,
                        verbose=True
                    ):
    """
    Determines the stopping cathode time position (DT_stop) for a given run.

    Parameters:
        run_info (dict or int): Run details containing "run_number" or the run ID directly if `id=True`.
        run_data (dict of pd.DataFrame): Dictionary mapping run numbers to their corresponding DataFrames.
        id (bool): If True, treats `run_info` as the run ID. Default is False.
        n_bins (int): Number of bins for the histogram. Default is 80.
        step_back (int): Number of bins to step back from the cathode peak. Default is 0.
        verbose (bool): If True, prints debug messages. Default is True.

    Returns:
        dict: A dictionary with the run number as the key and DT_stop as the value.
    """
    # Determine run ID from input
    run_id = run_info if id else run_info.get("run_number", None)
    if run_id is None:
        raise ValueError("run_info must contain a 'run_number' key or be the run ID if `id=True`.")
    
    # Initialize dictionary to store DT_stop for the run
    DT_stop = {}
    
    try:
        # Extract drift time (DT) data for the run
        DT = run_data[run_id]['DT']
        if DT.empty:
            if verbose:
                print(f"Warning: No valid DT data for run {run_id}. Skipping...")
            DT_stop[run_id] = None
            return DT_stop

        # Compute histogram of DT, ignoring negative values
        counts, bins = np.histogram(DT, bins=n_bins, range=(0, DT.max()))
        # Identify the bin with the highest count (cathode peak)
        cath_index = np.argmax(counts)

        # Validate step_back and calculate DT_stop
        if cath_index - step_back < 0:
            if verbose:
                print(f"Warning: Step back exceeds valid range for run {run_id}. Using highest bin edge.")
            DT_stop[run_id] = bins[cath_index]
        else:
            DT_stop[run_id] = bins[cath_index - step_back]
            
        # Debug output if verbose
        if verbose:
            print(f"Run {run_id}: DT_stop = {DT_stop[run_id]:.2f} μs")

    except KeyError:
        # Handle missing run data
        if verbose:
            print(f"Error: Run {run_id} not found in run_data.")
        DT_stop[run_id] = None

    except Exception as e:
        # Handle unexpected errors
        if verbose:
            print(f"Error processing run {run_id}: {e}")
        DT_stop[run_id] = None
            
    return DT_stop