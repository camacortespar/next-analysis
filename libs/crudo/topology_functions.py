import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from typing import Callable




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

def hits_clusterizer( eps          : float
                    , min_samples  : float
                    , scale_xy     : float = 14.55
                    , scale_z      : float = 3.7
                    , event_column : str = 'event'
                    ) -> Callable:
    """
    Cluster hits in 3D space for each event using DBSCAN.
    The coordinates are scaled to account for detector geometry differences in samplig 
    
    Parameters
    ----------
    eps         : float, Epsilon value for DBSCAN.
    min_samples : int, Min Samples value for DBSCAN.
    scale_xy    : float, scale factor for XY coordinates.
    scale_z     : float, scale factor for Z coordinate.
    
    Returns
    -------
    Callable
    A function that takes a DataFrame of hits and returns the same DataFrame 
    with an added 'cluster' column, which are the clusters labels assigned by DBSCAN
    (-1 for noise).
    """
    def cluster_tagger(df_hits: pd.DataFrame) -> pd.DataFrame:
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
    
    return cluster_tagger