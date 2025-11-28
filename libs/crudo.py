from   datetime import datetime
import locale
import numpy  as np
import os
import pandas as pd
from . import plotting_tools as pt
from   scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import NotFittedError
from typing import List, Callable


###############################
# ----- Data Management ----- #
###############################


def load_run_data(  
                    run_info, 
                    id=False, 
                    base_path="/lustre/ific.uv.es/prj/gl/neutrinos/users/ccortesp/NEXT-100/Sophronia/Alphas/", 
                    city="sophronia", 
                    trigger=2, 
                    key="/DST/Events", 
                    verbose=True
                ):
    """
    Loads HDF5 data for a specific run and returns a concatenated DataFrame.

    Parameters:
        run_info (dict or int): Run information containing "run_number" or the run ID directly if `id=True`.
        id (bool): If True, `run_info` is treated as the run ID. Default is False.
        base_path (str): Base directory for HDF5 files. Default is the specified path.
        city (str): Subdirectory name for the run files. Default is "sophronia".
        trigger (int or None): Trigger type (e.g., 2 for trg2). Default is 2. If None, no trigger is used.
        key (str): HDF5 dataset path to load. Default is "/DST/Events".
        verbose (bool): If True, prints debug messages. Default is True.

    Returns:
        dict: A dictionary with the run number as the key and the concatenated DataFrame as the value.
              Returns an empty DataFrame if no valid files are found.
    """
    # Determine run ID
    run_id = run_info if id else run_info.get("run_number", None)
    if run_id is None:
        raise ValueError("run_info must contain a 'run_number' key or be the run ID if `id=True`.")

    # Initialize storage for run data and valid file paths
    run_data = {}
    h5_files = []

    # Search for HDF5 files across LDCs (1 to 7)
    for ldc in range(1, 8):
        file_name = os.path.join(base_path, f"run_{run_id}_ldc{ldc}{'_trg2' if trigger == 2 else ''}_{city}.h5")
        if os.path.isfile(file_name):
            h5_files.append(file_name)
        elif verbose:
            print(f"Warning: File {file_name} does not exist, skipping...")

    # Return empty DataFrame if no files found
    if not h5_files:
        if verbose:
            print(f"Warning: No valid files found for run {run_id}.")
        run_data[run_id] = pd.DataFrame()
        return run_data

    # Load and concatenate data from valid files
    dataframes = []
    for file in h5_files:
        try:
            dataframes.append(pd.read_hdf(file, key=key))
        except Exception as e:
            if verbose:
                print(f"Error reading {file}: {e}")

    # Store concatenated DataFrame or empty DataFrame if no data loaded
    if dataframes:
        run_data[run_id] = pd.concat(dataframes, ignore_index=True)
        if verbose:
            print(f"{key}: Run {run_id} successfully loaded with data shape: {run_data[run_id].shape}")
    else:
        if verbose:
            print(f"Warning: No data loaded for run {run_id}.")
        run_data[run_id] = pd.DataFrame()

    return run_data

def filter_run_data(
                        run_info,
                        run_data, 
                        sel_criteria, 
                        id=False, 
                        verbose=True
                    ):
                    
    """
    Filters event-level data for a specific run based on custom criteria.

    Parameters:
        run_info (dict or int): Run information containing "run_number" or the run ID directly if `id=True`.
        run_data (dict of pd.DataFrame): Maps run numbers to their corresponding DataFrames.
        sel_criteria (callable): Function defining the filtering logic for events.        
        id (bool): If True, `run_info` is treated as the run ID. Default is False.
        verbose (bool): If True, prints debug messages. Default is True.

    Returns:
        dict: A dictionary with run numbers as keys and filtered DataFrames as values.
              Returns an empty DataFrame if filtering fails.
    """
    # Determine run ID
    run_id = run_info if id else run_info.get("run_number", None)
    if run_id is None:
        raise ValueError("run_info must contain a 'run_number' key or be the run ID if `id=True`.")

    # Initialize dictionary to store filtered data
    filtered_data = {}

    try:
        # Apply filtering criteria to the DataFrame grouped by 'event'
        filtered_data[run_id] = run_data[run_id].groupby('event').filter(sel_criteria)

        # Print success message if verbose is enabled
        if verbose:
            print(f"Run {run_id} filtered successfully. Data shape: {filtered_data[run_id].shape}")
    
    except Exception as e:
        # Handle errors and store an empty DataFrame for the run
        if verbose:
            print(f"Error filtering run {run_id}: {e}")
        filtered_data[run_id] = pd.DataFrame()

    return filtered_data

def merge_dfs(
                file1,
                file2,
                output_file
            ):
    """
    Merge two .pkl files containing dictionaries of DataFrames into a single output .pkl file.

    Parameters:
        file1 (str): Path to the first .pkl file.
        file2 (str): Path to the second .pkl file.
        output_file (str): Path to save the merged .pkl file.

    Returns:
        dict: Merged dictionary of DataFrames.
    """
    # Load the .pkl files
    try:
        df1 = pd.read_pickle(file1)
        df2 = pd.read_pickle(file2)
    except Exception as e:
        print(f"Error loading .pkl files: {e}")
        return None

    # Validate that the files contain dictionaries of DataFrames
    if not isinstance(df1, dict) or not isinstance(df2, dict):
        print("Error: Files must contain dictionaries of DataFrames.")
        return None

    # Merge the dictionaries
    merged_data = {**df1, **df2}

    # Save the merged dictionary to a .pkl file
    try:
        with open(output_file, "wb") as f:
            pd.to_pickle(merged_data, f)
        print(f"Merged data saved to: {output_file}")
    except Exception as e:
        print(f"Error saving merged data: {e}")

    return merged_data

def save_dataframes(
                        data,
                        output_path,
                        group_path=""
                    ):
    """
    Saves DataFrames from a nested dictionary into an HDF5 file.

    Parameters:
        data (dict): Nested dictionary containing DataFrames or other dictionaries.
        output_path (str): Path to the HDF5 file.
        group_path (str): HDF5 group path for nested keys. Default is "".

    Returns:
        None
    """
    for city, structure in data.items():
        # Replace '/' in keys to avoid conflicts in HDF5 paths
        save_key = str(city).replace('/', '_SLASH_')

        # Construct the HDF5 key path
        key = f"{group_path}/{save_key}" if group_path else save_key

        if isinstance(structure, pd.DataFrame):
            # Save non-empty DataFrames to HDF5
            if not structure.empty:
                try:
                    structure.to_hdf(
                        output_path,
                        key=key,
                        mode='a',           # Append mode to avoid overwriting
                        complevel=5,        # Moderate compression level
                        complib='blosc',    # Efficient compression library
                        format='table'      # Table format for flexibility
                    )
                    print(f"    Saved DataFrame to key: '{key}' {structure.shape}")
                except Exception as e:
                    print(f"  Error saving DataFrame to key '{key}': {e}")
            else:
                print(f"  Empty DataFrame for key '{key}': not saved.")
        elif isinstance(structure, dict):
            # Recursively process nested dictionaries
            save_dataframes(structure, output_path, key)


##############################
# ----- Analysis Tools ----- #
##############################


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


##################################
# ----- Energy Corrections ----- #
##################################


def correct_S1e(
                    df,
                    CV_fit,
                    DT_cath,
                    DT_column='DT',
                    S1e_column='S1e',
                    output_column='S1e_corr'
                ):
    """
    Corrects S1e values using a linear model with the cathode as the reference.

    Parameters:
        df (pd.DataFrame): DataFrame containing S1 energy and drift time data.
        CV_fit (tuple): Linear fit coefficients (slope, intercept).
        DT_cath (float): Reference drift time at the cathode.
        DT_column (str): Column name for drift time data. Default is 'DT'.
        S1e_column (str): Column name for S1 energy data. Default is 'S1e'.
        output_column (str): Column name for corrected S1 energy. Default is 'S1e_corr'.

    Returns:
        pd.DataFrame: DataFrame with a new column containing corrected S1e values.
    """
    # Unpack fit coefficients
    m, b = CV_fit

    # Validate required columns
    if DT_column not in df or S1e_column not in df:
        raise KeyError(f"Missing required columns: '{DT_column}' or '{S1e_column}'.")

    # Compute reference S1e value at the cathode
    S1e_ref = m * DT_cath + b

    # Apply correction to S1e values, handling NaN values
    df[output_column] = df[S1e_column] * (S1e_ref / (m * df[DT_column] + b))
    df.loc[df[S1e_column].isna(), output_column] = np.nan

    return df

def correct_S2e_LT(
                        df, 
                        LT_fit, 
                        DT_column='DT', 
                        S2e_column='S2e', 
                        output_column='S2e_corr_LT'
                    ):
    """
    Applies electron lifetime correction to S2 energy values using an exponential decay model.

    Parameters:
        df (pd.DataFrame): Input data containing drift time and S2 energy columns.
        LT_fit (tuple): Exponential fit coefficients (N0, tau).
        DT_column (str): Column name for drift time. Default is 'DT'.
        S2e_column (str): Column name for S2 energy. Default is 'S2e'.
        output_column (str): Column name for corrected S2 energy. Default is 'S2e_corr_LT'.

    Returns:
        pd.DataFrame: DataFrame with a new column containing lifetime-corrected S2 energy.
    """
    # Unpack fit coefficients
    N0, tau = LT_fit

    # Validate required columns
    if DT_column not in df or S2e_column not in df:
        raise KeyError(f"Missing required columns: '{DT_column}' or '{S2e_column}'.")

    # Apply lifetime correction
    df[output_column] = df[S2e_column] * np.exp(df[DT_column] / tau)

    return df

def correct_S2e_map(
                        df, 
                        mask, 
                        xy_bins=50, 
                        input_column='S2e_corr_LT'
                    ):
    """
    Corrects S2 energy using a radial energy map normalized to the center.

    Parameters:
        df (pd.DataFrame): DataFrame containing S2 energy and positions (X, Y).
        mask (pd.Series): Boolean mask to filter valid events for map generation.
        xy_bins (int): Number of bins for the XY map. Default is 50.
        input_column (str): Column name for the energy to be corrected. Default is 'S2e_corr_LT'.

    Returns:
        pd.DataFrame: DataFrame with a new column 'S2e_corr' containing corrected S2 energy.
    """
    # Extract relevant columns
    X, Y, E2 = df['X'], df['Y'], df[input_column]

    # Generate normalized energy map
    energy_map, x_edges, y_edges = pt.mapping(X[mask], Y[mask], wei=E2[mask], xy_bins=xy_bins, norm=True)

    # Assign bin indices for each event
    df['x_bin'] = np.digitize(X, x_edges) - 1  # 0-based indexing
    df['y_bin'] = np.digitize(Y, y_edges) - 1

    # Assign normalization factors and apply correction
    df['S2e_norm_factor'] = energy_map[df['x_bin'], df['y_bin']]
    df['S2e_corr'] = E2 / df['S2e_norm_factor']

    return df

def correct_S2e_map_fixed(
                            df, 
                            ref_Emap,
                            x_edges, 
                            y_edges, 
                            input_column='S2e_corr_LT'
                        ):
    """
    Corrects S2 energy using a fixed reference map with pre-defined bin edges.

    Parameters:
        df (pd.DataFrame): DataFrame with S2 energy and positions (X, Y).
        ref_Emap (np.ndarray): Precomputed reference energy map.
        x_edges (np.ndarray): X-axis bin edges from the reference map.
        y_edges (np.ndarray): Y-axis bin edges from the reference map.
        input_column (str): Column name for the energy to correct. Default is 'S2e_corr_LT'.

    Returns:
        pd.DataFrame: DataFrame with corrected S2 energy in 'S2e_corr'.
    """
    # Extract relevant columns
    X, Y, E2 = df['X'], df['Y'], df[input_column]

    # Map events to reference bins, ensuring valid indices
    df['x_bin'] = np.clip(np.digitize(X, x_edges) - 1, 0, len(x_edges) - 2)
    df['y_bin'] = np.clip(np.digitize(Y, y_edges) - 1, 0, len(y_edges) - 2)

    # Assign normalization factors from the reference map
    df['S2e_norm_factor'] = ref_Emap[df['x_bin'], df['y_bin']]

    # Handle bins with no data in the reference map
    df['S2e_norm_factor'] = np.where(df['S2e_norm_factor'] == 0, 1, df['S2e_norm_factor'])

    # Apply energy correction
    df['S2e_corr'] = E2 / df['S2e_norm_factor']

    return df


###########################
# ----- Extra Tools ----- #
###########################


def epoch_converter(epoch_time, h=False):
    """
    Converts epoch time to a formatted string in Spanish time.
    
    Parameters:
        epoch_time (float): The epoch time to convert.
        h (bool): If True, include the hour and minute in the format. Defaults to False.
        
    Returns:
        str: The formatted date string.
    """
    # Set locale to Spanish
    locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')           #adjust to your system locale if needed
    # Convert epoch time to a datetime object
    dt_object = datetime.fromtimestamp(epoch_time)
    # Use different formats based on the value of h
    if h:
        # formatted_date = dt_object.strftime('%H:%M')
        formatted_date = dt_object.strftime('%d/%m - %H:%M')  #day/month - hour:minute
    else:
        formatted_date = dt_object.strftime('%d/%m')          #day/month
    
    return formatted_date