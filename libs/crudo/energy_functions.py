#
# Energy Functions: All the tools related to energy quantities in NEXT that you might need.
#
from . import plotting_tools as pt
from . import utilities      as ut

from invisible_cities.core.core_functions import in_range
from invisible_cities.reco.corrections import read_maps, apply_all_correction
from invisible_cities.types.symbols import NormMethod
from invisible_cities.types.symbols import NormStrategy
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.interpolate import interp1d

# ----- Custom Functions ----- #

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
    It hands NaN values in the S1e column by propagating them to the corrected column.

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
                        DT_cath, 
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
    X, Y, DT, E2 = df['X'], df['Y'], df['DT'], df[input_column]

    # Mask
    mask = (DT < DT_cath)

    # Generate normalized energy map
    energy_map, x_edges, y_edges = pt.mapping(X[mask], Y[mask], wei=E2[mask], xy_bins=xy_bins, norm=True)

    # Map events to reference bins, ensuring valid indices
    df['x_bin'] = np.clip(np.digitize(X, x_edges) - 1, 0, len(x_edges) - 2)
    df['y_bin'] = np.clip(np.digitize(Y, y_edges) - 1, 0, len(y_edges) - 2)

    # # Assign bin indices for each event
    # df['x_bin'] = np.digitize(X, x_edges) - 1  # 0-based indexing
    # df['y_bin'] = np.digitize(Y, y_edges) - 1

    # Normalization factors
    df['S2e_norm_factor'] = energy_map[df['x_bin'], df['y_bin']]
    # Handle bins with no data in the reference map
    df['S2e_norm_factor'] = np.where(df['S2e_norm_factor'] == 0, 1, df['S2e_norm_factor'])

    # Apply energy correction
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

# ----- Krypton Functions ----- #
# Custom functions that are not in IC, but Krypton people worked on them
# Sources: https://github.com/mcidlaso/IC/tree/ICAROS_3D/invisible_cities/icaros
#          https://gist.github.com/gonzaponte/3dfff1a615af2070aa915c6d315b7bc0

def define_kr_normalization(
                                krmap     : pd.DataFrame,
                                method    : NormMethod,
                                xy_params : dict = None
                            ) -> float:
    """
    Normalizes a krypton map according to a specific method.

    Given a krypton map, adjusts the map values to be normalized 
    based on the selected normalization method.

    Parameters:
        krmap : pd.DataFrame
            Krypton map to be normalized.
        method : NormMethod
            Normalization method to use, defined in the NormMethod class.
        xy_params : dict
            X and Y limits defining the region for normalization.

    Returns:
        float: Normalization value calculated based on the selected method.
    """

    krmap = krmap.dropna(subset=['mu'])
    anode = krmap[krmap.k == 0]

    if method is NormMethod.maximum:
        E_reference_max = krmap.mu.max()
        return E_reference_max

    if method is NormMethod.mean_chamber:
        E_reference_chamber = krmap.mu.mean()
        return E_reference_chamber

    if method is NormMethod.median_chamber:
        E_median_chamber = krmap.mu.median()
        return E_median_chamber

    if method is NormMethod.mean_anode:
        E_reference_anode = anode.mu.mean()
        return E_reference_anode

    if method is NormMethod.median_anode:
        E_median_anode = anode.mu.median()
        return E_median_anode

    mask_region = ( in_range(krmap.x, xy_params['x_low'], xy_params['x_high']) &
                    in_range(krmap.y, xy_params['y_low'], xy_params['y_high'])
                   ).values

    krmap = krmap[mask_region]

    if method is NormMethod.mean_region_chamber:
        E_reference_region = krmap.mu.mean()
        return E_reference_region

    if method is NormMethod.median_region_chamber:
        E_median_region = krmap.mu.median()
        return E_median_region

    anode = krmap[krmap.k == 0]

    if method is NormMethod.mean_region_anode:
        E_reference_slice_anode = anode.mu.mean()
        return E_reference_slice_anode

    if method is NormMethod.median_region_anode:
        E_median_region_anode = anode.mu.median()
        return E_median_region_anode

def get_corr3d(
                    kr_fname    : str,
                    norm_method : NormMethod,
                    xy_params   : dict = None,
                    mev_units   : bool = False
                ):
    """
    Generates a 3D correction function for energy calibration using a krypton map.

    Parameters:
        kr_fname (str): Path to the krypton map file (HDF5 format).
        norm_method (NormMethod): Normalization method to be used for the krypton map.
        xy_params (dict): Optional parameters defining the XY region for normalization.

    Returns:
        function: A correction function that takes drift time (dt), x, and y as inputs.
    """
    krmap = pd.read_hdf(kr_fname, key='krmap/krmap')    # krmap/krmap
    krmap = krmap.loc[~krmap['mu'].isna()]
    krmap = krmap.loc[krmap['mu'] > 0]
    dtxy_map = krmap.loc[:, ['dt', 'x', 'y']].values
    if mev_units:
        norm = 0.04155  # Kr conversion factor from [pe] to [MeV]
        # print("Using fixed normalization for MeV units: ", norm)
    else:
        norm = define_kr_normalization(krmap, norm_method, xy_params)
        
    
    def corr(dt, x, y):
        dtxy_input = np.stack([dt, x, y], axis=1)
        e_data = griddata(dtxy_map, krmap['mu'].values, dtxy_input, method='nearest')
        return norm / e_data

    return corr


def get_corrt(kr_fname, variable='s2e', n=4):
    """
    Creates a time-dependent correction function from the krypton map's time evolution.

    Parameters:
        kr_fname (str): Path to the krypton map file.
        variable (str): Variable for time correction. Default is 's2e'.
        n (int): Smoothing parameter. Default is 4.

    Returns:
        function: Interpolation function for time correction.
    """
    time_data = pd.read_hdf(kr_fname, key="t_evol/t_evol")

    smoothed = ut.smooth(time_data[variable], n)
    corr = smoothed.min() / smoothed

    time_correction = interp1d(time_data['ts'], corr, kind="cubic", bounds_error=False, fill_value=(corr[0], corr[-1]))

    return time_correction

def correct_energy_by_kr_map(
                                df         : pd.DataFrame,
                                kr_fname   : str,
                                norm_method: NormMethod,
                                city       : str = 'zemrude',
                                mev_units  : bool = False,
                                output_col : str = 'Ec'
                            ) -> pd.DataFrame:
    """
    Applies energy correction using a krypton map and time evolution correction.

    This function uses a krypton map to apply spatial energy corrections based on 
    drift time (DT), X, and Y coordinates. Additionally, it applies a time-dependent 
    correction to account for temporal variations in energy calibration.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data to be corrected. 
                           Must include columns 'E', 'DT', 'X', 'Y', and 'time'.
        kr_fname (str): Path to the krypton map file (HDF5 format).
        norm_method (NormMethod): Normalization method to be used for the krypton map.

    Returns:
        pd.DataFrame: DataFrame with a new column 'E_corr_pe' containing the corrected energy.
    """
    if city == 'zemrude':
        # Get 3D spatial correction function
        corr3d_func = get_corr3d(kr_fname, norm_method=norm_method, mev_units=mev_units)
        # Get time-dependent correction function
        corrt_func = get_corrt(kr_fname)

        # Apply corrections
        df[output_col] = df['E'] * corr3d_func(df['DT'], df['X'], df['Y']) * corrt_func(df['time'])
        # df['E_corr_pe'] = df['E'] * corr3d_func(df['DT'], df['X'], df['Y'])

    elif city == 'icaros':
        cmap = read_maps(kr_fname)
        corr_func = apply_all_correction(cmap, apply_temp=True, norm_strat=NormStrategy.max)
        x_vals, y_vals, z_vals, t_vals = df.X.values, df.Y.values, df.Z.values, df.time.values
        
        df['corr_factor'] = corr_func(x_vals, y_vals, z_vals, t_vals)
        df[output_col] = df['E'] * df['corr_factor']

    # Replace NaN or negative corrected energy with 0
    df[output_col] = np.where(pd.notna(df[output_col]) & (df[output_col] > 0), df[output_col], 0)

    return df


# def correct_energy_by_map(df: pd.DataFrame, cmap) -> pd.DataFrame:
#     """
#     Applies energy correction from Kr map and cleans negative/NaN values.

#     NOTE:   This function is outdated and should not be used in production.
#             It uses a previous version of Kr map correction (Icaros). 
#             It is kept here for reference purposes only.
#     """
#     corr_func = apply_all_correction(cmap, apply_temp=True, norm_strat=NormStrategy.max)
#     x_vals, y_vals, z_vals, t_vals = df.X.values, df.Y.values, df.Z.values, df.time.values
    
#     df['corr_factor'] = corr_func(x_vals, y_vals, z_vals, t_vals)
#     df['E_corr'] = df['E'] * df['corr_factor']
    
#     # NaN or negative energy to 0: hit-level
#     df['E_corr'] = np.where(pd.notna(df['E_corr']) & (df['E_corr'] > 0), df['E_corr'], 0)
    
#     return df

# ----- High Energy Functions ----- #
def energy_pe_to_mev(  df: pd.DataFrame
                     , slope: float
                     , intercept: float
                     , input_column:  str = 'E_evt_pe'
                     , output_column: str = 'E_evt_mev' ) -> pd.DataFrame:
    """
    Converts energy from photoelectrons (pe) to mega-electronvolts (MeV).
    Use a linear model conversion (for example, HE energy scale).
    """
    df[output_column] = slope * df[input_column] + intercept
    return df