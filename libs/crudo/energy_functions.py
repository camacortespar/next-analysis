#
# Energy Functions: All the tools related to energy quantities in NEXT that you might need.
#
from . import plotting_tools     as pt

import numpy as np
import pandas as pd


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
    # if len(df) != len(mask):
    #     raise ValueError(f"Length mismatch: DataFrame has {len(df)} rows, but mask has {len(mask)}.")
    # # Ensure mask is boolean!
    # assert mask.dtype == bool, "Mask must be a boolean Series!"

    

    # Extract relevant columns
    X, Y, DT, E2 = df['X'], df['Y'], df['DT'], df[input_column]

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