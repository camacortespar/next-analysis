#
# Utilities: Unclassified but helpful tools/functions.
#

from datetime import datetime
import locale
import numpy as np
from typing import Tuple




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
    locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')           # Adjust to your system locale if needed
    # Convert epoch time to a datetime object
    dt_object = datetime.fromtimestamp(epoch_time)
    # Use different formats based on the value of h
    if h:
        formatted_date = dt_object.strftime('%H:%M')
        # formatted_date = dt_object.strftime('%d/%m - %H:%M')  # day/month - hour:minute
    else:
        formatted_date = dt_object.strftime('%d/%m')          # day/month
    
    return formatted_date


def smooth(y, n=4):
    """
    Smooths a given 1D array by applying a moving average filter.

    This function extends the input array by flipping it on both ends, 
    applies a moving average convolution, and then extracts the smoothed 
    values corresponding to the original array length.

    Parameters:
    y (array-like): The input 1D array to be smoothed.
    n (int, optional): The window size for the moving average filter. 
                       Default is 4.

    Returns:
    numpy.ndarray: The smoothed 1D array with the same length as the input.
    """
    m  = len(y)
    yf = np.flip(y)
    y  = np.concatenate([yf, y, yf])
    z  = np.ones(n) / n
    y  = np.convolve(y, z, mode="same")
    return y[m:2*m]

def weighted_avg(series, weight):
    if weight.sum() == 0:   # Avoid division by zero
        return np.nan
    return np.average(series, weights=weight)

def R_max_func(group_df):
    return np.sqrt(group_df['X']**2 + group_df['Y']**2).max()

# ----- Background Index ----- #
def back_index(rate_hz: float, rate_err_hz: float, mass_kg: float, delta_E_mev: float) -> Tuple[float, float]:
    """
    Calculates the Background Index (BI) and its statistical error.
    
    The error propagation assumes that mass_kg and delta_E_mev are constants
    with no associated uncertainty.
    
    Parameters:
    - rate_hz: The central value of the rate in Hz.
    - rate_err_hz: The statistical uncertainty of the rate in Hz.
    - mass_kg: The fiducial mass in kg.
    - delta_E_mev: The energy window width in MeV.
    
    Returns:
    - A tuple containing:
        - bi_cv (float): The central value of the Background Index in counts/(keV·kg·yr).
        - bi_err (float): The propagated statistical error of the BI.
    """
    # Define constants
    SECONDS_PER_YEAR = 31536000.0    # Number of seconds in a year
    
    # Check for valid inputs to prevent division by zero
    if mass_kg <= 0 or delta_E_mev <= 0:
        raise ValueError("Mass and energy window must be positive values.")
        
    # --- Compute Central Value ---
    rate_in_year = rate_hz * SECONDS_PER_YEAR
    delta_E_keV = delta_E_mev * 1000.0
    bi_cv = rate_in_year / (mass_kg * delta_E_keV)
    
    # --- Propagate the Error ---
    rate_err_in_year = rate_err_hz * SECONDS_PER_YEAR
    bi_err = rate_err_in_year / (mass_kg * delta_E_keV)
    
    return bi_cv, bi_err