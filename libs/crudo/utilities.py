#
# Utilities: Unclassified but helpful tools/functions.
#

from datetime import datetime
import locale
import numpy as np



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
        # formatted_date = dt_object.strftime('%H:%M')
        formatted_date = dt_object.strftime('%d/%m - %H:%M')  # day/month - hour:minute
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