import numpy  as np
from   scipy.special  import erf


##############################
# ----- Statistic Help ----- #
##############################


def efficiency(ok, lost):
    """
    Calculate the efficiency and its error based on the number of ok and lost events.
    Here we assume that N (total number of events) is known with zero error.

    Parameters:
        ok (int): Number of ok events.
        lost (int): Number of lost events.

    Returns:
        tuple: Efficiency and its error.
    """
    # Total number of events
    N = ok + lost

    # Error verbose for invalid inputs
    if ok < 0 or lost < 0:
        raise ValueError("The number of 'ok' and 'lost' events must be non-negative.")
    if N == 0:
        raise ValueError("The total number of events (N) must be greater than 0.")

    # Efficiency calculation
    efficiency = ok / N
  
    # Error calculation using the formula for binomial distribution
    error = np.sqrt(ok * (1 - (ok / N))) / N

    return efficiency, error

def chi2_value(data, model, sigma, dof=None):
    """
    Calculate the reduced chi-squared value for a fit.
    If no dof is given, returns the chi-squared (non-reduced) value.

    Parameters:
        data (array_like): The observed data.
        model (array_like): The model data.
        sigma(array_like): The uncertainty in the data.
        dof (int): Degrees of freedom (optional). If None, the function returns the non-reduced chi-squared value.

    Returns:
        chi_sq (float): The reduced chi-squared value if dof is provided, otherwise the non-reduced chi-squared value.
    """
    # Chi squared calculation from the method of least squares
    chi_sq = np.sum((data - model)**2 / sigma**2)

    # If dof is not provided, return the non-reduced chi-squared value
    if dof is None:
        return chi_sq
    else:
        nu = len(data) - dof     # Number of data points minus the number of fit parameters
        return chi_sq / nu

def gauss_int_err(A, mu, sigma, x_low, x_up, param_names, minuit_object):
    """
    Calculates the definite integral of a Gaussian function and propagates the error
    using the covariance matrix from the Minuit object.

    Args:
        A (float): Amplitude of the Gaussian.
        mu (float): Mean of the Gaussian.
        sigma (float): Standard deviation of the Gaussian.
        x_low (float): Lower limit of the integration.
        x_up (float): Upper limit of the integration.
        minuit_object (Minuit): Minuit object containing the fit and covariance matrix.
        param_names (list): List of parameter names for the Gaussian (e.g., ['A1', 'mu1', 'sigma1']).

    Returns:
        tuple: A tuple with the definite integral and its error.
    """
    # Preliminary
    a = x_low; b = x_up
    z_a = (a - mu) / (sigma * np.sqrt(2))
    z_b = (b - mu) / (sigma * np.sqrt(2))

    # Calculate the definite integral of the Gaussian function
    integral_CV = (np.sqrt(2 * np.pi) / 2) * A * sigma * (erf(z_b) - erf(z_a))

    # Calculate partial derivatives
    dI_dA = (np.sqrt(2 * np.pi) / 2) * sigma * (erf(z_b) - erf(z_a))
    dI_dmu = - A * (np.exp(-z_b**2) - np.exp(-z_a**2))
    dI_dsigma = A * (np.sqrt(2 * np.pi) / 2) * (erf(z_b) - erf(z_a)) - A * (z_b * np.exp(-z_b**2) - z_a * np.exp(-z_a**2))

    # Get the covariance matrix from the Minuit object
    cov_matrix = minuit_object.covariance
    idx_A     = minuit_object.parameters.index(param_names[0])
    idx_mu    = minuit_object.parameters.index(param_names[1])
    idx_sigma = minuit_object.parameters.index(param_names[2])

    # Calculate the variance of the integral using the covariance matrix
    integral_var = (dI_dA**2 * cov_matrix[idx_A][idx_A] +
                    dI_dmu**2 * cov_matrix[idx_mu][idx_mu] +
                    dI_dsigma**2 * cov_matrix[idx_sigma][idx_sigma] +
                    2 * dI_dA * dI_dmu * cov_matrix[idx_A][idx_mu] +
                    2 * dI_dA * dI_dsigma * cov_matrix[idx_A][idx_sigma] +
                    2 * dI_dmu * dI_dsigma * cov_matrix[idx_mu][idx_sigma])

    integral_err = np.sqrt(integral_var)

    return integral_CV, integral_err


#############################
# ----- Priors to Fit ----- #
#############################


# def outdated_prefit_1D(x_edges, y_counts, y_errors=None, x_min=None, x_max=None): 
#     """
#     Compute the bin centers, counts, and optional errors for preliminary 1D fits.
#     This function is useful for preparing data for fitting by filtering and summarizing
#     counts and errors within specified x-ranges.

#     Parameters:
#         x_edges (array-like): Bin edges for the x-values (e.g., from `np.histogram` or `np.linspace`).
#         y_counts (array-like): Array of counts or summed y-values for each bin.
#         y_errors (array-like, optional): Array of errors corresponding to y_counts. Default is None (no errors).
#         x_min (float, optional): Minimum x-value to include in the result. Default is None (no lower limit).
#         x_max (float, optional): Maximum x-value to include in the result. Default is None (no upper limit).

#     Returns:
#         tuple:
#             x_centers (np.ndarray): Centers of bins that contain valid data.
#             y_counts (np.ndarray): Total counts or summed y-values for each bin within the specified range.
#             y_errors (np.ndarray): Errors corresponding to y_counts, if provided, within the specified range.
#     """
#     # Ensure inputs are NumPy arrays for easier handling
#     x_edges  = np.asarray(x_edges)
#     y_counts = np.asarray(y_counts)
#     y_errors = np.asarray(y_errors) if y_errors is not None else None

#     # Calculate the bin centers
#     x_centers = 0.5 * (x_edges[1:] + x_edges[:-1])
#     filtered_counts = []
#     filtered_errors = []

#     # Loop over bins to filter data based on x_min and x_max
#     for i in range(len(x_edges) - 1):
#         # Check if the bin center is within the specified range
#         if (x_min is None or x_centers[i] >= x_min) and (x_max is None or x_centers[i] <= x_max):

#             counts = y_counts[i]
#             if counts == 0:
#                 # Skip empty bins
#                 filtered_counts.append(np.nan)
#                 filtered_errors.append(np.nan)
#                 continue
#             filtered_counts.append(y_counts[i])
#             filtered_errors.append(y_errors[i] if y_errors is not None else 0)  # Append error if provided
#         else:
#             # Mark as NaN if outside the specified range
#             filtered_counts.append(np.nan)
#             filtered_errors.append(np.nan)

#     # Convert lists to NumPy arrays
#     filtered_counts = np.array(filtered_counts)
#     filtered_errors = np.array(filtered_errors)

#     # Remove NaN values
#     valid_mask = ~np.isnan(filtered_counts)
#     x_centers = x_centers[valid_mask]
#     filtered_counts = filtered_counts[valid_mask]
#     filtered_errors = filtered_errors[valid_mask]

#     return x_centers, filtered_counts, filtered_errors

# def outdated_prefit_2D(x_edges, x_values, y_values, x_min=None, x_max=None, y_min=None, y_max=None):
#     """
#     Compute the median and error of y-values within bins of x-values, with optional filtering by x and y ranges.
#     This function is useful for preparing data for 2D fits by summarizing y-values within x-bins
#     and applying optional range filters to exclude unwanted data.

#     Parameters:
#         x_edges (array-like): Bin edges for x-values (e.g., from `np.histogram` or `np.linspace`).
#         x_values (array-like): Array of x-values to bin.
#         y_values (array-like): Array of y-values corresponding to x-values.
#         x_min (float, optional): Minimum x-value to include in the result. Default is None (no lower limit).
#         x_max (float, optional): Maximum x-value to include in the result. Default is None (no upper limit).
#         y_min (float, optional): Minimum y-value for filtering medians. Default is None (no lower limit).
#         y_max (float, optional): Maximum y-value for filtering medians. Default is None (no upper limit).

#     Returns:
#         tuple:
#             x_centers (np.ndarray): Centers of bins with valid data, filtered by x_min and x_max.
#             y_medians (np.ndarray): Median y-values for each bin, filtered by y_min and y_max.
#             y_errors (np.ndarray): Errors associated with the median y-values for each bin.
#     """
#     # Ensure inputs are NumPy arrays for consistent handling
#     x_edges  = np.asarray(x_edges)
#     x_values = np.asarray(x_values)
#     y_values = np.asarray(y_values)

#     # Calculate the bin centers from the edges
#     x_centers = 0.5 * (x_edges[1:] + x_edges[:-1])
#     y_medians = []
#     y_errors  = []

#     # Loop over bins to compute medians and errors
#     for i in range(len(x_edges) - 1):
#         # Select data points that fall within the current bin
#         in_bin = (x_values >= x_edges[i]) & (x_values < x_edges[i + 1])

#         # Skip bins outside the specified x-range
#         if (x_min is not None and x_centers[i] < x_min) or (x_max is not None and x_centers[i] > x_max):
#             y_medians.append(np.nan)
#             y_errors.append(np.nan)
#             continue

#         # Compute the median and error if the bin contains data
#         if np.sum(in_bin) > 0:
#             # Calculate the median of y-values in the bin
#             median = np.median(y_values[in_bin])
#             # print(f'y values = {y_values[in_bin]}')
#             # Apply optional filtering by y_min and y_max
#             if (y_min is not None and median < y_min) or (y_max is not None and median > y_max):
#                 y_medians.append(np.nan)
#                 y_errors.append(np.nan)
#             else:
#                 # Estimate the error using the scaled standard deviation
#                 error  = 1.253 * np.std(y_values[in_bin]) / np.sqrt(np.sum(in_bin))
#                 y_medians.append(median)
#                 y_errors.append(error)
#         else:
#             # Mark as NaN if the bin is empty
#             y_medians.append(np.nan)
#             y_errors.append(np.nan)

#     # Convert lists to NumPy arrays for output
#     y_medians = np.array(y_medians)
#     y_errors  = np.array(y_errors)

#     # Remove NaN values to return only valid data
#     valid_mask = ~np.isnan(y_medians)
#     x_centers = x_centers[valid_mask]
#     y_medians = y_medians[valid_mask]
#     y_errors  = y_errors[valid_mask]

#     return x_centers, y_medians, y_errors

def prefit_1D(x_data, bins): 
    """
    Compute the bin centers, counts, and optional errors for preliminary 1D fits.
    This function is useful for preparing data for fitting by filtering and summarizing
    counts and errors within specified x-ranges.

    Parameters:
        x_edges (array-like): Bin edges for the x-values (e.g., from `np.histogram` or `np.linspace`).
        y_counts (array-like): Array of counts or summed y-values for each bin.
        y_errors (array-like, optional): Array of errors corresponding to y_counts. Default is None (no errors).
        x_min (float, optional): Minimum x-value to include in the result. Default is None (no lower limit).
        x_max (float, optional): Maximum x-value to include in the result. Default is None (no upper limit).

    Returns:
        tuple:
            x_centers (np.ndarray): Centers of bins that contain valid data.
            y_counts (np.ndarray): Total counts or summed y-values for each bin within the specified range.
            y_errors (np.ndarray): Errors corresponding to y_counts, if provided, within the specified range.
    """
    if not isinstance(bins, (list, np.ndarray)):
        # If 'bins' is a number, create bin edges
        bin_edges = np.linspace(x_data.min(), x_data.max(), bins + 1)
    else:
        bin_edges = bins

    x_centers = [];     x_errors = []
    y_counts = [];      y_errors = []

    for i in range(len(bin_edges) - 1):
        low_edge, high_edge = bin_edges[i], bin_edges[i + 1]

        # Mask to select events within the current x bin
        mask_in_bin = (x_data >= low_edge) & (x_data < high_edge)
        counts_in_bin = x_data[mask_in_bin]

        # Skip bins with no events
        if len(counts_in_bin) == 0:
            continue

        # Calculate bin center and x error
        x_centers.append((low_edge + high_edge) / 2)
        x_errors.append((high_edge - low_edge) / 2)

        # Calculate counts and its error
        y_counts.append(len(counts_in_bin))
        y_errors.append(np.sqrt(len(counts_in_bin)))  # Assuming Poisson statistics for counts
            
    return np.array(x_centers), np.array(x_errors), np.array(y_counts), np.array(y_errors)

def prefit_2D(x_data, y_data, bins):
    """
    Create a profile from 2D data, calculating the mean of y for each bin of x.

    Args:
        x_data (np.array): Data for the x-axis.
        y_data (np.array): Data for the y-axis.
        bins (int or np.array): Number of bins or the bin edges for the x-axis.

    Returns:
        tuple: (x_centers, y_means, y_errors, x_errors)
               - x_centers: Centers of bins with events.
               - y_means: Mean value of y for each bin.
               - y_errors: Error in the mean for each bin.
               - x_errors: Error in x (half the bin width).
    """
    if not isinstance(bins, (list, np.ndarray)):
        # If 'bins' is a number, create bin edges
        bin_edges = np.linspace(x_data.min(), x_data.max(), bins + 1)
    else:
        bin_edges = bins

    x_centers = [];     x_errors = []
    y_means = [];       y_errors = []

    for i in range(len(bin_edges) - 1):
        low_edge, high_edge = bin_edges[i], bin_edges[i + 1]

        # Mask to select events within the current x bin
        mask_in_bin = (x_data >= low_edge) & (x_data < high_edge)
        y_in_bin = y_data[mask_in_bin]

        # Skip bins with no events
        if len(y_in_bin) == 0:
            continue

        # Calculate bin center and x error
        x_centers.append((low_edge + high_edge) / 2)
        x_errors.append((high_edge - low_edge) / 2)

        # Calculate mean and its error
        mean = np.mean(y_in_bin)
        std_dev = np.std(y_in_bin, ddof=1)      # ddof=1 for sample standard deviation
        error = std_dev / np.sqrt(len(y_in_bin)) if len(y_in_bin) > 1 else 0

        y_means.append(mean)
        y_errors.append(error)

    return np.array(x_centers), np.array(x_errors), np.array(y_means), np.array(y_errors)

#############################
# ----- Fit Functions ----- #
#############################


def linear_func(x, m, b):
    """
    Linear function.

    Parameters:
        x (array-like): Independent variable.
        m (float): Slope of the line.
        b (float): Intercept of the line.

    Returns:
        np.ndarray: Linear values for x.
    """
    return m * x + b

def log_func(x, A, b):
    """
    Natural logarithm function.

    Parameters:
        x (array-like): Independent variable.
        A (float): Scaling factor.
        b (float): Offset.

    Returns:
        np.ndarray: Natural logarithm values for x.
    """
    return A * np.log(x) + b

def exponential_decay(x, N0, tau, b):
    """
    Exponential decay function with a linear offset.

    Parameters:
        x (array-like): Independent variable (e.g., time).
        N0 (float): Initial value at x = 0.
        tau (float): Decay constant.
        b (float): Linear offset.

    Returns:
        np.ndarray: Exponentially decaying values for x with an added offset.
    """
    return N0 * np.exp(-x / tau) + b

def gauss_func(x, A1, mu1, sigma1):
    """
    Single Gaussian function.

    Parameters:
        x (array-like): Independent variable.
        A1 (float): Amplitude of the Gaussian.
        mu1 (float): Mean (center) of the Gaussian.
        sigma1 (float): Standard deviation (width) of the Gaussian.

    Returns:
        np.ndarray: Gaussian values for x.
    """
    return A1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2))

def bigauss_func(x, A1, mu1, sigma1, A2, mu2, sigma2):
    """
    Model for two Gaussian distributions.

    Parameters:
        x (array-like): Independent variable.
        A1 (float): Amplitude of the first Gaussian.
        mu1 (float): Mean (center) of the first Gaussian.
        sigma1 (float): Standard deviation (width) of the first Gaussian.
        A2 (float): Amplitude of the second Gaussian.
        mu2 (float): Mean (center) of the second Gaussian.
        sigma2 (float): Standard deviation (width) of the second Gaussian.

    Returns:
        np.ndarray: Combined Gaussian values for x.
    """
    g1 = A1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2))
    g2 = A2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2))
    return g1 + g2

def trigauss_func(x, A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3):
    """
    Model for three Gaussian distributions.

    Parameters:
        x (array-like): Independent variable.
        A1 (float): Amplitude of the first Gaussian.
        mu1 (float): Mean (center) of the first Gaussian.
        sigma1 (float): Standard deviation (width) of the first Gaussian.
        A2 (float): Amplitude of the second Gaussian.
        mu2 (float): Mean (center) of the second Gaussian.
        sigma2 (float): Standard deviation (width) of the second Gaussian.
        A3 (float): Amplitude of the third Gaussian.
        mu3 (float): Mean (center) of the third Gaussian.
        sigma3 (float): Standard deviation (width) of the third Gaussian.

    Returns:
        np.ndarray: Combined Gaussian values for x.
    """
    g1 = A1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2))
    g2 = A2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2))
    g3 = A3 * np.exp(-((x - mu3) ** 2) / (2 * sigma3 ** 2))
    return g1 + g2 + g3