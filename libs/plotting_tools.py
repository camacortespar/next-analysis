from   cycler import cycler                                         # type: ignore
import matplotlib.pyplot as plt                                     # type: ignore
from   matplotlib.colors import LinearSegmentedColormap             # type: ignore
from   matplotlib.offsetbox import (OffsetImage, AnnotationBbox)    # type: ignore
import numpy as np                                                  # type: ignore
import pandas as pd                                                 # type: ignore
from   PIL import Image                         # type: ignore


##########################################
# ----- NEXT Experiment Plot Style ----- #
##########################################


color_sequence = ("k", "m", "g", "b", "r",
                  "gray", "aqua", "gold", "lime", "purple",
                  "brown", "lawngreen", "tomato", "lightgray", "lightpink")

def auto_plot_style(overrides = dict()):
    plt.rcParams[ "figure.figsize"               ] = 10, 8
    plt.rcParams[   "font.size"                  ] = 25
    plt.rcParams[  "lines.markersize"            ] = 25
    plt.rcParams[  "lines.linewidth"             ] = 3
    plt.rcParams[  "patch.linewidth"             ] = 3
    plt.rcParams[   "axes.linewidth"             ] = 2
    plt.rcParams[   "grid.linewidth"             ] = 3
    plt.rcParams[   "grid.linestyle"             ] = "--"
    plt.rcParams[   "grid.alpha"                 ] = 0.5
    plt.rcParams["savefig.dpi"                   ] = 300
    plt.rcParams["savefig.bbox"                  ] = "tight"
    plt.rcParams[   "axes.formatter.use_mathtext"] = True
    plt.rcParams[   "axes.formatter.limits"      ] = (-3 ,4)
    plt.rcParams[  "xtick.major.size"            ] = 10
    plt.rcParams[  "ytick.major.size"            ] = 10
    plt.rcParams[  "xtick.minor.size"            ] = 5
    plt.rcParams[  "ytick.minor.size"            ] = 5
    plt.rcParams[   "axes.prop_cycle"            ] = cycler(color=color_sequence)
    plt.rcParams[  "image.cmap"                  ] = "gnuplot2"
    plt.rcParams.update(overrides)

def ccortesp_plot_style(overrides = dict()):
    plt.rcParams[ "figure.figsize"              ] = 10, 8
    plt.rcParams[ "font.size"                   ] = 15
    plt.rcParams[ "axes.formatter.use_mathtext" ] = True
    plt.rcParams[ "axes.formatter.limits"       ] = (-3 ,4)
    plt.rcParams[ "xtick.major.size"            ] = 10
    plt.rcParams[ "ytick.major.size"            ] = 10
    plt.rcParams[ "xtick.minor.size"            ] = 5
    plt.rcParams[ "ytick.minor.size"            ] = 5
    plt.rcParams[ "patch.linewidth"             ] = 3
    plt.rcParams[ "axes.linewidth"              ] = 2

    plt.rcParams[ "legend.frameon"              ] = True
    plt.rcParams[ "legend.edgecolor"            ] = "none"
    plt.rcParams[ "legend.facecolor"            ] = "white"
    
    plt.rcParams[ "grid.linewidth"              ] = 1
    plt.rcParams[ "grid.linestyle"              ] = "--"
    plt.rcParams[ "grid.alpha"                  ] = 0.5

    plt.rcParams[ "lines.markersize"            ] = 8
    plt.rcParams[ "lines.linewidth"             ] = 2
    
    plt.rcParams[ "savefig.dpi"                 ] = 300
    plt.rcParams[ "savefig.bbox"                ] = "tight"
    
    plt.rcParams[ "axes.prop_cycle"             ] = cycler(color=color_sequence)
    plt.rcParams[ "image.cmap"                  ] = "gnuplot2"
    plt.rcParams.update(overrides)



#################################
# ----- NEXT-100 Geometry ----- #
#################################


N100_rad = 983.3 / 2  # Radius [mm]
N100_hei = 1187       # Height [mm]
EL_gap = 9.7          # Electroluminescent gap [mm]

def plot_circle(RAD, LINESTYLE='-', col='black', label=None):
    """
    Create a circle.

    Parameters:
        rad (float): Radius of the circle.
        col (str): Color of the circle's edge. Default is 'black'.
        label (str): Optional label for the circle (e.g., for legends).

    Returns:
        matplotlib.patches.Circle: The circle object added to the axis.
    """
    # Create the circle
    circ = plt.Circle((0, 0), RAD, color=col, fill=False, ls=LINESTYLE, label=label)

    return circ

def selection_volume(z, dz, r, dr):
    """
    Calculate the selection volume of the NEXT-100 detector and its associated uncertainty.

    Parameters:
        z (float): Height of the cylinder (in mm).
        dz (float): Uncertainty in the height (in mm).
        r (float): Radius of the cylinder (in mm).
        dr (float): Uncertainty in the radius (in mm).

    Returns:
        tuple:
            - float: The calculated selection volume (in mm^3).
            - float: The propagated uncertainty in the volume (in mm^3).
    """
    # Calculate the volume of the cylinder
    volume_CV = np.pi * r**2 * z

    # Partial derivatives for uncertainty propagation
    dV_dz = np.pi * r**2
    dV_dr = 2 * np.pi * r * z

    # Propagation of uncertainty
    volume_err = np.sqrt((dV_dz * dz)**2 + (dV_dr * dr)**2)

    return volume_CV, volume_err


###############################
# ----- Personal Colors ----- #
###############################


hist_colors = ['black', 'crimson', 'darkorange', 'deepskyblue', 'green', 'navy', 'magenta', 'olive', 'mediumpurple', 'red', 'grey']

colors = [
    #(0.0, 0.0, 0.3),  # darkblue
    (0.3, 0.3, 1.0),  # lightblue
    (0.0, 0.0, 1.0),  # blue
    #(0.3, 0.3, 1.0),  # lightblue
    (0.0, 1.0, 0.0),  # green
    (1.0, 1.0, 0.0),  # yellow
    #(1.0, 0.3, 0.3),  # lightred
    (1.0, 0.0, 0.0),  # red
    (0.5, 0.0, 0.0),  # darkred
]

# Custom HSV Colormap
custom_hsv = LinearSegmentedColormap.from_list("custom_hsv", colors)

def plot_colormap(cmap, title="Colormap", figsize=(8, 2)):
    """
    Plot a given colormap as a gradient.

    Parameters:
        cmap (Colormap): Colormap to display.
        title (str): Title of the plot.
        figsize (tuple): Size of the figure.
    """
    gradient = np.linspace(0, 1, 256).reshape(1, -1)  # 1D gradient
    plt.figure(figsize=figsize)
    plt.imshow(gradient, aspect="auto", cmap=cmap)
    plt.gca().set_axis_off()
    plt.title(title, fontsize=14)
    plt.show()


###################################################
# ----- P l o t t i n g   F u n c t i o n s ----- #
####################################################


def mapping(x, y, wei=None, xy_bins=50, pos=False, norm=False):
    """
    Generate a 2D histogram map with optional normalization and position map.

    Parameters:
        x (array-like): x-coordinates of data points.
        y (array-like): y-coordinates of data points.
        wei (array-like, optional): Weights for the histogram. Default is None.
        xy_bins (int): Number of bins for both axes. Default is 50.
        pos (bool): If True, return only the position map (counts per bin). Default is False.
        norm (bool): If True, normalize maps by the center bin value. Default is False.

    Returns:
        tuple:
            - np.ndarray: Weighted map (normalized or unnormalized) or position map.
            - np.ndarray: Edges of x bins.
            - np.ndarray: Edges of y bins.

    Raises:
        ValueError: If normalization is requested and the center bin value is zero.
    """
    # Define bin edges for x and y axes
    x_bins = np.linspace(-600, 600, xy_bins)
    y_bins = np.linspace(-600, 600, xy_bins)
  
    # Compute position map (counts per bin)
    position_map, x_edges, y_edges = np.histogram2d(x, y, bins=[x_bins, y_bins])

    # Compute weighted map (sum of weights per bin)
    mapeo, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=wei)
    
    # Normalize weighted map by position map where counts are non-zero
    mapeo = np.divide(mapeo, position_map, out=np.zeros_like(mapeo), where=position_map != 0)
    
    if norm:
        # Compute bin centers for x and y axes
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

        # Find indices of the bin closest to the origin (0, 0)
        center_x_index = np.abs(x_centers).argmin()
        center_y_index = np.abs(y_centers).argmin()

        # Normalize maps by their center bin values
        pos_center_value   = position_map[center_x_index, center_y_index]
        mapeo_center_value = mapeo[center_x_index, center_y_index]

        if pos_center_value == 0 or mapeo_center_value == 0:
            raise ValueError("Normalization failed: Center bin value is zero.")

        position_map /= pos_center_value
        mapeo /= mapeo_center_value
       
    if pos:
        # Return only the position map if requested
        return position_map, x_edges, y_edges
    
    # Return the weighted map by default
    return mapeo, x_edges, y_edges

def hist_2D(x, y, x_bins=50, y_bins=50, wei=None):
    """
    Create a 2D histogram map for the given x and y data.

    Parameters:
        x (array-like): Data for the x-axis.
        y (array-like): Data for the y-axis.
        x_bins (int): Number of bins along the x-axis. Default is 50.
        y_bins (int): Number of bins along the y-axis. Default is 50.
        wei (array-like, optional): Weights for the histogram. Default is None.

    Returns:
        tuple: 
            - np.ndarray: 2D histogram map.
            - np.ndarray: Edges of the x bins.
            - np.ndarray: Edges of the y bins.
    """
    # Define bin edges for x and y axes
    X_bins = np.linspace(x.min(), x.max(), x_bins)
    Y_bins = np.linspace(y.min(), y.max(), y_bins)
    
    # Compute the 2D histogram with optional weights
    XY_map, x_edges, y_edges = np.histogram2d(x, y, bins=[X_bins, Y_bins], weights=wei)
    
    return XY_map, x_edges, y_edges


##########################
# ----- Visualizer ----- #
##########################


def event_display(
                        data: pd.DataFrame,
                        variable='E_corr',
                        event_column='event',
                        event=None
                    ):
    """
    Display event data with hit distributions in XY and YZ planes.

    Parameters:
        data (pd.DataFrame): Input data containing event information.
        variable (str): Column name for the variable to color the scatter plot. Default is 'E_corr'.
        event_column (str): Column name for the event IDs. Default is 'event'.
        event (int, optional): Specific event ID to plot. If None, a random event is chosen. Default is None.
    """
    # Check if the event column exists in the DataFrame
    if event_column not in data.columns:
        raise ValueError(f"No column named '{event_column}' found in the DataFrame.")

    # Get unique event IDs for the slider/dropdown
    event_ids = sorted(data[event_column].unique())

    # Define the plotting function that will be called by the widget
    def plot_event(evt_to_plot):

        # Select data for the chosen event
        event_data = data[data[event_column] == evt_to_plot]

        # Determine the energy column name
        if variable in data.columns:
            q = event_data[variable]
        else:
            raise ValueError(f"No {variable} variable found in the DataFrame.")

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))      # 1 row, 2 columns for subplots

        # --- XY Plot (First Subplot) ---
        scatter_xy = axs[0].scatter(event_data['X'], event_data['Y'], c=q, s=10, cmap='hsv', ec='none')
        axs[0].set_title(f'Back View')
        axs[0].set_xlabel('X [mm]')
        axs[0].set_ylabel('Y [mm]')
        axs[0].grid(True)
        axs[0].add_patch(plot_circle(N100_rad, col='black', label='NEXT-100 Radius'))
        axs[0].set_facecolor('whitesmoke')
        axs[0].axis('equal') # Ensure aspect ratio is equal for a proper spatial view

        # --- YZ Plot (Second Subplot) ---
        scatter_yz = axs[1].scatter(event_data['Z'], event_data['Y'],  c=q, s=10, cmap='hsv', ec='none')
        axs[1].set_title(f'Side View')
        axs[1].set_xlabel('Z [mm]')
        axs[1].set_ylabel('Y [mm]')
        # Add rectangle representing the NEXT-100 detector dimensions
        rect = plt.Rectangle((0, -N100_rad), N100_hei, 2*N100_rad,
                             edgecolor='black', facecolor='none', linestyle='-', label='NEXT-100 Volume')
        axs[1].add_patch(rect)
        axs[1].grid(True)
        axs[1].axis('equal')
        axs[1].set_facecolor('whitesmoke')

        # Y limits
        axs[0].set_ylim(-600, 600)
        
        plt.suptitle(f"Hit Distributions for Event: {evt_to_plot}", fontsize=15)
        plt.tight_layout()
        plt.show()

    if event is not None:
        plot_event(event)       # If a specific event is provided, plot it directly
    else:
        plot_event(np.random.choice(event_ids))