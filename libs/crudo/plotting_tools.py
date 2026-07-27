from   cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from   matplotlib.patches import Circle, Rectangle
from   matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import numpy as np
import pandas as pd
from   PIL import Image

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
    plt.rcParams[ "figure.figsize"              ] = 7, 5    # Change 10, 8 for papers
    plt.rcParams[ "font.size"                   ] = 25
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

# N100 means drift region where triggers are generated, not the full detector dimensions. 
N100_rad = 983.3 / 2  # Radius [mm]
N100_hei = 1187       # Height [mm]
Buffer_hei = 241      # Buffer gap [mm], same rad as drift region
# Electroluminescent region dimensions
EL_rad = 1100 / 2      # In [mm]
EL_hei = 9.7          # iN [mm]

def plot_circle(rad, linestyle='-', color='black', label=None):
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
    circ = plt.Circle((0, 0), rad, color=color, fill=False, ls=linestyle, lw=1.0, label=label)

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

comp_colors = ['navy', 'crimson']

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
custom_hsv = mcolors.LinearSegmentedColormap.from_list("custom_hsv", colors)

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
                    data         : pd.DataFrame
                  , variable     : str = 'E_corr'
                  , event_column : str = 'event'
                  , event        : int = None
                 ):
    """
    Display event data with hit distributions in XY and YZ planes.

    Parameters:
        data (pd.DataFrame)   : Input data containing event information.
        variable (str)        : Column name for the variable to color the scatter plot. Default is 'E_corr'.
        event_column (str)    : Column name for the event IDs. Default is 'event'.
        event (int, optional) : Specific event ID to plot. If None, a random event is chosen. Default is None.
    """
    if event_column not in data.columns:
        raise ValueError(f"  No column named '{event_column}' found in the DataFrame.")

    event_ids = sorted(data[event_column].unique())

    def plot_event(evt_to_plot):

        # Select data for the chosen event
        event_data = data[data[event_column] == evt_to_plot]

        if variable not in event_data.columns:
            raise ValueError(f"  No '{variable}' variable found in the DataFrame.")

        # Group total energy for coloring
        df_grouped_xy = event_data.groupby(['X', 'Y'], as_index=False)[variable].sum()
        df_grouped_zy = event_data.groupby(['Z', 'Y'], as_index=False)[variable].sum()

        fig, axs = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

        # --- XY Plot (First Subplot) ---
        scatter_xy = axs[0].scatter(df_grouped_xy['X'], df_grouped_xy['Y'], c=df_grouped_xy[variable], cmap='jet', s=15, ec='none')
        axs[0].set_title(f'Back View')
        axs[0].set_xlabel('X [mm]')
        axs[0].set_ylabel('Y [mm]')
        axs[0].set_xlim(-N100_rad*1.25, N100_rad*1.25)
        axs[0].set_ylim(-N100_rad*1.25, N100_rad*1.25)        
        axs[0].set_aspect('equal')
        axs[0].add_patch(Circle((0, 0), N100_rad, color='black', fill=False, lw=1.5, label='NEXT-100 Radius'))
        axs[0].set_facecolor('whitesmoke')
        axs[0].grid(True)

        # --- YZ Plot (Second Subplot) ---
        scatter_yz = axs[1].scatter(df_grouped_zy['Z'], df_grouped_zy['Y'], c=df_grouped_zy[variable], cmap='jet', s=15, ec='none')
        axs[1].set_title(f'Side View')
        axs[1].set_xlabel('Z [mm]')
        # Add rectangle representing the NEXT-100 detector dimensions
        rect = plt.Rectangle((0, -N100_rad), N100_hei, 2*N100_rad,
                             ec='black', fc='none', ls='-', lw=1.0, label='NEXT-100 Volume')
        axs[1].add_patch(rect)
        axs[1].set_facecolor('whitesmoke')
        axs[1].grid(True)

        # Global adjustments
        plt.suptitle(f"Hit Distributions for Event: {evt_to_plot}", y=0.92, fontsize=20)
        plt.tight_layout(rect=[0, 0, 0.9, 1.0])
        # cbar_ax = fig.add_axes([0.9, 0.2, 0.01, 0.55])
        # cbar = fig.colorbar(scatter_xy, cax=cbar_ax, label=f"Total {variable}")
        fig.colorbar(scatter_xy, ax=axs[1], label=f"Total {variable}")
        plt.show()

    if event is not None:
        plot_event(event)   # If a specific event is provided, plot it directly
    else:
        plot_event(np.random.choice(event_ids))

# Event display for clustered hits
# Source:  https://github.com/SamueleTorelli/ASpirit/blob/main/src/HE_plot_functions.py
def display_event_cluster(
                            data         : pd.DataFrame
                          , variable     : str = 'E_corr_pe'
                          , event_column : str = 'event'
                          , event        : int = None
                         ):
    if event_column not in data.columns:
        raise ValueError(f"No column named '{event_column}' found in the DataFrame.")

    if event is not None:
        df_reco_event = data[data[event_column] == event]
    else:
        event_ids = data[event_column].unique()
        event = np.random.choice(event_ids)
        df_reco_event = data[data[event_column] == event]

    # Group total energy for coloring
    df_grouped_xy = df_reco_event.groupby(['X', 'Y'], as_index=False)[variable].sum()
    df_grouped_zy = df_reco_event.groupby(['Z', 'Y'], as_index=False)[variable].sum()
    df_grouped_xz = df_reco_event.groupby(['X', 'Z'], as_index=False)[variable].sum()
    
    # Group by cluster (includes scattered)
    df_clustered_xy = df_reco_event.groupby(['X', 'Y', 'cluster'], as_index=False)[variable].sum()
    df_clustered_zy = df_reco_event.groupby(['Z', 'Y', 'cluster'], as_index=False)[variable].sum()
    df_clustered_xz = df_reco_event.groupby(['X', 'Z', 'cluster'], as_index=False)[variable].sum()
    
    color_sequence = ("k", "m", "g", "b", "r",
                      "gray", "aqua", "gold", "lime", "purple",
                      "brown", "lawngreen", "tomato", "lightgray", "lightpink")

    fig, axes = plt.subplots(3, 2, figsize=(16, 21), sharex='row', sharey='row')
    
    # --- TOP LEFT: All hits X vs Y ---
    sc0 = axes[0, 0].scatter(df_grouped_xy['X'], df_grouped_xy['Y'], c=df_grouped_xy[variable], cmap='jet', s=15, ec='none')
    axes[0, 0].set_title("Back View")
    axes[0, 0].set_xlabel("X [mm]")
    axes[0, 0].set_ylabel("Y [mm]")
    axes[0, 0].set_xlim(-N100_rad*1.25, N100_rad*1.25)
    axes[0, 0].set_ylim(-N100_rad*1.25, N100_rad*1.25)
    axes[0, 0].set_aspect('equal', adjustable='box')
    axes[0, 0].add_patch(Circle((0, 0), N100_rad, color='black', fill=False, lw=1.5, label='NEXT-100 Radius'))
    axes[0, 0].set_facecolor("whitesmoke")
    axes[0, 0].grid(True)
    fig.colorbar(sc0, ax=axes[0, 0], label=f"Total {variable}")
    
    # --- TOP RIGHT: Clustered hits X vs Y ---
    for cl in sorted(df_clustered_xy['cluster'].unique()):
        cluster_df = df_clustered_xy[df_clustered_xy['cluster'] == cl]
        color = color_sequence[-3] if cl == -1 else color_sequence[cl]
        label = 'Scattered' if cl == -1 else f'Cluster {cl}'
        axes[0, 1].scatter(cluster_df['X'], cluster_df['Y'], s=15, ec='none', label=label, c=color)
    axes[0, 1].set_title("Clustered Hits")
    axes[0, 1].set_xlabel("X [mm]")
    axes[0, 1].set_ylabel("Y [mm]")
    axes[0, 1].set_xlim(-N100_rad*1.25, N100_rad*1.25)
    axes[0, 1].set_ylim(-N100_rad*1.25, N100_rad*1.25)
    axes[0, 1].set_aspect('equal', adjustable='box')
    axes[0, 1].add_patch(Circle((0, 0), N100_rad, color='red', fill=False, lw=1.5, label='NEXT-100 Radius'))
    axes[0, 1].set_facecolor("whitesmoke")
    axes[0, 1].grid(True)
    axes[0, 1].legend(loc='upper right', markerscale=1.5, fontsize=8, facecolor='none', edgecolor='none')
    
    # --- MIDDLE LEFT: All hits Z vs Y ---
    sc2 = axes[1, 0].scatter(df_grouped_zy['Z'], df_grouped_zy['Y'], c=df_grouped_zy[variable], cmap='jet', s=15, ec='none')
    axes[1, 0].set_title("Side View")
    axes[1, 0].set_xlabel("Z [mm]")
    axes[1, 0].set_ylabel("Y [mm]")
    axes[1, 0].set_ylim(-N100_rad*1.25, N100_rad*1.25)
    axes[1, 0].set_facecolor("whitesmoke")
    axes[1, 0].grid(True)
    fig.colorbar(sc2, ax=axes[1, 0], label=f"Total {variable}")
    
    # --- MIDDLE RIGHT: Clustered hits Z vs Y ---
    for cl in sorted(df_clustered_zy['cluster'].unique()):
        cluster_df = df_clustered_zy[df_clustered_zy['cluster'] == cl]
        color = color_sequence[-3] if cl == -1 else color_sequence[cl]
        label = 'Scattered' if cl == -1 else f'Cluster {cl}'
        axes[1, 1].scatter(cluster_df['Z'], cluster_df['Y'], s=15, ec='none', label=label, c=color)
    axes[1, 1].set_xlabel("Z [mm]")
    axes[1, 1].set_ylabel("Y [mm]")
    axes[1, 1].set_facecolor("whitesmoke")
    axes[1, 1].grid(True)
    axes[1, 1].legend(loc='best', markerscale=1.5, fontsize=12, facecolor='none', edgecolor='none')
    
    # --- BOTTOM LEFT: All hits X vs Z ---
    sc4 = axes[2, 0].scatter(df_grouped_xz['Z'], df_grouped_xz['X'], c=df_grouped_xz[variable],  cmap='jet', s=15, ec='none')
    axes[2, 0].set_title("Top View")
    axes[2, 0].set_xlabel("Z [mm]")
    axes[2, 0].set_ylabel("X [mm]")
    axes[2, 0].set_ylim(-N100_rad*1.25, N100_rad*1.25)
    axes[2, 0].set_facecolor("whitesmoke")
    axes[2, 0].grid(True)
    fig.colorbar(sc4, ax=axes[2, 0], label=f"Total {variable}")
    
    # --- BOTTOM RIGHT: Clustered hits X vs Z ---
    for cl in sorted(df_clustered_xz['cluster'].unique()):
        cluster_df = df_clustered_xz[df_clustered_xz['cluster'] == cl]
        color = color_sequence[-3] if cl == -1 else color_sequence[cl]
        label = 'Scattered' if cl == -1 else f'Cluster {cl}'
        axes[2, 1].scatter(cluster_df['Z'],cluster_df['X'], s=15, ec='none', label=label, c=color)
    axes[2, 1].set_xlabel("Z [mm]")
    axes[2, 1].set_ylabel("X [mm]")
    axes[2, 1].set_facecolor("whitesmoke")
    axes[2, 1].grid(True)
    axes[2, 1].legend(loc='best', markerscale=1.5, fontsize=12, facecolor='none', edgecolor='none')
    
    # Global adjustments
    plt.suptitle(f"Hit Distributions for Event: {event}", y=0.92, fontsize=20)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.show()