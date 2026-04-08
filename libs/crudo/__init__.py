"""
Crudo: the central library for NEXT-100 analysis.

This package unifies analysis tools into a single entry point,
organising functionalities into the following sub-modules.

Available modules:
- data_management: tools for loading, processing, and managing data and simulation files.
- energy_functions: functions for energy-related analysis.
- fit_functions: a custom core of statistics and fitting tools.
- plotting_tools: set of tools for visualising data and results.
- topology_functions: functions for analysing topological features of events.
- utilities: miscellaneous helper functions.
"""

from . import data_management    as dm
from . import energy_functions   as ef
from . import fit_functions      as ff
from . import plotting_tools     as pt
from . import topology_functions as tf
from . import utilities          as ut

# print("Crudo package loaded successfully.\nAvailable sub-modules: data_management (dm), energy_functions (ef), fit_functions (ff), plotting_tools (pt), topology_functions (tf), utilities (ut).")