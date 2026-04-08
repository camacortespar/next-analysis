import sys
sys.path.append('/lhome/ific/c/ccortesp/Analysis')


import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import Callable, List, Optional, Tuple, Union

# =============================================================================
# ----- HELPER FUNCTIONS -----
# =============================================================================

def split_isolated_clusters_3D(distance: List[float], nhit: int) -> Callable:
    '''
    Tags hits into isolated (or non) using a 3D anisotropic algorithm.
    Does not re-weight any variable.
    '''
    def split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Edge case: If the group is already tiny, all hits are considered isolated
        if len(df) <= nhit: return df.iloc[:0], df
        # Normalise distances
        xyz = df[['X', 'Y', 'Z']].values / distance
        dist = np.sqrt(3)
        # Use NearestNeighbors to find neighbors within the specified radius
        try:
            nbrs = NearestNeighbors(radius=dist, algorithm='ball_tree').fit(xyz)
            neighbors = nbrs.radius_neighbors(xyz, return_distance=False)
            mask_non_iso = np.array([len(neigh) > nhit for neigh in neighbors])
            return df[mask_non_iso], df[~mask_non_iso]      # (non-isolated, isolated)
        except Exception as e:
            print(f"Error in NearestNeighbors: {e}")
            return df.iloc[:0], df.iloc[:0]
    return split


# =============================================================================
# ----- PROCESSING FUNCTIONS -----
# =============================================================================





def process_alphas(df_alpha: pd.DataFrame, q_threshold) -> pd.DataFrame:
    """
    Processes alpha hits: applies a simple charge cut.
    """
    if df_alpha.empty:
        return pd.DataFrame(columns=list(df_alpha.columns) + ['E_hit_pe'])

    final_alpha_df = df_alpha[df_alpha['Q'] >= q_threshold].copy()
    # To ensure concatenation works, add the 'E_final_pe' column
    final_alpha_df['E_hit_pe'] = final_alpha_df['E_corr']
    return final_alpha_df