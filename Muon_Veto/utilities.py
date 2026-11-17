from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

# ---

try:
    import hdf5plugin
    import h5py
    READER = "h5py"
except ImportError:
    try:
        import tables
        READER = "tables"
    except ImportError:
        print("[-] Please run: pip install hdf5plugin")
        sys.exit(1)

# ---8

# --- Module Functions --- #
def load_veto_data(file_path: Path) -> Tuple[np.ndarray, int, float, float]:
    """Loads HDF5 charge and event datasets, returning data and exact duration."""
    print(f"Loading data from: {file_path}")
    if READER == "h5py":
        with h5py.File(file_path, "r") as f:
            charges = np.array(f["charges"], dtype=np.float32)
            timestamps = np.array(f["events"]["timestamp"])
    else:
        with tables.open_file(file_path, "r") as f:
            charges = f.root.charges.read().astype(np.float32)
            timestamps = f.root.events.read()["timestamp"]

    n_events, _ = charges.shape
    time_diff = timestamps[-1] - timestamps[0]
    elapsed_time_s = (time_diff / 1e9) if time_diff > 1e11 else (time_diff / 1000.0 if time_diff > 0 else 1.0)
    total_daq_rate = n_events / elapsed_time_s

    print(f"Loaded {n_events:,} events over {elapsed_time_s:.2f} seconds.")
    print(f"Total DAQ Trigger Rate: {total_daq_rate:.1f} Hz\n")
    return charges, n_events, elapsed_time_s, total_daq_rate

def characterize_electronic_noise( 
                                   charges        : np.ndarray,
                                   free_channels  : List[int],
                                   channel_config : Dict[int, Dict[str, int]]
                                 ) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float], float, float, float, float]:
    """Measures pure electronic noise spread (MAD) across unconnected channels."""
    print("=" * 80)
    print("ELECTRONIC NOISE FROM UNCONNECTED CHANNELS")
    print("=" * 80)
    print(f"{'Channel':<10} | {'Gain':<6} | {'Pedestal (ADC)':<16} | {'Noise RMS (ADC)':<16} | {'Threshold (ADC)':<20}")
    print("-" * 80)

    free_pedestals, free_sigmas, ch_threshold, norm_sigmas = {}, {}, {}, []

    for ch in free_channels:
        ch_data = charges[:, ch]
        med = float(np.median(ch_data))
        mad = float(np.median(np.abs(ch_data - med)))
        sigma = 1.4826 * mad if mad > 0 else float(np.std(ch_data))
        if sigma < 1.0:
            sigma = float(np.std(ch_data))

        g = channel_config[ch]["gain"]
        norm_sigmas.append(sigma / g)
        free_pedestals[ch] = med
        free_sigmas[ch] = sigma
        thresh = med + 5.0 * sigma
        ch_threshold[ch] = thresh

        print(f"Ch {ch:<7} | {g:<6} | {med:<16.2f} | {sigma:<16.2f} | {thresh:<16.2f}")

    ref_ped = float(np.median(list(free_pedestals.values())))
    ref_sigma = float(np.median(list(free_sigmas.values())))
    ref_thresh = float(np.median(list(ch_threshold.values())))
    sigma_per_gain = float(np.median(norm_sigmas))

    print("-" * 80)
    print(f"--> Median Pedestal = {ref_ped:.2f} ADC")
    print(f"--> Median Sigma = ±{ref_sigma:.2f} ADC")
    print(f"--> Median Threshold = {ref_thresh:.2f} ADC")
    print(f"--> Normalized Electronic Sigma = {sigma_per_gain:.2f} ADC / Gain")
    print("=" * 80 + "\n")

    return free_pedestals, free_sigmas, ch_threshold, ref_ped, ref_sigma, ref_thresh, sigma_per_gain

def compute_channel_thresholds(
    charges: np.ndarray,
    channel_config: Dict[int, Dict[str, int]],
    ref_ped: float,
    ref_sigma: float,
    ref_thresh: float,
    sigma_per_gain: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates channel baseline modes and gain-scaled 5-sigma hit thresholds."""
    n_events, n_channels = charges.shape
    pedestals = np.zeros(n_channels)
    ped_rms = np.zeros(n_channels)
    thresholds = np.zeros(n_channels)

    lim_d = ref_ped - 5.0 * ref_sigma
    lim_u = ref_thresh

    for ch in range(n_channels):
        ch_data = charges[:, ch]
        counts, bin_edges = np.histogram(ch_data, bins=100, range=(lim_d, lim_u))
        mode_ped = bin_edges[np.argmax(counts)]
        pedestals[ch] = mode_ped

        ch_gain = channel_config[ch]["gain"]
        ch_expected_elec_sigma = sigma_per_gain * ch_gain

        mad = np.median(np.abs(ch_data - mode_ped))
        ch_meas_sigma = 1.4826 * mad

        eff_sigma = min(max(ch_meas_sigma, ch_expected_elec_sigma), 25.0)
        ped_rms[ch] = eff_sigma

        min_cut = 35.0 * (ch_gain / 10.0)
        thresholds[ch] = pedestals[ch] + max(5.0 * eff_sigma, min_cut)

    return pedestals, ped_rms, thresholds

def evaluate_bar_coincidences(
    charges: np.ndarray,
    thresholds: np.ndarray,
    pedestals: np.ndarray,
    ped_rms: np.ndarray,
    bar_map: Dict[int, Tuple[int, int]],
    elapsed_time_s: float,
    saturation_adc: float = 3800.0
) -> Tuple[Dict[int, Dict[str, Any]], List[np.ndarray]]:
    """Evaluates software coincidences, saturation percentages, and rates per bar."""
    bar_stats = {}
    bar_masks = []

    print("=" * 150)
    print(f"{'Bar':<5} | {'SiPM A,B':<8} | {'Ped A (RMS)':<16} | {'Ped B (RMS)':<16} | {'Threshold A':<9} | {'Threshold B':<9} | {'Sat A %':<9} | {'Sat B %':<9} | {'Coinc Hits':<11} | {'Rate (Hz)':<10} | {'Status':<12}")
    print("=" * 150)

    for bar_id, (ch_a, ch_b) in bar_map.items():
        q_a, q_b = charges[:, ch_a], charges[:, ch_b]
        coinc_mask = (q_a > thresholds[ch_a]) & (q_b > thresholds[ch_b])
        bar_masks.append(coinc_mask)
        n_hits = int(np.sum(coinc_mask))
        rate_hz = n_hits / elapsed_time_s

        if n_hits > 0:
            sat_a_pct = (np.sum(q_a[coinc_mask] >= saturation_adc) / n_hits) * 100.0
            sat_b_pct = (np.sum(q_b[coinc_mask] >= saturation_adc) / n_hits) * 100.0
        else:
            sat_a_pct, sat_b_pct = 0.0, 0.0

        if rate_hz > 50.0:
            status = "SINGING! ⚠️"
        elif rate_hz < 1.0:
            status = "DEAF / OFF"
        else:
            status = "OK (Cosmic)"

        bar_stats[bar_id] = {
            "ch_a": ch_a, "ch_b": ch_b,
            "hits": n_hits, "rate": rate_hz,
            "sat_a_pct": sat_a_pct, "sat_b_pct": sat_b_pct,
            "mask": coinc_mask,
            "q_a": q_a, "q_b": q_b,
            "thres_a": thresholds[ch_a], "thres_b": thresholds[ch_b],
            "status": status,
        }

        ped_a_str = f"{pedestals[ch_a]:.1f} (±{ped_rms[ch_a]:.1f})"
        ped_b_str = f"{pedestals[ch_b]:.1f} (±{ped_rms[ch_b]:.1f})"
        print(f"{bar_id:<5} | {ch_a:>2}, {ch_b:<2}   | {ped_a_str:<16} | {ped_b_str:<16} | {thresholds[ch_a]:<11.2f} | {thresholds[ch_b]:<11.2f} | {sat_a_pct:<8.2f}% | {sat_b_pct:<8.2f}% | {n_hits:<11,d} | {rate_hz:<10.1f} | {status:<12}")

    print("=" * 150 + "\n")
    return bar_stats, bar_masks

def reconcile_triggers(
    bar_masks: List[np.ndarray],
    bar_stats: Dict[int, Dict[str, Any]],
    n_events: int,
    elapsed_time_s: float,
    total_daq_rate: float
) -> None:
    """Reconciles physical single-bar, tracking, and ghost/empty triggers."""
    coinc_matrix = np.column_stack(bar_masks)
    bars_fired = np.sum(coinc_matrix, axis=1)

    n_zero = int(np.sum(bars_fired == 0))
    n_single = int(np.sum(bars_fired == 1))
    n_multi = int(np.sum(bars_fired >= 2))
    sum_bar_hits = int(np.sum([b["hits"] for b in bar_stats.values()]))

    print("=" * 65)
    print(" DAQ TRIGGER INFORMATION")
    print("=" * 65)
    print(f"Total DAQ Triggers Recorded:      {n_events:>10,d}  ({total_daq_rate:.1f} Hz)")
    print(f"Sum of Hits across all Bars:      {sum_bar_hits:>10,d}  ({sum_bar_hits / elapsed_time_s:.1f} Hz)")
    print("-" * 65)
    print(f"  * Events with EXACTLY 1 Bar:    {n_single:>10,d}  ({n_single / n_events * 100:.1f}%)")
    print(f"  * Events with >= 2 Bars:        {n_multi:>10,d}  ({n_multi / n_events * 100:.1f}%)")
    print(f"  * Events with 0 Bars:           {n_zero:>10,d}  ({n_zero / n_events * 100:.1f}%)")
    print("=" * 65)
    if n_zero > 0:
        print(f"[!] Notice: {n_zero:,} triggers ({n_zero / elapsed_time_s:.1f} Hz) were DAQ noise / ghost triggers")
        print("    where none of the active bars satisfied the coincidence cut.")
    print("\n")

def plot_unconnected_noise( 
                            charges        : np.ndarray,
                            free_channels  : List[int],
                            free_pedestals : Dict[int, float],
                            free_sigmas    : Dict[int, float],
                            ch_threshold   : Dict[int, float],
                            channel_config : Dict[int, Dict[str, int]],
                            dac_threshold  : int | float 
                          ) -> None:
    """Noise profiles of unconnected channels."""
    free_rows = (len(free_channels) + 3) // 4
    fig_f, axes_f = plt.subplots(free_rows, 4, figsize=(18, 4.5 * free_rows), sharey=True)
    axes_f = axes_f.flatten()
    max_x  = np.max(list(ch_threshold.values()))

    for idx, ch in enumerate(free_channels):
        ax = axes_f[idx]
        ch_data = charges[:, ch]
        med = free_pedestals[ch]
        sig = free_sigmas[ch]
        thresh = ch_threshold[ch]
        g = channel_config[ch]["gain"]

        ax.hist(ch_data, bins=100, range=(0, 2.0*max_x), color="mediumseagreen", edgecolor="none")
        ax.axvline(med, color="darkblue", ls="--", label=f"Pedestal ({med:.1f} ADC)")
        ax.axvline(thresh, color="crimson", ls=":", label=f"+5σ Threshold ({thresh:.1f} ADC)")
        ax.set_title(f"Ch {ch} (Gain {g})\nRMS σ = ±{sig:.2f} ADC", fontsize=10, loc="left")
        ax.set_xlabel("ADC Counts", fontsize=9)
        ax.set_xlim(0, 2.0*max_x)
        if idx % 4 == 0:
            ax.set_ylabel("Counts", fontsize=9)
        ax.legend(fontsize=7.5, loc="upper right", facecolor="none", framealpha=0)

    # Hide unused subplot boxes
    for ax in axes_f[len(free_channels):]:
        ax.axis("off")

    plt.suptitle(f"AND Config | DAC = {dac_threshold}", fontsize=20, y=1.0)
    plt.tight_layout()
    plt.show()

def plot_active_bars(
    bar_stats: Dict[int, Dict[str, Any]],
    bar_map: Dict[int, Tuple[int, int]],
    dac_threshold: int | float,
    saturation_adc: float = 3800.0
) -> None:
    """Diagnostic Plot 2: Active bars dual background & coincidence overlay."""
    n_bars = len(bar_map)
    n_cols = 4 if n_bars > 9 else 3
    n_rows = int(np.ceil(n_bars / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.6 * n_rows), sharex=True)
    axes = axes.flatten()

    for idx, bar_id in enumerate(bar_map.keys()):
        ax = axes[idx]
        ch_a = bar_stats[bar_id]["ch_a"]
        ch_b = bar_stats[bar_id]["ch_b"]
        q_a = bar_stats[bar_id]["q_a"]
        q_b = bar_stats[bar_id]["q_b"]
        coinc_mask = bar_stats[bar_id]["mask"]

        if np.sum(coinc_mask) > 10:
            max_seen = max(np.percentile(q_a[coinc_mask], 99.5), np.percentile(q_b[coinc_mask], 99.5))
            x_limit = min(max(max_seen * 1.3, 1200), 4150)
        else:
            max_raw = max(np.percentile(q_a, 99.5), np.percentile(q_b, 99.5))
            x_limit = min(max(max_raw * 1.3, 1000), 4150)

        bins = np.linspace(0, x_limit, 100)

        # Subsampled raw backgrounds with scale-matching weights
        sub_size = min(len(q_a), 25000)
        sub_idx = np.random.choice(len(q_a), sub_size, replace=False)
        weights = np.full(sub_size, len(q_a) / sub_size)

        ax.hist(q_a[sub_idx], bins=bins, weights=weights, color="lightskyblue", alpha=0.35, label=f"Raw A (Ch {ch_a})")
        ax.hist(q_b[sub_idx], bins=bins, weights=weights, color="lightcoral", alpha=0.35, label=f"Raw B (Ch {ch_b})")

        if np.sum(coinc_mask) > 0:
            ax.hist(q_a[coinc_mask], bins=bins, color="royalblue", histtype="step", linewidth=1.6, label=f"Coinc A (Ch {ch_a})")
            ax.hist(q_b[coinc_mask], bins=bins, color="crimson", histtype="step", linewidth=1.6, label=f"Coinc B (Ch {ch_b})")

        # if x_limit >= saturation_adc:
        ax.axvline(saturation_adc, color="darkorange", linestyle="--", alpha=0.7)

        ax.set_title(f"Bar {bar_id} ({bar_stats[bar_id]['status']}) - Rate: {bar_stats[bar_id]['rate']:.1f} Hz", fontsize=10, fontweight="bold")
        ax.set_yscale("log")
        ax.set_xlim(0, 4150)
        ax.legend(fontsize=7, loc="upper right", framealpha=0)

        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Charge [ADC counts]", fontsize=9)
        if idx % n_cols == 0:
            ax.set_ylabel("Events (Log)", fontsize=9)

    for ax in axes[n_bars:]:
        ax.axis("off")

    plt.suptitle(f"Module | DAC = {dac_threshold}", fontsize=20, y=1.0)
    plt.tight_layout()
    plt.show()

def plot_correlations(
    bar_stats: Dict[int, Dict[str, Any]],
    bar_map: Dict[int, Tuple[int, int]],
    dac_threshold: int | float,
    saturation_adc: float = 3800.0,
    max_scatter_points: int = 3000
) -> None:
    """Diagnostic Plot 3: 2D coincidence correlation scatter per bar."""
    n_bars = len(bar_map)
    n_cols = 4 if n_bars > 9 else 3
    n_rows = int(np.ceil(n_bars / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.6 * n_rows))
    axes = axes.flatten()

    for idx, bar_id in enumerate(bar_map.keys()):
        ax = axes[idx]
        ch_a = bar_stats[bar_id]["ch_a"]
        ch_b = bar_stats[bar_id]["ch_b"]
        q_a = bar_stats[bar_id]["q_a"]
        q_b = bar_stats[bar_id]["q_b"]
        coinc_mask = bar_stats[bar_id]["mask"]

        if np.sum(coinc_mask) > 5:
            scatter_idx = np.where(coinc_mask)[0]
            if len(scatter_idx) > max_scatter_points:
                scatter_idx = np.random.choice(scatter_idx, max_scatter_points, replace=False)
            ax.scatter(q_a[scatter_idx], q_b[scatter_idx], color="purple", s=4, alpha=0.35)
        else:
            ax.text(0.5, 0.5, "No Coinc Hits", ha="center", va="center", transform=ax.transAxes, color="gray")

        ax.axhline(saturation_adc, color="darkorange", linestyle=":", alpha=0.5)
        ax.axvline(saturation_adc, color="darkorange", linestyle=":", alpha=0.5)
        ax.set_title(f"Bar {bar_id}: Ch {ch_a} vs Ch {ch_b}", fontsize=10, fontweight="bold")
        ax.set_xlim(0, 4150)
        ax.set_ylim(0, 4150)
        ax.set_xlabel(f"Ch {ch_a} [ADC]", fontsize=8.5)
        ax.set_ylabel(f"Ch {ch_b} [ADC]", fontsize=8.5)

    for ax in axes[n_bars:]:
        ax.axis("off")

    plt.suptitle(f"SiPM A vs SiPM B Correlation | DAC = {dac_threshold}", fontsize=20, y=1.0)
    plt.tight_layout()
    plt.show()

# ---
def run_veto_analysis(
    file_path: str | Path,
    bar_map: Dict[int, Tuple[int, int]],
    free_channels: List[int],
    channel_config: Dict[int, Dict[str, int]],
    dac_threshold: int | float,
    show_noise_plot: bool = True,
    show_spectra_plot: bool = True,
    show_correlation_plot: bool = True,
    saturation_adc: float = 3800.0
) -> Dict[str, Any]:
    """Master pipeline to process, reconcile, and plot cosmic muon veto runs."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Ingest
    charges, n_events, elapsed_time_s, total_daq_rate = load_veto_data(file_path)

    # 2. Electronic Noise (Free Channels)
    free_peds, free_sigs, ch_thresh, ref_ped, ref_sig, ref_thresh, sigma_per_gain = characterize_electronic_noise(
        charges, free_channels, channel_config
    )

    # 3. Baselines & Thresholds (All Channels)
    pedestals, ped_rms, thresholds = compute_channel_thresholds(
        charges, channel_config, ref_ped, ref_sig, ref_thresh, sigma_per_gain
    )

    # 4. Bar Coincidences
    bar_stats, bar_masks = evaluate_bar_coincidences(
        charges, thresholds, pedestals, ped_rms, bar_map, elapsed_time_s, saturation_adc
    )

    # 5. Attribution Reconciliation
    reconcile_triggers(bar_masks, bar_stats, n_events, elapsed_time_s, total_daq_rate)

    # 6. Diagnostic Plots
    if show_noise_plot:
        plot_unconnected_noise(charges, free_channels, free_peds, free_sigs, ch_thresh, channel_config, dac_threshold)

    if show_spectra_plot:
        plot_active_bars(bar_stats, bar_map, dac_threshold, saturation_adc)

    if show_correlation_plot:
        plot_correlations(bar_stats, bar_map, dac_threshold, saturation_adc)

    print("\nDone! All diagnostic plots and rate reconciliations generated.")
    return {
        "elapsed_time_s": elapsed_time_s,
        "total_daq_rate": total_daq_rate,
        "pedestals": pedestals,
        "ped_rms": ped_rms,
        "thresholds": thresholds,
        "bar_stats": bar_stats,
    }

def run_noise_analysis(
    file_path: str | Path,
    free_channels: List[int],
    channel_config: Dict[int, Dict[str, int]],
    dac_threshold: int | float,
    show_noise_plot: bool = True,
    saturation_adc: float = 3800.0
) -> Dict[str, Any]:
    """Master pipeline to process, reconcile, and plot cosmic muon veto runs."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Ingest
    charges, n_events, elapsed_time_s, total_daq_rate = load_veto_data(file_path)

    # 2. Electronic Noise (Free Channels)
    free_peds, free_sigs, ch_thresh, ref_ped, ref_sig, ref_thresh, sigma_per_gain = characterize_electronic_noise(
        charges, free_channels, channel_config
    )


    # 6. Diagnostic Plots
    if show_noise_plot:
        plot_unconnected_noise(charges, free_channels, free_peds, free_sigs, ch_thresh, channel_config, dac_threshold)

    print("\nDone! All diagnostic plots and rate reconciliations generated.")


# ---

# - - - TO CHECK - - - #

def calc_rate(directory):
    """
    Calculates measured and corrected rates for data runs in a specified directory.

    The corrected rate accounts for events lost due to buffer overflows.

    Args:
        directory (str): The full path to the directory containing the run data files.

    Returns:
        dict: A dictionary where keys are run numbers and values are dictionaries
              containing rates, efficiency, run type, and status.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory not found: {directory}")
        return {}

    file_prefix = 'muons_run_'
    file_suffix = '.h5'
    hdf5_key = 'data'
    timestamp_column = 'timestamp'
    lost_buffer_column = 'lostBuffer' # Columna con los eventos perdidos
    run_results = {}

    all_files = os.listdir(directory)
    
    run_files_info = []
    for filename in all_files:
        if filename.startswith(file_prefix) and filename.endswith(file_suffix):
            match = re.search(r'muons_run_(\d+)\.h5', filename)
            if match:
                run_number = int(match.group(1))
                run_type = 'No Source' if run_number % 2 != 0 else 'With Source'
                run_files_info.append({'filename': filename, 'run_number': run_number, 'type': run_type})

    run_files_info.sort(key=lambda x: x['run_number'])

    if not run_files_info:
        print(f"Warning: No valid run files found in {directory}.")
        return {}
        
    for file_info in run_files_info:
        run_num = file_info['run_number']
        file_name = file_info['filename']
        run_type = file_info['type']
        file_path = os.path.join(directory, file_name)

        try:
            data_df = pd.read_hdf(file_path, hdf5_key)

            if data_df.empty or timestamp_column not in data_df.columns:
                status_msg = 'Empty DataFrame or Timestamp Column Missing'
                run_results[run_num] = {'Run Type': run_type, 'Status': status_msg}
                continue

            # --- LÓGICA DE CORRECCIÓN AÑADIDA ---
            if lost_buffer_column not in data_df.columns:
                status_msg = f"Column '{lost_buffer_column}' Not Found"
                run_results[run_num] = {'Run Type': run_type, 'Status': status_msg}
                continue

            first_timestamp = data_df[timestamp_column].iloc[0]
            last_timestamp = data_df[timestamp_column].iloc[-1]
            time_difference = (last_timestamp - first_timestamp) * 0.001 # a segundos

            if time_difference > 0:
                events_registered = len(data_df)
                events_lost = data_df[lost_buffer_column].sum()
                total_events_estimated = events_registered + events_lost
                
                # Calcular tasas y eficiencia
                measured_rate = events_registered / time_difference
                corrected_rate = total_events_estimated / time_difference
                daq_efficiency = (events_registered / total_events_estimated) if total_events_estimated > 0 else 1.0

                run_results[run_num] = {
                    'Measured Rate (Hz)': f'{measured_rate:.6f}',
                    'Corrected Rate (Hz)': f'{corrected_rate:.6f}',
                    'DAQ Efficiency': f'{daq_efficiency:.4f}',
                    'Events Registered': events_registered,
                    'Events Lost': int(events_lost),
                    'Run Type': run_type, 
                    'Status': 'Success'
                }
            else:
                run_results[run_num] = {
                    'Run Type': run_type, 
                    'Status': 'Zero/Negative Time Diff'
                }

        except KeyError:
            run_results[run_num] = {'Run Type': run_type, 'Status': f"HDF5 Key '{hdf5_key}' Not Found"}
        except Exception as e:
            run_results[run_num] = {'Run Type': run_type, 'Status': f'Error: {e}'}

    return run_results

def plot_rates(data_df, skip_bars=None):
    """
    Creates a scatter plot of the corrected rates, grouped by run type.

    Args:
        data_df (pd.DataFrame): The DataFrame containing the consolidated rates.
        skip_bars (list, optional): List of bar labels to exclude, e.g. ["B2", "C5"].
    """
    if data_df.empty:
        print("Cannot plot: The input DataFrame is empty.")
        return

    # Usar la columna con la tasa corregida para el ploteo
    rate_column_to_plot = 'Corrected Rate (Hz)'

    # Filtrar solo filas exitosas y con un valor de tasa válido
    plot_data = data_df[data_df['Status'] == 'Success'].copy()
    plot_data.dropna(subset=[rate_column_to_plot], inplace=True) # Eliminar filas donde la tasa es N/A

    if plot_data.empty:
        print("No successful rates to plot.")
        return

    if skip_bars:
        plot_data = plot_data[~plot_data['Bar Label'].isin(skip_bars)]

    plot_data = plot_data.sort_values(by=['Bar Letter', 'Bar Number'])

    no_source_data = plot_data[plot_data['Run Type'] == 'No Source']
    with_source_data = plot_data[plot_data['Run Type'] == 'With Source']

    plt.style.use('ggplot')
    plt.figure(figsize=(14, 8))

    plt.scatter(
        no_source_data['Bar Label'],
        no_source_data[rate_column_to_plot],
        color='blue',
        marker='o',
        label='Without Source'
    )

    plt.scatter(
        with_source_data['Bar Label'],
        with_source_data[rate_column_to_plot],
        color='red',
        marker='^',
        label='With Source'
    )

    plt.title('Muon Veto Corrected Rates for All Bars')
    plt.xlabel('Bar')
    plt.ylabel('Corrected Rate (Hz)')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.show()

    base_directory = '/lhome/ific/c/ccortesp/Analysis/Muon_Veto/data/'
    final_rates_list = []
        
    for root, dirs, files in os.walk(base_directory):
        if os.path.basename(os.path.dirname(root)) in ['A', 'B', 'C', 'D', 'E', 'F']:
            bar_letter = os.path.basename(os.path.dirname(root))
            bar_number = os.path.basename(root)

            if bar_letter not in ['A', 'B', 'C', 'D', 'E', 'F'] or not bar_number.isdigit():
                continue
                
            rates_dict = calc_rate(root)
                
            # Desempaquetar el diccionario con los nuevos campos
            for run_num, data in rates_dict.items():
                run_entry = {
                    'Bar Letter': bar_letter,
                    'Bar Number': int(bar_number),
                    'Bar Label': f"{bar_letter}{bar_number}",
                    'Run Number': run_num,
                    'Run Type': data.get('Run Type'),
                    'Status': data.get('Status'),
                    # Usamos .get() para evitar errores si una corrida falló y no tiene estas claves
                    'Measured Rate (Hz)': data.get('Measured Rate (Hz)'),
                    'Corrected Rate (Hz)': data.get('Corrected Rate (Hz)'),
                    'DAQ Efficiency': data.get('DAQ Efficiency'),
                    'Events Registered': data.get('Events Registered'),
                    'Events Lost': data.get('Events Lost')
                }
                final_rates_list.append(run_entry)
                    
    # Crear el DataFrame final
    if final_rates_list:
        final_df = pd.DataFrame(final_rates_list)
        # Convertir columnas numéricas, usando 'coerce' para manejar valores N/A o faltantes
        numeric_cols = [
            'Run Number', 'Measured Rate (Hz)', 'Corrected Rate (Hz)', 
            'DAQ Efficiency', 'Events Registered', 'Events Lost'
        ]
        for col in numeric_cols:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
        
        # Imprime el nuevo DataFrame para verificar. Verás las nuevas columnas.
        # print(final_df.sort_values(by=['Bar Letter', 'Bar Number', 'Run Number']).reset_index(drop=True))
    else:
        print("No valid run data found in the specified directories.")
        final_df = pd.DataFrame() # Crear un DataFrame vacío para evitar errores posteriores

def COMPUTE_RATE_FILE(FILE_PATH):
    """
    Helper function to calculate the event rate from a single HDF5 file.

    This function reads an HDF5 file containing event data, computes the event rate 
    (events per second), and calculates the associated error. The event rate is 
    determined based on the timestamps of the events in the file.

    Parameters:
    -----------
    FILE_PATH : str
        Path to the HDF5 file containing the event data.

    Returns:
    --------
    tuple
        A tuple containing:
        - Rate_CV (float): The calculated event rate (events per second).
        - Rate_Err (float): The error in the calculated event rate.
        If the time difference is invalid or an error occurs, returns (-1, -1).
    """
    # HDF5 key to access the data
    h5_key = 'data'

    try:
        # Read the HDF5 file into a DataFrame
        data_df = pd.read_hdf(FILE_PATH, key=h5_key)

        # Check if the DataFrame is not empty and contains the 'timestamp' column
        if not data_df.empty and 'timestamp' in data_df.columns:
                
            # Sort the DataFrame by timestamp to ensure correct chronological order
            data_df = data_df.sort_values(by='timestamp').reset_index(drop=True)
            
            # Extract the first and last timestamps
            first_timestamp = data_df['timestamp'].iloc[0]
            last_timestamp = data_df['timestamp'].iloc[-1]
            
            time_difference = (last_timestamp - first_timestamp) * 0.001        # in [s]

            # Calculate the event count and its error
            evt_CV = len(data_df)
            evt_err = np.sqrt(evt_CV)

            # Ensure the time difference is valid before calculating the rate
            if time_difference > 0:
                Rate_CV  = evt_CV / time_difference
                Rate_Err = evt_err / time_difference

                return Rate_CV, Rate_Err
            else:
                # Invalid time difference, return error values
                return -1, -1

    # Handle specific errors
    except KeyError:
        print(f"Warning: HDF5 key '{h5_key}' not found in file {FILE_PATH}. Skipping this file.")
    except Exception as e:
        print(f"An error occurred while processing {FILE_PATH}: {e}")

    # Return error values in case of failure
    return -1, -1

def PLOT_CHARGE_HIST_FILE(FILE_PATH, PLOT_LABEL, SIGNAL_CHANNELS=[6, 7], X_LIMITS=[0, 2000]):
    """
    Helper function to plot the charge histogram from a single HDF5 file.

    This function reads charge data from an HDF5 file and generates a histogram 
    for the charge distribution of specified channels. The histogram includes 
    a control/noise channel and two active signal channels. The function also 
    allows setting custom x-axis limits for the histogram.

    Parameters:
    -----------
    FILE_PATH : str
        Path to the HDF5 file containing the charge data.
    PLOT_LABEL : str
        Label for the plot title, typically describing the dataset or file.
    SIGNAL_CHANNELS : list of int, optional
        List of two integers specifying the indices of the active signal channels 
        to be plotted. Default is [6, 7].
    X_LIMITS : list of int, optional
        List of two integers specifying the x-axis limits for the histogram. 
        Default is [0, 2000].

    Returns:
    --------
    None
        The function generates and displays a histogram plot. It does not return any value.
    """
    try:
        # Open the HDF5 file in read mode
        with h5py.File(FILE_PATH, "r") as f:

            # Check if the 'charges' dataset exists in the file
            if 'charges' not in f:
                print(f"Warning: 'charges' dataset not found in {FILE_PATH}.")
                return

            # Read the 'charges' dataset into a NumPy array
            charges_df = f["charges"][:]
                
            # Extract charge data for the control/noise channel and the two signal channels
            charge_control = charges_df[:, 5]
            charge_signal_1 = charges_df[:, SIGNAL_CHANNELS[0]]
            charge_signal_2 = charges_df[:, SIGNAL_CHANNELS[1]]  # Second signal channel

            # ----- Plotting ----- #
            # Define histogram bins with a fixed width of 10 ADC counts
            bins = np.arange(X_LIMITS[0], X_LIMITS[1] + 10, 10)

            # Set the plot style
            plt.style.use('ggplot')
            plt.figure(figsize=(10, 6))    
            
            # Plot histograms for each channel
            plt.hist(charge_control,  bins=bins, histtype='step', color='red',   linewidth=1.0, alpha=1.0, label='Noise')
            plt.hist(charge_signal_1, bins=bins, histtype='step', color='blue',  linewidth=1.0, alpha=1.0, label=f'Channel {SIGNAL_CHANNELS[0]} (Active)')
            plt.hist(charge_signal_2, bins=bins, histtype='step', color='green', linewidth=1.0, alpha=1.0, label=f'Channel {SIGNAL_CHANNELS[1]} (Active)')

            # Add labels, title, and legend
            plt.xlabel('Charge (ADC)')
            plt.xlim(X_LIMITS)
            plt.ylabel('Counts')
            plt.yscale('log')
            plt.legend()
            plt.title(f'{PLOT_LABEL}')
            
            # Add grid and adjust layout
            plt.grid(True, which='both', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()

    except Exception as e:
        # Handle any exceptions that occur during file reading or plotting
        print(f"An error occurred while processing {FILE_PATH}: {e}")