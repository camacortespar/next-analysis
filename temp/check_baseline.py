#!/usr/bin/env python3
"""Check the baseline of Irene-corrected PMT waveforms right before S2.

This script mimics Irene's exact pipeline for PMT waveforms (reading raw RD/pmtrwf, 
applying baseline subtraction, BLR deconvolution, and ADC-to-PE calibration), 
finds the S2 start from Irene's PMAPs, measures the 500 samples right before it, 
and checks if they are centered at 0.0 pes/sample.

Examples:
  python check_irene_baseline.py 16123 --detector next100 --max-events 200
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import tables as tb

ANALYSIS_ROOT = Path("/analysis")
PRODUCTION = {
    "demo": "v2.3.1-pyrrha/20250218",
    "next100": "v3.0.0/20260727",
}
DETECTOR_DB = {"demo": "demopp", "next100": "next100"}
DB_ALIASES = {"demoppdb": "demopp", "next100db": "next100"}
PMT_SAMPLE_NS = 25.0
DEFAULT_BASELINE_SAMPLES = 14000
FILE_KEY = re.compile(r"_(\d+)_ldc(\d+)_trg(\d+)", re.IGNORECASE)


def data_directories(args):
    base = (args.analysis_root or ANALYSIS_ROOT) / str(args.run) / "hdf5"
    production = args.production or PRODUCTION[args.detector]
    trigger = 0 if args.trigger is None else args.trigger
    raw = base / "data"
    irene = base / "prod" / production / "irene" / f"trigger{trigger}"
    return Path(args.raw_dir or raw), Path(args.irene_dir or irene)


def database_name_and_path(requested, load_db):
    database = DB_ALIASES.get(requested.lower(), requested.lower())
    if database not in ("demopp", "next100", "new", "flex100"):
        database = requested
    path = Path(load_db.get_db(database)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"IC database resolves to missing file {path}.")
    return database, path


def keyed_files(directory, run, ldc, trigger):
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    result = {}
    for path in directory.rglob("*.h5"):
        if f"run_{run}_" not in path.name:
            continue
        match = FILE_KEY.search(path.name)
        if match is None:
            continue
        key = tuple(map(int, match.groups()))
        if ldc is not None and key[1] != ldc:
            continue
        if trigger is not None and key[2] != trigger:
            continue
        result[key] = path
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=int, help="Run number")
    parser.add_argument("--detector", choices=("demo", "next100"), default="next100")
    parser.add_argument("--analysis-root", type=Path, help="Root directory")
    parser.add_argument("--production", help="Production version subpath")
    parser.add_argument("--database", help="IC DB selector")
    parser.add_argument("--ic-dir", type=Path, help="IC checkout directory")
    parser.add_argument("--raw-dir", type=Path, help="Raw directory override")
    parser.add_argument("--irene-dir", type=Path, help="Irene directory override")
    parser.add_argument("--ldc", type=int, help="Only process this LDC")
    parser.add_argument("--trigger", type=int, help="Only process this trigger number")
    parser.add_argument("--max-events", type=int, default=200, help="Max events to process")
    parser.add_argument("--samples-before", type=int, default=500, help="Samples before S2 start")
    parser.add_argument("--save-fig", type=Path, help="Save output plot PNG")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # --- Setup Invisible Cities Environment ---
    ic_dir = args.ic_dir
    if ic_dir is None and "ICTDIR" in os.environ:
        ic_dir = Path(os.environ["ICTDIR"])
    if ic_dir is None:
        spec = importlib.util.find_spec("invisible_cities")
        if spec is None or spec.origin is None:
            raise RuntimeError("No IC checkout found; set --ic-dir or ICTDIR")
        ic_dir = Path(spec.origin).resolve().parents[1]
    os.environ["ICTDIR"] = str(ic_dir)
    sys.path.insert(0, str(ic_dir))
    
    try:
        from invisible_cities.database import load_db
        from invisible_cities.sierpe import blr
    except ImportError as exc:
        raise RuntimeError("Could not import IC or BLR extension. Check your environment.") from exc

    database, _ = database_name_and_path(args.database or DETECTOR_DB[args.detector], load_db)
    pmt_db = load_db.DataPMT(database, args.run)
    lookup = pmt_db.set_index("SensorID")

    raw_dir, irene_dir = data_directories(args)
    raw_files = keyed_files(raw_dir, args.run, args.ldc, args.trigger)
    irene_files = keyed_files(irene_dir, args.run, args.ldc, args.trigger)
    
    common_keys = set(raw_files) & set(irene_files)
    if not common_keys:
        raise FileNotFoundError("No matching raw/Irene file pairs found.")

    dt_us = PMT_SAMPLE_NS / 1000.0
    n_samples = args.samples_before
    corrected_baselines = []

    print(f"[*] Processing Run {args.run}: Applying Irene's BLR & Gain calibration to check pre-S2 baseline...")

    events_processed = 0

    for key in sorted(common_keys):
        if events_processed >= args.max_events:
            break

        raw_path = raw_files[key]
        irene_path = irene_files[key]

        with tb.open_file(raw_path, "r") as raw, tb.open_file(irene_path, "r") as irene:
            if "/Run/events" not in raw or "/RD/pmtrwf" not in raw or "/PMAPS" not in irene:
                continue
            if not hasattr(irene.root.PMAPS, "S2"):
                continue

            evt_numbers = raw.root.Run.events.read(field="evt_number")
            raw_wfs_all = raw.root.RD.pmtrwf[:]  # Shape: (n_events, n_channels, n_samples)
            s2_table = irene.root.PMAPS.S2.read()
            
            # Fetch config values (like baseline sample size) from Irene's config node
            n_base = DEFAULT_BASELINE_SAMPLES
            if "/config/irene" in irene:
                cfg = {row["variable"].decode(): row["value"].decode() for row in irene.root.config.irene.read()}
                n_base = int(cfg.get("n_baseline", DEFAULT_BASELINE_SAMPLES))

            sensors = raw.root.Sensors.DataPMT.read()
            irene_events = np.unique(s2_table["event"])

            for row_idx, evt_num in enumerate(evt_numbers):
                if events_processed >= args.max_events:
                    break

                evt_int = int(evt_num)
                if evt_int not in irene_events:
                    continue

                evt_s2 = s2_table[s2_table["event"] == evt_int]
                if len(evt_s2) == 0:
                    continue

                # Find S2 start time
                first_peak = np.min(evt_s2["peak"])
                s2_bins = evt_s2[evt_s2["peak"] == first_peak]
                s2_start_us = float(s2_bins["time"][0] - s2_bins["bwidth"][0] / 2) / 1000.0

                s2_start_idx = int(s2_start_us / dt_us)
                end_idx = max(0, s2_start_idx)
                start_idx = max(0, end_idx - n_samples)

                if end_idx <= start_idx:
                    continue

                # --- REPLICATE IRENE'S BLR & CALIBRATION PIPELINE PER CHANNEL ---
                evt_raw_pmts = raw_wfs_all[row_idx] # shape: (n_channels, n_samples)
                
                for ch_idx, wf_raw in enumerate(evt_raw_pmts):
                    sensor = sensors[ch_idx]
                    sid = int(sensor["sensorID"])
                    
                    if sid not in lookup.index:
                        continue
                    calib = lookup.loc[sid]
                    if int(calib.Active) == 0:
                        continue
                        
                    coeff_c, coeff_blr = float(calib.coeff_c), float(calib.coeff_blr)
                    gain = float(calib.adc_to_pes)
                    
                    if not (coeff_c > 0 and coeff_blr > 0 and gain > 0):
                        continue

                    # 1. Compute pedestal from the first few samples (like Irene does)
                    pedestal = wf_raw[:min(n_base, wf_raw.size)].mean()
                    
                    # 2. Apply BLR deconvolution (pedestal - raw) inverted + BLR filter
                    corrected_wf = blr.deconvolve_signal(pedestal - wf_raw.astype(float), coeff_c, coeff_blr)
                    
                    # 3. Convert from ADC to photoelectrons (pes/sample)
                    pes_wf = np.asarray(corrected_wf, dtype=float) / gain

                    # 4. Extract the 500 samples right before S2 start
                    window_data = pes_wf[start_idx:end_idx]
                    if len(window_data) > 1:
                        corrected_baselines.append(np.mean(window_data))

                events_processed += 1

    if not corrected_baselines:
        print("[!] No corrected baseline data collected.")
        return

    print(f"[*] Extracted {len(corrected_baselines):,} channel-event pre-S2 baseline points using Irene's calibration.")

    # --- PLOTTING ---
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    bins = np.linspace(np.percentile(corrected_baselines, 1), np.percentile(corrected_baselines, 99), 100)

    ax.hist(
        corrected_baselines,
        bins=bins,
        histtype="stepfilled",
        color="#2b8cbe",
        alpha=0.7,
        edgecolor="black",
        label=f"Pre-S2 Corrected Baseline ({n_samples} samples)\nMean: {np.mean(corrected_baselines):.4f} pes/sample"
    )

    ax.set_title(f"Run {args.run} | Irene-Processed PMT Baseline Check (Pre-S2 Window)", fontsize=13)
    ax.set_xlabel("Corrected Baseline [pes / sample]", fontsize=11)
    ax.set_ylabel("Entries (Channels × Events)", fontsize=11)
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1.5, label="Ideal Zero Baseline")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("whitesmoke")

    if args.save_fig:
        fig.savefig(args.save_fig, dpi=180)
        print(f"Saved plot to {args.save_fig}")

    plt.show()

if __name__ == "__main__":
    main()