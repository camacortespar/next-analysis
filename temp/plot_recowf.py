#!/usr/bin/env python3
"""Browse decoded waveforms with Irene S1/S2 peak intervals shaded.

Examples (in an IC environment):
  python scripts/plot_recowf.py 15198 --detector demo --file-index 0
  python scripts/plot_recowf.py 15198 --detector demo --sipm --sensors 12000,12001
  python scripts/plot_recowf.py 16090 --pmt --ldc 2
  python scripts/plot_recowf.py 15198 --detector demo --events 1,4-6 --both --save-dir /tmp/reco

Press Enter in the plot window or terminal for the next event. Press q in the
plot, or q followed by Enter in the terminal, to quit. Ctrl+C also quits;
closing the window advances. PMTs are BLR-corrected and gain-calibrated as in Irene;
SiPMs use Irene's mode-baseline subtraction and gain calibration. With all
SiPMs selected their calibrated sum is shown by default; NEXT-100 PMTs also
default to a sum. This reads existing decoded and Irene files only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import re
import select
import sqlite3
import sys

import numpy as np
import tables as tb


# Site defaults. The root contains one directory per run; each hdf5/data and
# hdf5/prod tree may contain ldc1, ldc2, ... subdirectories.
ANALYSIS_ROOT = Path("/analysis")
PRODUCTION = {"demo": "v2.3.1-pyrrha/20250218",
              #"next100": "v2.3.1/20250219",
  	      "next100": "v3.0.0/20260727",
}
DETECTOR_DB = {"demo": "demopp", "next100": "next100"}
DB_ALIASES = {"demoppdb": "demopp", "next100db": "next100"}
PMT_SAMPLE_NS = 25.0
SIPM_SAMPLE_NS = 1000.0
DEFAULT_BASELINE_SAMPLES = 14000

FILE_KEY = re.compile(r"_(\d+)_ldc(\d+)_trg(\d+)", re.IGNORECASE)
COLORS = {"S1": "#2b8cbe", "S2": "#e58b23"}


@dataclass(frozen=True)
class FilePair:
    key: tuple[int, int, int]
    raw: Path
    irene: Path


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
        database = requested  # IC also accepts a path to a custom SQLite file.
    path = Path(load_db.get_db(database)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"IC database {requested!r} resolves to missing file {path}. "
            "Use demopp/DEMOPPDB, next100/NEXT100DB, or a SQLite file path."
        )
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        table = conn.execute("SELECT 1 FROM sqlite_master "
                             "WHERE type='table' AND name='ChannelPosition'").fetchone()
    if table is None:
        raise ValueError(f"{path} is not an IC detector database (no ChannelPosition table)")
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
        if key in result:
            raise ValueError(f"Duplicate files for {key}: {result[key]} and {path}")
        result[key] = path
    return result


def matched_files(args):
    raw_dir, irene_dir = data_directories(args)
    raw = keyed_files(raw_dir, args.run, args.ldc, args.trigger)
    irene = keyed_files(irene_dir, args.run, args.ldc, args.trigger)
    if not raw:
        raise FileNotFoundError(f"No decoded run-{args.run} waveform files under {raw_dir}")
    if not irene:
        raise FileNotFoundError(f"No run-{args.run} Irene files under {irene_dir}")
    keys = set(raw) & set(irene)
    if args.file_index is not None:
        keys = {key for key in keys if key[0] == args.file_index}
    if not keys:
        raise FileNotFoundError(
            "No matching decoded/Irene file keys (file index, LDC, trigger). "
            f"Raw: {raw_dir}; Irene: {irene_dir}"
        )
    if set(raw) != set(irene):
        print(f"Note: {len(set(raw) - set(irene))} decoded files lack Irene partners; "
              f"{len(set(irene) - set(raw))} Irene files lack decoded partners.", file=sys.stderr)
    return [FilePair(key, raw[key], irene[key]) for key in sorted(keys)]


def parse_event_spec(value):
    selected = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            first, last = map(int, item.split("-", 1))
            if last < first:
                raise ValueError(f"Invalid event range: {item}")
            selected.update(range(first, last + 1))
        else:
            selected.add(int(item))
    if not selected:
        raise ValueError("--events must contain at least one event number")
    return selected


def selected_events(pairs, args):
    requested = parse_event_spec(args.events) if args.events else None
    found = set()
    yielded = 0
    for pair in pairs:
        with tb.open_file(pair.raw) as h5:
            if "/Run/events" not in h5:
                raise ValueError(f"Missing /Run/events: {pair.raw}")
            event_numbers = h5.root.Run.events.read(field="evt_number")
        for row, event in enumerate(event_numbers):
            event = int(event)
            if requested is not None and event not in requested:
                continue
            if args.start_event is not None and event < args.start_event:
                continue
            if args.end_event is not None and event > args.end_event:
                continue
            found.add(event)
            yield pair, row, event
            yielded += 1
            if args.max_events is not None and yielded >= args.max_events:
                return
    if requested is not None and not found:
        print(f"No requested event numbers were found in matched files: {sorted(requested)}",
              file=sys.stderr)


def config_values(h5):
    if "/config/irene" not in h5:
        return {}
    return {row["variable"].decode(): row["value"].decode()
            for row in h5.root.config.irene.read()}


def sample_widths(config):
    return (float(config.get("pmt_samp_wid", PMT_SAMPLE_NS)),
            float(config.get("sipm_samp_wid", SIPM_SAMPLE_NS)))


def event_peak_intervals(h5, event):
    """Return one shaded span per PMAP peak, using rebinned bin widths."""
    result = []
    if "/PMAPS" not in h5:
        return result
    for kind in ("S1", "S2"):
        if not hasattr(h5.root.PMAPS, kind):
            continue
        table = getattr(h5.root.PMAPS, kind)
        records = table.read_where(f"event == {int(event)}")
        for peak in np.unique(records["peak"]):
            bins = records[records["peak"] == peak]
            # Irene's rebinned time is a charge-weighted point inside the bin.
            # Sum widths to recover the full peak duration, anchored at the
            # first bin's time minus half its width.
            start_ns = float(bins["time"][0] - bins["bwidth"][0] / 2)
            stop_ns = start_ns + float(bins["bwidth"].sum())
            result.append((kind, int(peak), start_ns / 1000, stop_ns / 1000,
                           float(bins["ene"].sum())))
    return result


def sensor_rows(h5, kind):
    name = "DataPMT" if kind == "pmt" else "DataSiPM"
    if f"/Sensors/{name}" not in h5:
        raise ValueError(f"Missing /Sensors/{name} in {h5.filename}")
    return getattr(h5.root.Sensors, name).read()


def choose_sensors(rows, requested, sensor_range, id_field):
    ids = rows[id_field].astype(int)
    mask = np.ones(len(ids), dtype=bool)
    if requested:
        chosen = {int(value) for value in requested.split(",")}
        missing = chosen - set(ids)
        if missing:
            raise ValueError(f"Unknown {id_field} value(s): {sorted(missing)}")
        mask &= np.isin(ids, sorted(chosen))
    if sensor_range:
        lo, hi = sensor_range
        if hi < lo:
            raise ValueError("--sensor-range needs LOW <= HIGH")
        mask &= (ids >= lo) & (ids <= hi)
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        raise ValueError(f"No {id_field} values match the sensor selection")
    return indices


def pmt_waveforms(h5, row, indices, sensors, db_frame, n_baseline, blr,
                  skip_uncalibrated=False):
    if "/RD/pmtrwf" not in h5:
        raise ValueError(f"Missing /RD/pmtrwf in {h5.filename}")
    raw = h5.root.RD.pmtrwf[row, indices, :].astype(float)
    lookup = db_frame.set_index("SensorID")
    output = []
    plotted_ids = []
    omitted = []
    for wf, sensor in zip(raw, sensors):
        sid = int(sensor["sensorID"])
        if sid not in lookup.index:
            if skip_uncalibrated:
                omitted.append((sid, "absent from database"))
                continue
            raise ValueError(f"PMT {sid} is absent from the selected detector database; "
                             "use --skip-uncalibrated to omit it")
        calib = lookup.loc[sid]
        if int(calib.Active) == 0:
            omitted.append((sid, "inactive in database"))
            continue
        coeff_c, coeff_blr = float(calib.coeff_c), float(calib.coeff_blr)
        gain = float(calib.adc_to_pes)
        valid = (np.all(np.isfinite((coeff_c, coeff_blr, gain)))
                 and coeff_c > 0 and coeff_blr > 0 and gain > 0)
        if not valid:
            detail = (f"gain={gain}, coeff_c={coeff_c}, coeff_blr={coeff_blr}, "
                      f"Active={int(calib.Active)}")
            if skip_uncalibrated:
                omitted.append((sid, detail))
                continue
            raise ValueError(f"PMT {sid} has invalid calibration ({detail}); "
                             "check --ic-dir/ICTDIR or use --skip-uncalibrated")
        pedestal = wf[:min(n_baseline, wf.size)].mean()
        corrected = blr.deconvolve_signal(pedestal - wf, coeff_c, coeff_blr)
        output.append(np.asarray(corrected, dtype=float) / gain)
        plotted_ids.append(sid)
    if not output:
        raise ValueError("No active, calibrated PMTs remain in this selection")
    return np.asarray(plotted_ids), np.asarray(output), omitted


def sipm_waveforms(h5, row, indices, sensors, db_frame, csf):
    if "/RD/sipmrwf" not in h5:
        raise ValueError(f"Missing /RD/sipmrwf in {h5.filename}")
    raw = h5.root.RD.sipmrwf[row, indices, :]
    lookup = db_frame.set_index("SensorID")
    gains = []
    for sensor in sensors:
        sid = int(sensor["sensorID"])
        if sid not in lookup.index:
            raise ValueError(f"SiPM {sid} is absent from the selected detector database")
        gains.append(abs(float(lookup.loc[sid].adc_to_pes)))
    return csf.calibrate_sipms(raw, np.asarray(gains))


def plot_event(run, event, key, traces, spans, widths_ns, sum_traces,
               show_trigger, time_window, title_note=""):
    import matplotlib.pyplot as plt

    kinds = list(traces)
    fig, axes = plt.subplots(len(kinds), 1, figsize=(13, 5 if len(kinds) == 1 else 8),
                             squeeze=False, constrained_layout=True)
    for ax, kind in zip(axes[:, 0], kinds):
        ids, waveforms = traces[kind]
        dt_us = widths_ns[kind] / 1000
        x = np.arange(waveforms.shape[1]) * dt_us
        if sum_traces[kind]:
            plotted = np.sum(waveforms, axis=0)
            ax.plot(x, plotted, color="#242424", lw=.85,
                    label=f"sum of {len(ids)} {kind.upper()}s")
        else:
            for sid, wf in zip(ids, waveforms):
                ax.plot(x, wf, lw=.75, label=f"{kind.upper()} {sid}")
        seen = set()
        for peak_kind, peak_index, start, stop, energy in spans:
            ax.axvspan(start, stop, color=COLORS[peak_kind], alpha=.22,
                       label=peak_kind if peak_kind not in seen else None)
            seen.add(peak_kind)
        if show_trigger is not None:
            ax.axvline(show_trigger, color="#b63342", lw=1, ls="--", label="trigger")
        if time_window:
            ax.set_xlim(*time_window)
        else:
            ax.set_xlim(0, waveforms.shape[1] * dt_us)
        ax.set(ylabel="charge [pes / sample]", title=f"{kind.upper()} reconstructed waveform")
        ax.grid(alpha=.18)
        #if len(ids) <= 15 or sum_traces[kind]:
        ax.legend(loc="upper right", fontsize=7, ncol=5, facecolor="white", framealpha=0.8)
    axes[-1, 0].set_xlabel("waveform time [us]")
    count_s1 = sum(span[0] == "S1" for span in spans)
    count_s2 = sum(span[0] == "S2" for span in spans)
    fig.suptitle(f"Run {run} | file {key[0]:04d}, LDC {key[1]}, trigger {key[2]} | "
                 f"event {event} | Irene: {count_s1} S1, {count_s2} S2{title_note}")
    return fig


def browse_event(fig):
    import matplotlib.pyplot as plt

    stopped = False

    def on_key(pressed):
        nonlocal stopped
        if pressed.key in ("enter", "return", "q", "escape"):
            stopped = pressed.key in ("q", "escape")
            plt.close(fig)

    callback = fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show(block=False)
    if sys.stdin.isatty():
        print("Enter: next event | q + Enter: quit | Ctrl+C: quit", flush=True)
    read_stdin = True
    try:
        while plt.fignum_exists(fig.number):
            plt.pause(0.1)
            if not read_stdin:
                continue
            try:
                ready = select.select([sys.stdin], [], [], 0)[0]
            except (OSError, ValueError):
                read_stdin = False
                continue
            if ready:
                command = sys.stdin.readline()
                if not command:
                    read_stdin = False
                elif command.strip().lower() in ("q", "quit", "exit"):
                    stopped = True
                    break
                elif not command.strip():
                    break
    except KeyboardInterrupt:
        stopped = True
        print("\nStopped.", file=sys.stderr)
    finally:
        fig.canvas.mpl_disconnect(callback)
        plt.close(fig)
    return stopped


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=int, help="Run number")
    parser.add_argument("--detector", choices=("demo", "next100"), default="next100")
    parser.add_argument("--analysis-root", type=Path,
                        help="root containing run directories (default: /analysis)")
    parser.add_argument("--production", help="production version/date subpath under "
                        "hdf5/prod (default depends on detector)")
    parser.add_argument("--database", help="IC DB selector (demopp/DEMOPPDB or "
                        "next100/NEXT100DB), or SQLite path; defaults by detector")
    parser.add_argument("--ic-dir", type=Path,
                        help="IC checkout; defaults to ICTDIR or installed IC")
    parser.add_argument("--raw-dir", type=Path, help="decoded waveform directory")
    parser.add_argument("--irene-dir", type=Path, help="Irene directory")
    parser.add_argument("--ldc", type=int, help="only this LDC")
    parser.add_argument("--trigger", type=int, help="only this trigger number")
    parser.add_argument("--file-index", type=int, help="only this file index")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pmt", action="store_true", help="show PMT waveforms (default)")
    group.add_argument("--sipm", action="store_true", help="show SiPM waveforms")
    group.add_argument("--both", action="store_true", help="show PMT and SiPM panels")
    parser.add_argument("--sensors", help="comma-separated physical sensor IDs")
    parser.add_argument("--sensor-range", nargs=2, type=int, metavar=("LOW", "HIGH"),
                        help="inclusive physical sensor ID range")
    parser.add_argument("--channel-ids", action="store_true",
                        help="interpret --sensors and --sensor-range as electronics channels")
    parser.add_argument("--skip-uncalibrated", action="store_true",
                        help="omit PMTs with missing/invalid calibration; inactive PMTs are always omitted")
    parser.add_argument("--sum", action="store_true", help="sum the selected calibrated sensors")
    parser.add_argument("--overlay", action="store_true",
                        help="overlay individual sensors instead of default full-detector sums")
    parser.add_argument("--events", help="event numbers, for example 1,4-7,12")
    parser.add_argument("--start-event", type=int, help="first event number (inclusive)")
    parser.add_argument("--end-event", type=int, help="last event number (inclusive)")
    parser.add_argument("--max-events", type=int, help="stop after this many events")
    parser.add_argument("--trigger-time", type=float, help="optional trigger line in us")
    parser.add_argument("--time-window", nargs=2, type=float, metavar=("START", "STOP"),
                        help="x-axis limits in us")
    parser.add_argument("--save-dir", type=Path, help="also save one PNG per event")
    parser.add_argument("--no-show", action="store_true", help="save PNGs without opening a GUI")
    args = parser.parse_args(argv)
    if args.no_show and args.save_dir is None:
        parser.error("--no-show requires --save-dir")
    if args.max_events is not None and args.max_events < 1:
        parser.error("--max-events must be positive")
    if args.time_window and args.time_window[1] <= args.time_window[0]:
        parser.error("--time-window requires START < STOP")
    if args.sum and args.overlay:
        parser.error("--sum and --overlay cannot be combined")
    if args.production:
        production = Path(args.production)
        if production.is_absolute() or ".." in production.parts:
            parser.error("--production must be a relative version/date subpath")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
        from invisible_cities.calib import calib_sensors_functions as csf
    except ImportError as exc:
        raise RuntimeError("Could not import IC and its BLR extension. Activate the IC "
                           "environment and set --ic-dir to a compiled IC checkout.") from exc

    database, db_path = database_name_and_path(
        args.database or DETECTOR_DB[args.detector], load_db)
    need_pmt = not args.sipm
    need_sipm = args.sipm or args.both
    if args.both and (args.sensors or args.sensor_range):
        raise ValueError("Sensor selection with --both is ambiguous; use one sensor type at a time")
    pmt_db = load_db.DataPMT(database, args.run) if need_pmt else None
    sipm_db = load_db.DataSiPM(database, args.run) if need_sipm else None
    pairs = matched_files(args)
    print(f"Matched {len(pairs)} decoded/Irene file pairs; "
          f"database={database} ({db_path}), IC={ic_dir}.")
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    reported_omissions = set()
    for pair, row, event in selected_events(pairs, args):
        with tb.open_file(pair.raw) as raw, tb.open_file(pair.irene) as irene:
            config = config_values(irene)
            pmt_ns, sipm_ns = sample_widths(config)
            spans = event_peak_intervals(irene, event)
            traces = {}
            omitted = []
            id_field = "channel" if args.channel_ids else "sensorID"
            if need_pmt:
                sensors = sensor_rows(raw, "pmt")
                indices = choose_sensors(sensors, args.sensors, args.sensor_range, id_field)
                selected = sensors[indices]
                n_base = int(config.get("n_baseline", DEFAULT_BASELINE_SAMPLES))
                ids, wfs, omitted = pmt_waveforms(
                    raw, row, indices, selected, pmt_db, n_base, blr,
                    args.skip_uncalibrated)
                traces["pmt"] = (ids, wfs)
                for sid, reason in omitted:
                    if (sid, reason) not in reported_omissions:
                        print(f"Warning: omitting PMT {sid}: {reason}", file=sys.stderr)
                        reported_omissions.add((sid, reason))
            if need_sipm:
                sensors = sensor_rows(raw, "sipm")
                indices = choose_sensors(sensors, args.sensors, args.sensor_range, id_field)
                selected = sensors[indices]
                wfs = sipm_waveforms(raw, row, indices, selected, sipm_db, csf)
                traces["sipm"] = (selected["sensorID"].astype(int), wfs)
            in_irene = event in set(irene.root.Run.events.read(field="evt_number"))

        if not in_irene:
            note = " | event absent from Irene"
        elif not spans:
            note = " | no Irene peaks"
        else:
            note = ""
        if omitted:
            ids = ",".join(str(sid) for sid, _ in omitted[:3])
            note += f" | omitted PMT {ids}"
            if len(omitted) > 3:
                note += f" +{len(omitted) - 3}"
        sum_traces = {
            kind: args.sum or (not args.overlay and not args.sensors
                               and not args.sensor_range
                               and (kind == "sipm" or args.detector == "next100"))
            for kind in traces
        }
        fig = plot_event(args.run, event, pair.key, traces, spans,
                         {"pmt": pmt_ns, "sipm": sipm_ns},
                         sum_traces, args.trigger_time, args.time_window, note)

        if "pmt" in traces and not sum_traces.get("pmt", False):
            ids_pmt, wfs_pmt = traces["pmt"]
            print("   [Interactive] Press Enter to view PMTs one by one, or 'q' to quit/skip.")
            
            for idx, (sid, wf) in enumerate(zip(ids_pmt, wfs_pmt)):
                fig_single, ax_single = plt.subplots(figsize=(12, 4), constrained_layout=True)
                dt_us = pmt_ns / 1000.0
                x = np.arange(len(wf)) * dt_us
                
                ax_single.plot(x, wf, color="#2b8cbe", lw=1.0, label=f"PMT {sid}")
                
                seen_peaks = set()
                for peak_kind, peak_index, start, stop, energy in spans:
                    ax_single.axvspan(start, stop, color=COLORS[peak_kind], alpha=0.22,
                                     label=peak_kind if peak_kind not in seen_peaks else None)
                    seen_peaks.add(peak_kind)
                
                ax_single.set(ylabel="charge [pes / sample]",
                              title=f"Run {args.run} | Event {event} | PMT ID: {sid} (Channel {idx+1}/{len(ids_pmt)})",
                              xlabel="waveform time [us]")
                ax_single.grid(alpha=0.18)
                ax_single.legend(loc="upper right", fontsize=8)
                
                stopped_chan = browse_event(fig_single)
                if stopped_chan:
                    break

        if args.save_dir:
            target = args.save_dir / (f"run{args.run}_file{pair.key[0]:04d}_"
                                      f"ldc{pair.key[1]}_trg{pair.key[2]}_event{event}.png")
            fig.savefig(target, dpi=160)
            print(f"  saved {target}")
        if args.no_show:
            plt.close(fig)
            continue

        if browse_event(fig):
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"plot_recowf: {exc}")
