"""Pull NWM operational short-range streamflow for a few reaches via CIROH Kerchunk refs.

Hand check for docs/research-plan.md (Phase 2, step 9), not package code. Output
has the columns the plan asks for (section 6.2): reference_time, lead_time, time,
feature_id, streamflow (m3/s), source_product. By default each cycle also gets
its t0 row from analysis_assim tm00 (lead_time 0), as in docs/ROADMAP.md.

Usage:
    uv run python scripts/pull_nwm_short_range.py                     # 2026-04-19 t00z
    uv run python scripts/pull_nwm_short_range.py --date 20260419 --cycle 0 --cycle 12 \
        --reach 18229923 --reach 18239649 --out flow.csv

Why not xarray/zarr: on Windows, zarr's async HTTP reads crash the interpreter on
exit (segfault), and zarr refuses a synchronous file system. The refs point at
single-chunk NetCDF variables, so we fetch each chunk through a synchronous
ReferenceFileSystem and decode it with numcodecs, using the codecs in the refs.
"""

import argparse
import json
import time as timer
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numcodecs
import numpy as np
import pandas as pd
import s3fs
from fsspec.implementations.reference import ReferenceFileSystem

REFS = "ciroh-nwm-zarr-copy/national-water-model"
DEFAULT_REACHES = [18229923, 18239649]  # Black Warrior, Alabama River
s3 = s3fs.S3FileSystem(anon=True)


def ref_path(date: str, product: str, cycle: int, step: str) -> str:
    return f"{REFS}/nwm.{date}/{product}/nwm.t{cycle:02d}z.{product}.channel_rt.{step}.conus.nc.json"


def _meta(refs: dict, key: str) -> dict:
    value = refs[key]
    return json.loads(value) if isinstance(value, str) else value


def read_array(rfs: ReferenceFileSystem, refs: dict, name: str) -> tuple[np.ndarray, dict]:
    """Read one 1-D variable as raw values plus its attributes (no scaling or masking)."""
    zarray = _meta(refs, f"{name}/.zarray")
    n_chunks = -(-zarray["shape"][0] // zarray["chunks"][0])
    parts = []
    for i in range(n_chunks):
        buf = rfs.cat(f"{name}/{i}")
        if zarray["compressor"]:
            buf = numcodecs.get_codec(zarray["compressor"]).decode(buf)
        for codec in reversed(zarray["filters"] or []):
            buf = numcodecs.get_codec(codec).decode(buf)
        parts.append(np.frombuffer(buf, dtype=zarray["dtype"]))
    values = np.concatenate(parts)[: zarray["shape"][0]]
    attrs = _meta(refs, f"{name}/.zattrs")
    attrs.setdefault("_fill_value", zarray["fill_value"])
    return values, attrs


def decode_time(values: np.ndarray, attrs: dict) -> pd.DatetimeIndex:
    """CF time, e.g. 'minutes since 1970-01-01 00:00:00 UTC', to UTC timestamps."""
    unit, _, epoch = attrs["units"].partition(" since ")
    unit = {"seconds": "s", "minutes": "min", "hours": "h", "days": "D"}[unit]
    return pd.Timestamp(epoch.removesuffix(" UTC")) + pd.to_timedelta(values, unit=unit)


def decode_flow(values: np.ndarray, attrs: dict) -> np.ndarray:
    """int32 counts to m3/s, with fill and missing values (and anything negative) as NaN."""
    bad = {attrs.get("_fill_value"), attrs.get("_FillValue"), attrs.get("missing_value")} - {None}
    flow = values.astype("float64") * attrs.get("scale_factor", 1.0) + attrs.get("add_offset", 0.0)
    flow[np.isin(values, list(bad)) | (flow < 0)] = np.nan
    return flow.round(2)  # stored precision is 0.01; drops float32 scale_factor noise


class Reader:
    """Reads one channel_rt file per call; looks up reach positions once and reuses them."""

    def __init__(self, reaches: list[int]):
        self.reaches = np.asarray(reaches, dtype="int64")
        self.feature_ids = None
        self.positions = None

    def _positions(self, rfs, refs) -> np.ndarray:
        n = _meta(refs, "feature_id/.zarray")["shape"][0]
        if self.feature_ids is None or len(self.feature_ids) != n:
            self.feature_ids, _ = read_array(rfs, refs, "feature_id")
            pos = np.searchsorted(self.feature_ids, self.reaches)
            pos = np.clip(pos, 0, n - 1)
            missing = self.reaches[self.feature_ids[pos] != self.reaches]
            if len(missing):
                raise ValueError(f"feature_id not in NWM operational CONUS channel_rt: {missing.tolist()}")
            self.positions = pos
        return self.positions

    def read(self, path: str, product: str, cycle_time: pd.Timestamp) -> pd.DataFrame:
        ref = json.loads(s3.cat(path))
        refs = ref.get("refs", ref)  # version 1 refs nest under "refs"; version 0 is flat
        rfs = ReferenceFileSystem({"version": 1, "refs": refs}, remote_protocol="https")
        pos = self._positions(rfs, refs)
        valid = decode_time(*read_array(rfs, refs, "time"))[0]
        raw, attrs = read_array(rfs, refs, "streamflow")
        return pd.DataFrame({
            "reference_time": cycle_time,
            # Measured from the forecast cycle: analysis files carry a reference_time
            # 3 h before the cycle (start of the look-back window), not the cycle itself.
            "lead_time": int((valid - cycle_time) / pd.Timedelta(hours=1)),
            "time": valid,
            "feature_id": self.reaches,
            "streamflow": decode_flow(raw[pos], attrs),
            "source_product": product,
        })


def fetch_short_range(
    reaches: list[int],
    date: str,
    cycles: list[int],
    lead_times: list[int],
    t0: bool = True,
    workers: int = 8,
) -> pd.DataFrame:
    """Long-format frame: reference_time, lead_time, time (valid, UTC), feature_id,
    streamflow (m3/s), source_product. Missing files are reported and skipped."""
    jobs = []
    for cycle in cycles:
        cycle_time = pd.Timestamp(date) + pd.Timedelta(hours=cycle)
        steps = [("analysis_assim", "tm00")] if t0 else []
        steps += [("short_range", f"f{h:03d}") for h in lead_times]
        jobs += [(ref_path(date, product, cycle, step), product, cycle_time) for product, step in steps]

    reader = Reader(reaches)

    def run(job):
        try:
            return reader.read(*job)
        except FileNotFoundError:
            print(f"missing, skipped: {job[0]}")
            return None

    # Read one file first so reach positions are resolved before going parallel.
    frames = [run(jobs[0])]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        frames += pool.map(run, jobs[1:])
    frames = [f for f in frames if f is not None]
    if not frames:
        raise RuntimeError("no files read")
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["reference_time", "feature_id", "lead_time"], ignore_index=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--date", default="20260419", help="cycle date, YYYYMMDD (default: the 2026-04-19 Alabama event)")
    p.add_argument("--cycle", type=int, action="append", help="cycle hour 0-23, repeatable (default: 0)")
    p.add_argument("--all-cycles", action="store_true", help="all 24 hourly cycles")
    p.add_argument("--lead", type=int, action="append", help="forecast hour 1-18, repeatable (default: all)")
    p.add_argument("--reach", type=int, action="append", help="NWM feature_id (COMID), repeatable")
    p.add_argument("--no-t0", action="store_true", help="skip the analysis_assim tm00 row per cycle")
    p.add_argument("--workers", type=int, default=8, help="files read in parallel (default: 8)")
    p.add_argument("--out", help="output .csv or .parquet (default: nwm_short_range_<date>.csv)")
    args = p.parse_args(argv)

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    if args.date >= today:
        p.error(f"refs exist up to yesterday ({(datetime.now(timezone.utc) - timedelta(days=1)):%Y%m%d}); "
                "use the native files on s3://noaa-nwm-pds for today")
    out = args.out or f"nwm_short_range_{args.date}.csv"
    if out.endswith(".parquet"):
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            p.error("Parquet output needs pyarrow: run `uv add pyarrow`, or use a .csv name")
    cycles = list(range(24)) if args.all_cycles else (args.cycle or [0])
    leads = args.lead or list(range(1, 19))
    reaches = args.reach or DEFAULT_REACHES

    start = timer.perf_counter()
    df = fetch_short_range(reaches, args.date, cycles, leads, t0=not args.no_t0, workers=args.workers)
    seconds = timer.perf_counter() - start

    if out.endswith(".parquet"):
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    n_files = df.groupby(["reference_time", "lead_time"]).ngroups
    print(df.head(12).to_string(index=False))
    print(f"\n{len(df)} rows from {n_files} files in {seconds:.1f} s "
          f"({df.streamflow.isna().sum()} missing values) -> {out}")


if __name__ == "__main__":
    main()
