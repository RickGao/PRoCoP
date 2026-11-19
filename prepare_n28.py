"""Prepare paired training maps directly from the CircuitNet-N28 archives."""

import argparse
import csv
import gzip
import io
import re
import tarfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

import numpy as np
from scipy.ndimage import zoom


ARCHIVES = {
    "macro_region.tar.gz": {"macro_region"},
    "RUDY/RUDY.tar.gz": {"RUDY"},
    "RUDY/RUDY_pin.tar.gz": {"RUDY_pin"},
    "congestion/congestion_global_routing/overflow_based.tar.gz": {
        "congestion_GR_horizontal_overflow", "congestion_GR_vertical_overflow",
    },
}
FEATURES = ("macro_region", "RUDY", "RUDY_pin")
LABELS = ("congestion_GR_horizontal_overflow", "congestion_GR_vertical_overflow")
SAMPLE_NAME = re.compile(
    r"(?:\d+-)?((?:RISCY-FPU|RISCY|zero-riscy)-[ab])-\d+"
    r"-c[\d.]+-u[\d.]+-m(?:\d+|None)-p\d+-f[01](?:\.npy)?"
)


@contextmanager
def open_archive(path):
    with gzip.open(path, "rb") as outer:
        # Some copies have one layer removed already.
        mode = "r|gz" if outer.peek(2)[:2] == b"\x1f\x8b" else "r|"
        with tarfile.open(fileobj=outer, mode=mode) as archive:
            yield archive


def catalog(root):
    designs = defaultdict(list)
    with open_archive(root / "macro_region.tar.gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if member.isfile() and path.parent.name == "macro_region" and not path.name.startswith("."):
                match = SAMPLE_NAME.fullmatch(path.name)
                if not match:
                    raise ValueError(f"Unrecognized CircuitNet-N28 sample name: {member.name}")
                designs[match[1]].append(path.name.removesuffix(".npy"))
    if len(designs) < 3:
        raise ValueError(f"Expected at least three design groups, found {sorted(designs)}")
    for design, names in designs.items():
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate sample names for {design}")
    return designs


def read_selected(path, channels, names, cache):
    found = {channel: set() for channel in channels}
    with open_archive(path) as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            channel = name.parent.name
            sample = name.name.removesuffix(".npy")
            if not member.isfile() or channel not in channels or sample not in names:
                continue
            if sample in found[channel]:
                raise ValueError(f"Duplicate {channel} for {name.stem}")
            with archive.extractfile(member) as file:
                array = np.load(io.BytesIO(file.read()), allow_pickle=False)
            if array.ndim != 2 or not np.isfinite(array).all():
                raise ValueError(f"Expected a finite 2D array in {member.name}")
            np.save(cache / channel / sample, array, allow_pickle=False)
            found[channel].add(sample)
    for channel, present in found.items():
        missing = names - present
        if missing:
            raise ValueError(f"{channel} is missing {len(missing)} selected samples, e.g. {min(missing)}")
    print(f"Read {sum(map(len, found.values()))} maps from {path.name}", flush=True)


def normalize(array):
    low, high = array.min(), array.max()
    if high == low:
        return np.zeros(array.shape, dtype=np.float32)
    return ((array - low) / (high - low)).astype(np.float32)


def prepare(root, output, limit, seed):
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    for archive in ARCHIVES:
        if not (root / archive).is_file():
            raise FileNotFoundError(root / archive)
    designs = catalog(root)
    rng = np.random.default_rng(seed)
    rows = []
    for design, names in sorted(designs.items()):
        ordered = sorted(names)
        selected = (
            ordered if limit is None or limit >= len(ordered)
            else sorted(rng.choice(ordered, limit, replace=False))
        )
        rows.extend((design, str(name)) for name in selected)
        print(f"{design}: {len(selected)} / {len(ordered)} layouts", flush=True)
    selected_names = {name for _, name in rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".prepare-", dir=output.parent) as tmp:
        cache = Path(tmp) / "raw"
        staging = Path(tmp) / "prepared"
        for channel in (*FEATURES, *LABELS):
            (cache / channel).mkdir(parents=True)
        for folder in ("feature", "label"):
            (staging / folder).mkdir(parents=True)
        with ThreadPoolExecutor(max_workers=4) as threads:
            scans = [
                threads.submit(read_selected, root / archive, channels, selected_names, cache)
                for archive, channels in ARCHIVES.items()
            ]
            for scan in scans:
                scan.result()
        with (staging / "manifest.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("design", "feature", "label"))
            for design, name in rows:
                maps = [
                    np.load(cache / channel / f"{name}.npy", allow_pickle=False)
                    for channel in (*FEATURES, *LABELS)
                ]
                if len({array.shape for array in maps}) != 1:
                    raise ValueError(f"Input and label grids differ for {name}")
                resized = [
                    zoom(array, tuple(256 / size for size in array.shape), order=3)
                    for array in maps
                ]
                features = np.stack([normalize(array) for array in resized[:3]], axis=-1)
                label = normalize(resized[3] + resized[4])[..., None]
                feature_path, label_path = f"feature/{name}.npy", f"label/{name}.npy"
                np.save(staging / feature_path, features, allow_pickle=False)
                np.save(staging / label_path, label, allow_pickle=False)
                writer.writerow((design, feature_path, label_path))
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
    print(f"Prepared {len(rows)} layouts: {output / 'manifest.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="routability_features directory")
    parser.add_argument("--out", type=Path, required=True, help="new processed data directory")
    parser.add_argument("--per-design", type=int, help="limit each design for a pilot run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.per_design is not None and args.per_design < 1:
        parser.error("--per-design must be positive")
    prepare(args.source, args.out, args.per_design, args.seed)


if __name__ == "__main__":
    main()
