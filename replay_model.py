"""Replay one archived six-fold study method with its original validation cutoff."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main():
    with (ROOT / "runs/rankings/local_per_design.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=sorted({row["method"] for row in rows}))
    parser.add_argument("--design", default="RISCY-FPU-b", choices=sorted({row["design"] for row in rows}))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--out", type=Path, help="New JSON file, relative to the project root or absolute")
    parser.add_argument("--print-command", action="store_true", help="Show the underlying evaluation command and exit")
    args = parser.parse_args()

    row = next(row for row in rows if row["method"] == args.method and row["design"] == args.design)
    report = json.loads((ROOT / row["source"]).read_text())
    if args.out is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        args.out = Path("runs/replays") / f"{args.design}_{args.method}_{stamp}.json"
    command = [sys.executable, "compare_hotspot_methods.py", "--split", "test", "--device", args.device,
               "--methods", args.method, "--thresholds-from", report["thresholds_from"], "--out", str(args.out)]
    for role, folder in report["runs"].items():
        command.extend(["--" + role.replace("_", "-"), folder])
    if args.print_command:
        print(shlex.join(command))
        return
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
