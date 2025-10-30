#!/usr/bin/env python3
"""
plot_loss_from_log.py

Parse a training log file and plot Training Loss vs Epoch and vs Time.

Usage examples:
  python plot_loss_from_log.py --log models/DETR-4/log.txt --out losses.png --show

The script looks for lines like:
  2025-09-14 11:08:34,510 - INFO - Epoch [1/300] - Training Loss: 3.3135

It extracts timestamp, epoch number and training loss.
"""

import re
import argparse
from datetime import datetime
from pathlib import Path
import sys

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
except Exception as e:
    print("Missing required packages. Please install with: pip install matplotlib pandas numpy")
    raise


LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .*Epoch \[(?P<epoch>\d+)/(?:\d+)\] - Training Loss: (?P<loss>[0-9.]+)"
)


def parse_log(path: Path):
    records = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            ts = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S,%f")
            epoch = int(m.group("epoch"))
            loss = float(m.group("loss"))
            records.append({"ts": ts, "epoch": epoch, "loss": loss})
    if not records:
        raise ValueError(f"No matching records found in {path}")
    df = pd.DataFrame(records)
    df = df.sort_values(["ts", "epoch"]).reset_index(drop=True)
    return df


def smooth(series, window: int):
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1, center=True).mean()


def plot(df, out: Path = None, show: bool = False, smooth_window: int = 1):
    # Try seaborn style if available, otherwise fallback to a default style
    try:
        plt.style.use("seaborn-darkgrid")
    except Exception:
        try:
            plt.style.use("seaborn")
        except Exception:
            plt.style.use("classic")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss vs Epoch
    ax = axes[0]
    ax.plot(df['epoch'], df['loss'], marker='o', linestyle='-', alpha=0.6, label='raw')
    if smooth_window > 1:
        ax.plot(df['epoch'], smooth(df['loss'], smooth_window), color='red', lw=2, label=f'smoothed ({smooth_window})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss vs Epoch')
    ax.legend()

    # Loss vs Time
    ax = axes[1]
    ax.plot(df['ts'], df['loss'], marker='o', linestyle='-', alpha=0.6, label='raw')
    if smooth_window > 1:
        ax.plot(df['ts'], smooth(df['loss'], smooth_window), color='red', lw=2, label=f'smoothed ({smooth_window})')
    ax.set_xlabel('Time')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss vs Time')
    ax.legend()

    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    if out:
        fig.savefig(out, dpi=150)
        print(f"Saved plot to {out}")
    if show:
        plt.show()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot training loss from log file")
    parser.add_argument("--log", "-l", type=Path, default=Path("models/DETR-4/log.txt"), help="Path to log.txt")
    parser.add_argument("--out", "-o", type=Path, default=None, help="Output image file (png) to save")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    parser.add_argument("--smooth", type=int, default=1, help="Smoothing window (integer > 1)")
    args = parser.parse_args(argv)

    if not args.log.exists():
        print(f"Log file not found: {args.log}")
        sys.exit(2)

    df = parse_log(args.log)
    plot(df, out=args.out, show=args.show, smooth_window=args.smooth)


if __name__ == '__main__':
    main()
