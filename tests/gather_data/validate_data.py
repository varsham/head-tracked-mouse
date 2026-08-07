"""
Filters low-quality frames out of calibration_points.csv before it's used
for training (Week 3: data cleaning).

Two filters:
  - Blinking: average EAR (left + right) below EAR_BLINK_THRESHOLD.
  - Poor pose fit: head-pose reprojection error in the worst
    REPROJECTION_ERROR_PERCENTILE of this session -- a proxy for partial
    occlusion or bad landmark detection, since a blocked/misdetected
    landmark makes the fitted rigid head model disagree with what was
    actually observed.

Usage: python validate_data.py [input_csv] [output_csv]
Defaults to calibration_points.csv -> calibration_points_clean.csv.
"""

import csv
import sys
from pathlib import Path

EAR_BLINK_THRESHOLD = 0.2
REPROJECTION_ERROR_PERCENTILE = 95  # drop the worst 5% of pose fits

DEFAULT_INPUT = "calibration_points.csv"
DEFAULT_OUTPUT = "calibration_points_clean.csv"


def load_rows(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def percentile(values, pct):
    if not values:
        return float("inf")
    values = sorted(values)
    idx = min(int(len(values) * pct / 100), len(values) - 1)
    return values[idx]


def validate(rows):
    reprojection_errors = [float(r["pose_reprojection_error"]) for r in rows]
    reprojection_cutoff = percentile(reprojection_errors, REPROJECTION_ERROR_PERCENTILE)

    kept, dropped_blink, dropped_obstructed = [], 0, 0
    for row in rows:
        ear_avg = (float(row["left_ear"]) + float(row["right_ear"])) / 2
        reprojection_error = float(row["pose_reprojection_error"])

        if ear_avg < EAR_BLINK_THRESHOLD:
            dropped_blink += 1
            continue
        if reprojection_error > reprojection_cutoff:
            dropped_obstructed += 1
            continue

        kept.append(row)

    return kept, dropped_blink, dropped_obstructed, reprojection_cutoff


def main():
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_INPUT)
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(DEFAULT_OUTPUT)

    rows, fieldnames = load_rows(input_path)
    kept, dropped_blink, dropped_obstructed, cutoff = validate(rows)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    total = len(rows)
    print(f"Read {total} frames from {input_path}")
    print(f"Dropped {dropped_blink} for blinking (avg EAR < {EAR_BLINK_THRESHOLD})")
    print(
        f"Dropped {dropped_obstructed} for poor pose fit "
        f"(reprojection error > {cutoff:.1f}px, "
        f"{REPROJECTION_ERROR_PERCENTILE}th percentile cutoff)"
    )
    print(f"Kept {len(kept)} frames ({len(kept) / total:.1%}) -> {output_path}")


if __name__ == "__main__":
    main()
