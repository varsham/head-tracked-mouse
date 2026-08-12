"""
Filters low-quality frames out of calibration_points.csv before it's used
for training (Week 3: data cleaning).

Three filters:
  - Blinking: average EAR (left + right) below EAR_BLINK_THRESHOLD.
  - Poor pose fit: head-pose reprojection error in the worst
    REPROJECTION_ERROR_PERCENTILE of this session -- a proxy for partial
    occlusion or bad landmark detection, since a blocked/misdetected
    landmark makes the fitted rigid head model disagree with what was
    actually observed.
  - Head drift: pitch/yaw too far from that point's own median pitch/yaw --
    since the head is meant to stay still now (eyes are the only feature
    driving gaze prediction), a frame where the head visibly moved despite
    the live warning in calibration_dots.py is bad training data.

Usage: python validate_data.py [input_csv] [output_csv]
Defaults to calibration_points.csv -> calibration_points_clean.csv.
"""

import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

EAR_BLINK_THRESHOLD = 0.2
REPROJECTION_ERROR_PERCENTILE = 95  # drop the worst 5% of pose fits
DRIFT_THRESHOLD_DEG = 5.0  # max allowed pitch/yaw distance from a point's own median

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


def compute_point_baselines(rows):
    """Median (pitch, yaw) per point_index -- each point's own "head still" reference."""
    by_point = defaultdict(lambda: {"pitch": [], "yaw": []})
    for r in rows:
        point_id = int(r["point_index"])
        by_point[point_id]["pitch"].append(float(r["pitch"]))
        by_point[point_id]["yaw"].append(float(r["yaw"]))

    return {
        point_id: (statistics.median(vals["pitch"]), statistics.median(vals["yaw"]))
        for point_id, vals in by_point.items()
    }


def validate(rows):
    reprojection_errors = [float(r["pose_reprojection_error"]) for r in rows]
    reprojection_cutoff = percentile(reprojection_errors, REPROJECTION_ERROR_PERCENTILE)
    baselines = compute_point_baselines(rows)

    kept, dropped_blink, dropped_obstructed, dropped_drift = [], 0, 0, 0
    for row in rows:
        ear_avg = (float(row["left_ear"]) + float(row["right_ear"])) / 2
        reprojection_error = float(row["pose_reprojection_error"])
        base_pitch, base_yaw = baselines[int(row["point_index"])]
        drift = math.hypot(float(row["pitch"]) - base_pitch, float(row["yaw"]) - base_yaw)

        if ear_avg < EAR_BLINK_THRESHOLD:
            dropped_blink += 1
            continue
        if reprojection_error > reprojection_cutoff:
            dropped_obstructed += 1
            continue
        if drift > DRIFT_THRESHOLD_DEG:
            dropped_drift += 1
            continue

        kept.append(row)

    return kept, dropped_blink, dropped_obstructed, dropped_drift, reprojection_cutoff


def main():
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_INPUT)
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(DEFAULT_OUTPUT)

    rows, fieldnames = load_rows(input_path)
    kept, dropped_blink, dropped_obstructed, dropped_drift, cutoff = validate(rows)

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
    print(
        f"Dropped {dropped_drift} for head drift "
        f"(pitch/yaw > {DRIFT_THRESHOLD_DEG}deg from that point's median)"
    )
    print(f"Kept {len(kept)} frames ({len(kept) / total:.1%}) -> {output_path}")


if __name__ == "__main__":
    main()
