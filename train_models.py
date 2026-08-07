"""
Trains and compares two gaze-regression models on the cleaned calibration
data (Week 4):
  - polynomial_regression: degree-2 polynomial features + linear regression,
    the classic technique for webcam gaze mapping.
  - random_forest: handles nonlinearity/feature interactions without manual
    feature engineering.

Evaluated with k-fold cross-validation, split by point_index rather than
individual frame: the 100 frames captured per calibration point are
near-duplicates (same head position, small noise), so a frame-level shuffle
would leak near-identical samples into both train and test. Cross-validation
also means every point gets tested on exactly once across the k folds,
rather than a single random split determining the whole comparison (a
single split can flip which model "wins" just from which points happened
to land in the test set).

Scored with mean Euclidean pixel distance between predicted and true target
-- the same metric the Week 5 debug overlay will use, so this number means
something for actual cursor accuracy, not just an abstract loss.

Final models are retrained on the full dataset (all points) and saved to
models/*.pkl -- cross-validation is for picking/trusting a model, the saved
model itself should use every point available.

Usage: python train_models.py [clean_csv]
Defaults to tests/gather_data/calibration_points_clean.csv.
"""

import csv
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

FEATURE_COLUMNS = [
    "pitch", "yaw", "roll",
    "left_iris_x", "left_iris_y", "right_iris_x", "right_iris_y",
]
TARGET_COLUMNS = ["target_x", "target_y"]

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "tests" / "gather_data" / "calibration_points_clean.csv"
MODELS_DIR = PROJECT_ROOT / "models"
K_FOLDS = 5
RANDOM_SEED = 42


def build_model_factories():
    """Factories, not instances, so each fold (and the final fit) gets a fresh model."""
    return {
        "polynomial_regression": lambda: make_pipeline(
            PolynomialFeatures(degree=2), LinearRegression()
        ),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1
        ),
    }


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def k_fold_point_splits(rows, k, seed):
    """Yields (train_rows, test_rows) for each fold, split by point_index so
    every point is tested on exactly once across all folds."""
    point_ids = sorted({int(row["point_index"]) for row in rows})
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(point_ids)
    folds = np.array_split(shuffled, k)

    for i in range(k):
        test_points = set(folds[i].tolist())
        train_rows = [r for r in rows if int(r["point_index"]) not in test_points]
        test_rows = [r for r in rows if int(r["point_index"]) in test_points]
        yield train_rows, test_rows


def to_arrays(rows):
    X = np.array([[float(r[c]) for c in FEATURE_COLUMNS] for r in rows])
    y = np.array([[float(r[c]) for c in TARGET_COLUMNS] for r in rows])
    return X, y


def pixel_error_stats(y_true, y_pred):
    distances = np.linalg.norm(y_true - y_pred, axis=1)
    return {
        "mean": distances.mean(),
        "median": np.median(distances),
        "p90": np.percentile(distances, 90),
        "max": distances.max(),
    }


def cross_validate(rows, model_factories, k, seed):
    results = {name: [] for name in model_factories}

    for fold_idx, (train_rows, test_rows) in enumerate(k_fold_point_splits(rows, k, seed), start=1):
        X_train, y_train = to_arrays(train_rows)
        X_test, y_test = to_arrays(test_rows)

        for name, factory in model_factories.items():
            model = factory()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            stats = pixel_error_stats(y_test, y_pred)
            results[name].append(stats)
            print(f"  fold {fold_idx}/{k}  {name:<22} mean={stats['mean']:>7.1f}px")

    return results


def print_cv_summary(results):
    header = f"{'model':<22}{'cv mean px':>12}{'cv std':>10}"
    print(f"\n{header}")
    for name, fold_stats in results.items():
        fold_means = np.array([s["mean"] for s in fold_stats])
        print(f"{name:<22}{fold_means.mean():>12.1f}{fold_means.std():>10.1f}")


def main():
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    rows = load_rows(input_path)
    n_points = len({int(r["point_index"]) for r in rows})
    print(f"Loaded {len(rows)} frames from {n_points} points ({K_FOLDS}-fold cross-validation)\n")

    model_factories = build_model_factories()
    results = cross_validate(rows, model_factories, K_FOLDS, RANDOM_SEED)
    print_cv_summary(results)

    # Final models: retrained on every point, since CV was only for scoring.
    MODELS_DIR.mkdir(exist_ok=True)
    X_all, y_all = to_arrays(rows)
    for name, factory in model_factories.items():
        model = factory()
        model.fit(X_all, y_all)
        joblib.dump(model, MODELS_DIR / f"{name}.pkl")

    print(f"\nSaved final models (trained on all {n_points} points) to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
