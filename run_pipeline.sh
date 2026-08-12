#!/usr/bin/env bash
# Runs calibration -> validation -> training end to end, then optionally
# launches the live test with the model you pick.
#
# Usage: ./run_pipeline.sh [random_forest|polynomial_regression]
#   No argument: calibrate, validate, train, then stop.
#   With a model name: also launch live_gaze_test.py with that model.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/mp_env/bin/python"
GATHER_DIR="$SCRIPT_DIR/tests/gather_data"
GAZE_TEST_DIR="$SCRIPT_DIR/tests/gaze_test_env"
MODELS_DIR="$SCRIPT_DIR/models"

usage() {
    echo "Usage: $0 [random_forest|polynomial_regression]" >&2
    exit 1
}

MODEL=""
if [ "$#" -gt 1 ]; then
    usage
elif [ "$#" -eq 1 ]; then
    MODEL="$1"
    case "$MODEL" in
        random_forest|polynomial_regression) ;;
        *) usage ;;
    esac
fi

# calibration_dots.py and validate_data.py write/read their CSVs relative
# to the current directory, not the script's location -- run them from
# tests/gather_data so they always hit the right files.
echo "== Step 1/3: calibration =="
(cd "$GATHER_DIR" && "$PYTHON" calibration_dots.py)

echo "== Step 2/3: validation =="
(cd "$GATHER_DIR" && "$PYTHON" validate_data.py)

echo "== Step 3/3: training =="
"$PYTHON" "$SCRIPT_DIR/train_models.py"

if [ -n "$MODEL" ]; then
    echo "== Live test: $MODEL =="
    "$PYTHON" "$GAZE_TEST_DIR/live_gaze_test.py" "$MODELS_DIR/$MODEL.pkl"
else
    echo
    echo "Pipeline done. Re-run with 'random_forest' or 'polynomial_regression'"
    echo "as an argument to also launch the live test with that model."
fi
