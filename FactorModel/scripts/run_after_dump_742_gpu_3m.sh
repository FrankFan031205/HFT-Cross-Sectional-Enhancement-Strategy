#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/fwz/projects/HFT_010-dev_fwz"
FACTOR_DIR="${PROJECT_ROOT}/FactorModel"

TAG="20241022_20250114_742"

INPUT="${FACTOR_DIR}/data/raw/factor_features_${TAG}.csv"
FEATURE_YAML="${FACTOR_DIR}/data/raw/feature_cols_20241022_20241122_100.yaml"
DUMP_LOG="${FACTOR_DIR}/logs/dump_factor_features_${TAG}.log"

ML_SCRIPT="${FACTOR_DIR}/src/train_cs_ml_factors_20.py"
DL1_SCRIPT="${FACTOR_DIR}/src/train_cs_dl_factors_20.py"
DL2_SCRIPT="${FACTOR_DIR}/src/train_cs_dl_factors_20_v2.py"

ML_LOG="${FACTOR_DIR}/logs/train_cs_ml_factors_20_${TAG}_gpu.log"
DL1_LOG="${FACTOR_DIR}/logs/train_cs_dl_factors_20_${TAG}_gpu6.log"
DL2_LOG="${FACTOR_DIR}/logs/train_cs_dl_factors_20_v2_${TAG}_gpu7.log"

ML_EVAL="${FACTOR_DIR}/outputs/eval_cs_ml_factors_20_${TAG}.csv"
DL1_EVAL="${FACTOR_DIR}/outputs/eval_cs_dl_factors_20_${TAG}.csv"
DL2_EVAL="${FACTOR_DIR}/outputs/eval_cs_dl_v2_factors_20_${TAG}.csv"

ML_GPU="${ML_GPU:-4}"
DL1_GPU="${DL1_GPU:-6}"
DL2_GPU="${DL2_GPU:-7}"

MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-3000000}"
MIN_ROWS="${MIN_ROWS:-25000000}"

TRAIN_START=20241022
TRAIN_END=20241220
VALID_START=20241223
VALID_END=20241231
TEST_START=20250102
TEST_END=20250114

mkdir -p "${FACTOR_DIR}/logs" "${FACTOR_DIR}/outputs" "${FACTOR_DIR}/models"
cd "${FACTOR_DIR}"

echo "================================================================================"
echo "3M 742 ML + DL GPU pipeline"
echo "time: $(date)"
echo "TAG=${TAG}"
echo "INPUT=${INPUT}"
echo "FEATURE_YAML=${FEATURE_YAML}"
echo "ML_GPU=${ML_GPU}"
echo "DL1_GPU=${DL1_GPU}"
echo "DL2_GPU=${DL2_GPU}"
echo "MAX_TRAIN_ROWS=${MAX_TRAIN_ROWS}"
echo "TRAIN=${TRAIN_START}-${TRAIN_END}"
echo "VALID=${VALID_START}-${VALID_END}"
echo "TEST=${TEST_START}-${TEST_END}"
echo "================================================================================"

nvidia-smi || true

wait_for_dump() {
    echo
    echo "================================================================================"
    echo "Waiting for 3M 742 factor dump"
    echo "dump log: ${DUMP_LOG}"
    echo "================================================================================"

    while true; do
        if [[ -f "${DUMP_LOG}" ]] && grep -q "^DONE" "${DUMP_LOG}"; then
            echo "$(date) dump log shows DONE"
            break
        fi

        if pgrep -f "dump_factor_features_fixed_universe.py" >/dev/null 2>&1; then
            echo "$(date) dump still running..."
            if [[ -f "${DUMP_LOG}" ]]; then
                tail -8 "${DUMP_LOG}" || true
            fi
            sleep 300
        else
            echo "$(date) no dump process found"

            if [[ -f "${INPUT}" ]]; then
                echo "input file exists, will validate it"
                break
            fi

            echo "ERROR: dump stopped and input file does not exist"
            if [[ -f "${DUMP_LOG}" ]]; then
                tail -80 "${DUMP_LOG}" || true
            fi
            exit 1
        fi
    done
}

validate_input() {
    echo
    echo "================================================================================"
    echo "Validating input CSV"
    echo "================================================================================"

    if [[ ! -s "${INPUT}" ]]; then
        echo "ERROR: missing input: ${INPUT}"
        exit 1
    fi

    if [[ ! -s "${FEATURE_YAML}" ]]; then
        echo "ERROR: missing feature yaml: ${FEATURE_YAML}"
        exit 1
    fi

    python - <<PY
import pandas as pd

path = "${INPUT}"
min_rows = int("${MIN_ROWS}")

head = pd.read_csv(path, nrows=5)
cols = head.columns.tolist()

fwz_cols = [c for c in cols if c.startswith("fwz")]
label_cols = [c for c in cols if c.startswith("label_")]

print("num columns:", len(cols))
print("num fwz cols:", len(fwz_cols))
print("label cols:", label_cols)
print(head.head())

need = {"label_30", "label_60", "label_90", "label_120"}

if len(fwz_cols) == 0:
    raise SystemExit("ERROR: no fwz columns")

if not need.issubset(set(cols)):
    raise SystemExit(f"ERROR: missing label columns: {need - set(cols)}")

rows = 0
date_min = None
date_max = None
stocks = set()

for chunk in pd.read_csv(path, usecols=["date", "securityid"], dtype={"securityid": str}, chunksize=1000000):
    rows += len(chunk)
    dmin = int(chunk["date"].min())
    dmax = int(chunk["date"].max())
    date_min = dmin if date_min is None else min(date_min, dmin)
    date_max = dmax if date_max is None else max(date_max, dmax)
    stocks.update(chunk["securityid"].astype(str).str.zfill(6).unique())

print("rows:", rows)
print("date range:", date_min, date_max)
print("unique stocks:", len(stocks))

if rows < min_rows:
    raise SystemExit(f"ERROR: rows {rows} < MIN_ROWS {min_rows}")
PY

    echo "input validation passed"
}

run_task() {
    local name="$1"
    local eval_file="$2"
    local log_file="$3"
    shift 3
    local cmd=("$@")

    echo
    echo "================================================================================"
    echo "Launching ${name}"
    echo "eval: ${eval_file}"
    echo "log : ${log_file}"
    echo "time: $(date)"
    echo "================================================================================"

    if [[ -s "${eval_file}" && "${FORCE:-0}" != "1" ]]; then
        echo "skip ${name}: eval file already exists. Set FORCE=1 to rerun."
        return 0
    fi

    printf '%q ' "${cmd[@]}"
    echo

    "${cmd[@]}" > "${log_file}" 2>&1 &
    local pid=$!
    echo "${name} pid=${pid}"
    echo "${pid}" > "${log_file}.pid"
}

summarize_eval() {
    local eval_file="$1"
    local title="$2"

    if [[ ! -s "${eval_file}" ]]; then
        echo "missing eval for ${title}: ${eval_file}"
        return 0
    fi

    echo
    echo "================================================================================"
    echo "${title}"
    echo "${eval_file}"
    echo "================================================================================"

    python - <<PY
import pandas as pd

path = "${eval_file}"
df = pd.read_csv(path)

test = df[df["split"] == "test"].copy()
test = test.sort_values(
    ["mean_cs_rankic_raw_label", "mean_long_short_raw_label"],
    ascending=False,
)

cols = [
    "factor", "model", "horizon", "rows", "num_datetimes",
    "mean_cs_rankic_raw_label", "cs_rankicir_raw_label",
    "mean_long_short_raw_label", "long_short_ir_raw_label",
    "overall_rankic_raw",
]
cols = [c for c in cols if c in test.columns]

print(test[cols].head(30).to_string(index=False))
PY
}

wait_for_dump
validate_input

for f in "${ML_SCRIPT}" "${DL1_SCRIPT}" "${DL2_SCRIPT}"; do
    if [[ ! -s "$f" ]]; then
        echo "ERROR: missing script: $f"
        exit 1
    fi
done

echo
echo "================================================================================"
echo "Starting training jobs"
echo "================================================================================"

run_task \
  "20 ML factors 3M" \
  "${ML_EVAL}" \
  "${ML_LOG}" \
  env CUDA_VISIBLE_DEVICES="${ML_GPU}" python -u "${ML_SCRIPT}" \
    --input "data/raw/factor_features_${TAG}.csv" \
    --feature_cols "data/raw/feature_cols_20241022_20241122_100.yaml" \
    --tag "${TAG}" \
    --max_train_rows "${MAX_TRAIN_ROWS}" \
    --train_start "${TRAIN_START}" \
    --train_end "${TRAIN_END}" \
    --valid_start "${VALID_START}" \
    --valid_end "${VALID_END}" \
    --test_start "${TEST_START}" \
    --test_end "${TEST_END}"

run_task \
  "20 DL batch1 factors 3M" \
  "${DL1_EVAL}" \
  "${DL1_LOG}" \
  env CUDA_VISIBLE_DEVICES="${DL1_GPU}" python -u "${DL1_SCRIPT}" \
    --input "data/raw/factor_features_${TAG}.csv" \
    --feature_cols "data/raw/feature_cols_20241022_20241122_100.yaml" \
    --tag "${TAG}" \
    --max_train_rows "${MAX_TRAIN_ROWS}" \
    --train_start "${TRAIN_START}" \
    --train_end "${TRAIN_END}" \
    --valid_start "${VALID_START}" \
    --valid_end "${VALID_END}" \
    --test_start "${TEST_START}" \
    --test_end "${TEST_END}"

run_task \
  "20 DL batch2 factors 3M" \
  "${DL2_EVAL}" \
  "${DL2_LOG}" \
  env CUDA_VISIBLE_DEVICES="${DL2_GPU}" python -u "${DL2_SCRIPT}" \
    --input "data/raw/factor_features_${TAG}.csv" \
    --feature_cols "data/raw/feature_cols_20241022_20241122_100.yaml" \
    --tag "${TAG}" \
    --max_train_rows "${MAX_TRAIN_ROWS}" \
    --train_start "${TRAIN_START}" \
    --train_end "${TRAIN_END}" \
    --valid_start "${VALID_START}" \
    --valid_end "${VALID_END}" \
    --test_start "${TEST_START}" \
    --test_end "${TEST_END}"

echo
echo "================================================================================"
echo "Waiting for training jobs"
echo "================================================================================"

FAILED=0

for pidfile in "${ML_LOG}.pid" "${DL1_LOG}.pid" "${DL2_LOG}.pid"; do
    if [[ ! -f "${pidfile}" ]]; then
        continue
    fi

    pid=$(cat "${pidfile}")
    echo "waiting pid=${pid}"

    if wait "${pid}"; then
        echo "pid=${pid} finished successfully"
    else
        echo "pid=${pid} failed"
        FAILED=1
    fi
done

echo
echo "================================================================================"
echo "Training finished at $(date)"
echo "FAILED=${FAILED}"
echo "================================================================================"

nvidia-smi || true

summarize_eval "${ML_EVAL}" "3M ML TEST RANKING"
summarize_eval "${DL1_EVAL}" "3M DL BATCH1 TEST RANKING"
summarize_eval "${DL2_EVAL}" "3M DL BATCH2 TEST RANKING"

echo
echo "================================================================================"
echo "Generated files"
echo "================================================================================"

ls -lh \
  "${FACTOR_DIR}/outputs/ml_cs_hidden_factors_20_${TAG}.csv" \
  "${FACTOR_DIR}/outputs/eval_cs_ml_factors_20_${TAG}.csv" \
  "${FACTOR_DIR}/outputs/ml_cs_dl_hidden_factors_20_${TAG}.csv" \
  "${FACTOR_DIR}/outputs/eval_cs_dl_factors_20_${TAG}.csv" \
  "${FACTOR_DIR}/outputs/ml_cs_dl_v2_hidden_factors_20_${TAG}.csv" \
  "${FACTOR_DIR}/outputs/eval_cs_dl_v2_factors_20_${TAG}.csv" 2>/dev/null || true

if [[ "${FAILED}" != "0" ]]; then
    echo "Some jobs failed. Check logs:"
    echo "${ML_LOG}"
    echo "${DL1_LOG}"
    echo "${DL2_LOG}"
    exit 1
fi

echo "ALL DONE."
