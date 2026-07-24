#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/fwz/projects/HFT_010-dev_fwz"
FACTOR_DIR="${PROJECT_ROOT}/FactorModel"
TAG="20241022_20241122_742"

INPUT="${FACTOR_DIR}/data/raw/factor_features_${TAG}.csv"
FEATURE_YAML="${FACTOR_DIR}/data/raw/feature_cols_20241022_20241122_100.yaml"
DUMP_LOG="${FACTOR_DIR}/logs/dump_factor_features_${TAG}.log"

ML_SCRIPT="${FACTOR_DIR}/src/train_cs_ml_factors_20.py"
DL_SCRIPT="${FACTOR_DIR}/src/train_cs_dl_factors_20.py"
DL_V2_SCRIPT="${FACTOR_DIR}/src/train_cs_dl_factors_20_v2.py"

ML_LOG="${FACTOR_DIR}/logs/train_cs_ml_factors_20_${TAG}.log"
DL_LOG="${FACTOR_DIR}/logs/train_cs_dl_factors_20_${TAG}.log"
DL_V2_LOG="${FACTOR_DIR}/logs/train_cs_dl_factors_20_v2_${TAG}.log"

ML_EVAL="${FACTOR_DIR}/outputs/eval_cs_ml_factors_20_${TAG}.csv"
DL_EVAL="${FACTOR_DIR}/outputs/eval_cs_dl_factors_20_${TAG}.csv"
DL_V2_EVAL="${FACTOR_DIR}/outputs/eval_cs_dl_v2_factors_20_${TAG}.csv"

MIN_ROWS="${MIN_ROWS:-10000000}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-2000000}"

mkdir -p "${FACTOR_DIR}/logs" "${FACTOR_DIR}/outputs" "${FACTOR_DIR}/models"
cd "${FACTOR_DIR}"

echo "================================================================================"
echo "Run 742-universe factor pipeline"
echo "time: $(date)"
echo "TAG=${TAG}"
echo "INPUT=${INPUT}"
echo "MIN_ROWS=${MIN_ROWS}"
echo "MAX_TRAIN_ROWS=${MAX_TRAIN_ROWS}"
echo "================================================================================"

wait_for_dump() {
  echo "Waiting for dump to finish..."
  while true; do
    if [[ -f "${DUMP_LOG}" ]] && grep -q "^DONE" "${DUMP_LOG}"; then
      echo "Dump log shows DONE."
      break
    fi

    if pgrep -f "dump_factor_features_742_from_universe.py" >/dev/null 2>&1; then
      echo "$(date) dump still running..."
      if [[ -f "${DUMP_LOG}" ]]; then tail -8 "${DUMP_LOG}" || true; fi
      sleep 300
    else
      echo "No dump process found."
      if [[ -f "${INPUT}" ]]; then
        echo "Input file exists, will validate it."
        break
      fi
      echo "ERROR: dump is not running and input file does not exist."
      if [[ -f "${DUMP_LOG}" ]]; then tail -80 "${DUMP_LOG}" || true; fi
      exit 1
    fi
  done
}

validate_input() {
  echo "Validating input..."
  if [[ ! -s "${INPUT}" ]]; then
    echo "ERROR: input file missing or empty: ${INPUT}"
    exit 1
  fi
  if [[ ! -s "${FEATURE_YAML}" ]]; then
    echo "ERROR: feature yaml missing or empty: ${FEATURE_YAML}"
    exit 1
  fi

  python - <<PY
import pandas as pd

path = "${INPUT}"
min_rows = int("${MIN_ROWS}")

head = pd.read_csv(path, nrows=5)
cols = head.columns.tolist()
fwz_cols = [c for c in cols if c.startswith("fwz")]
need_labels = {"label_30", "label_60", "label_90", "label_120"}

print("num columns:", len(cols))
print("num fwz cols:", len(fwz_cols))
print("label cols:", [c for c in cols if c.startswith("label_")])
print(head.head())

if len(fwz_cols) == 0:
    raise SystemExit("ERROR: no fwz columns")
if not need_labels.issubset(set(cols)):
    raise SystemExit(f"ERROR: missing labels {need_labels - set(cols)}")

n = 0
date_min = None
date_max = None
stocks = set()

for chunk in pd.read_csv(path, usecols=["date", "securityid"], dtype={"securityid": str}, chunksize=1000000):
    n += len(chunk)
    dmin = int(chunk["date"].min())
    dmax = int(chunk["date"].max())
    date_min = dmin if date_min is None else min(date_min, dmin)
    date_max = dmax if date_max is None else max(date_max, dmax)
    stocks.update(chunk["securityid"].astype(str).str.zfill(6).unique().tolist())

print("rows:", n)
print("date range:", date_min, date_max)
print("unique stocks:", len(stocks))

if n < min_rows:
    raise SystemExit(f"ERROR: rows {n} < MIN_ROWS {min_rows}")
PY
  echo "Input validation passed."
}

run_stage() {
  local name="$1"
  local eval_file="$2"
  local log_file="$3"
  shift 3
  local cmd=("$@")

  echo
  echo "================================================================================"
  echo "Stage: ${name}"
  echo "time: $(date)"
  echo "eval_file=${eval_file}"
  echo "log_file=${log_file}"
  echo "================================================================================"

  if [[ -s "${eval_file}" && "${FORCE:-0}" != "1" ]]; then
    echo "Skip ${name}: eval file already exists. Set FORCE=1 to rerun."
    return 0
  fi

  printf "Command: "
  printf "%q " "${cmd[@]}"
  echo

  "${cmd[@]}" > "${log_file}" 2>&1

  echo "${name} finished at $(date)"
  tail -60 "${log_file}" || true

  if [[ ! -s "${eval_file}" ]]; then
    echo "ERROR: eval file not found after stage: ${eval_file}"
    exit 1
  fi
}

summarize_eval() {
  local eval_file="$1"
  local title="$2"

  echo
  echo "================================================================================"
  echo "${title}"
  echo "${eval_file}"
  echo "================================================================================"

  python - <<PY
import pandas as pd
path = "${eval_file}"
df = pd.read_csv(path)
if "split" not in df.columns:
    print(df.head(20).to_string(index=False))
    raise SystemExit

test = df[df["split"] == "test"].copy()
test = test.sort_values(["mean_cs_rankic_raw_label", "mean_long_short_raw_label"], ascending=False)
cols = [
    "factor", "model", "horizon", "rows", "num_datetimes",
    "mean_cs_rankic_raw_label", "cs_rankicir_raw_label",
    "mean_long_short_raw_label", "long_short_ir_raw_label",
    "overall_rankic_raw"
]
cols = [c for c in cols if c in test.columns]
print(test[cols].head(30).to_string(index=False))
PY
}

wait_for_dump
validate_input

for f in "${ML_SCRIPT}" "${DL_SCRIPT}" "${DL_V2_SCRIPT}"; do
  if [[ ! -s "$f" ]]; then
    echo "ERROR: missing script $f"
    exit 1
  fi
done

run_stage "20 non-DL ML factors" "${ML_EVAL}" "${ML_LOG}"   python -u "${ML_SCRIPT}"     --input "data/raw/factor_features_${TAG}.csv"     --feature_cols "data/raw/feature_cols_20241022_20241122_100.yaml"     --tag "${TAG}"     --max_train_rows "${MAX_TRAIN_ROWS}"

summarize_eval "${ML_EVAL}" "ML TEST RANKING"

run_stage "20 DL factors batch 1" "${DL_EVAL}" "${DL_LOG}"   python -u "${DL_SCRIPT}"     --input "data/raw/factor_features_${TAG}.csv"     --feature_cols "data/raw/feature_cols_20241022_20241122_100.yaml"     --tag "${TAG}"     --max_train_rows "${MAX_TRAIN_ROWS}"

summarize_eval "${DL_EVAL}" "DL BATCH 1 TEST RANKING"

run_stage "20 DL factors batch 2" "${DL_V2_EVAL}" "${DL_V2_LOG}"   python -u "${DL_V2_SCRIPT}"     --input "data/raw/factor_features_${TAG}.csv"     --feature_cols "data/raw/feature_cols_20241022_20241122_100.yaml"     --tag "${TAG}"     --max_train_rows "${MAX_TRAIN_ROWS}"

summarize_eval "${DL_V2_EVAL}" "DL BATCH 2 TEST RANKING"

echo
echo "================================================================================"
echo "ALL DONE at $(date)"
echo "Generated files:"
ls -lh   "${FACTOR_DIR}/outputs/ml_cs_hidden_factors_20_${TAG}.csv"   "${FACTOR_DIR}/outputs/eval_cs_ml_factors_20_${TAG}.csv"   "${FACTOR_DIR}/outputs/ml_cs_dl_hidden_factors_20_${TAG}.csv"   "${FACTOR_DIR}/outputs/eval_cs_dl_factors_20_${TAG}.csv"   "${FACTOR_DIR}/outputs/ml_cs_dl_v2_hidden_factors_20_${TAG}.csv"   "${FACTOR_DIR}/outputs/eval_cs_dl_v2_factors_20_${TAG}.csv" || true
echo "================================================================================"
