#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="$SCRIPT_DIR/test_zipserv_fa2"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

if [[ ! -x "$BIN" ]]; then
  echo "Binary not found or not executable: $BIN"
  echo "Please build first: cd $SCRIPT_DIR && make -j"
  exit 1
fi

COMBOS=(
  "64 64"
  "64 128"
  "64 256"
  "64 512"
  "128 128"
  "128 256"
  "128 512"
  "128 1024"
  "256 256"
  "256 512"
  "256 1024"
  "256 2048"
  "512 512"
  "512 1024"
  "512 2048"
  "512 4096"
  "1024 1024"
  "1024 2048"
  "1024 4096"
  "1024 8192"
  "2048 2048"
  "2048 4096"
  "2048 8192"
  "2048 16384"
  "4096 4096"
  "4096 8192"
  "4096 16384"
  "4096 32768"
  "8192 8192"
  "8192 16384"
  "8192 32768"
  "8192 65536"
)

failed=0
mode=0
for combo in "${COMBOS[@]}"; do
  d_model="${combo%% *}"
  seq_len="${combo##* }"
  log_file="$LOG_DIR/test_zipserv_fa2_d${d_model}_s${seq_len}.log"

  echo "============================================================"
  echo "Running isolated case: d_model=$d_model seq_len=$seq_len"
  echo "Log: $log_file"

  (
    cd "$SCRIPT_DIR" || exit 1
    "$BIN" "$d_model" "$seq_len" "$mode"
  ) 2>&1 | tee "$log_file"

  status=${PIPESTATUS[0]}
  if [[ $status -ne 0 ]]; then
    echo "Case failed: d_model=$d_model seq_len=$seq_len (exit=$status)"
    failed=$((failed + 1))
  fi
done

echo "============================================================"
if [[ $failed -eq 0 ]]; then
  echo "All isolated cases finished successfully."
  exit 0
else
  echo "$failed case(s) failed. Check logs under: $LOG_DIR"
  exit 1
fi
