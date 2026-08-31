#!/bin/bash
# llm_verify_runner.sh
# ====================
# Wrapper that runs the LLM verification script in a loop with auto-restart.
# If the node process dies (sandbox kill / OOM / error), this restarts it
# and the script resumes from where it left off (via llm_verified_pseudo.jsonl).
#
# Runs until the progress file shows all 1391 entries done, or max 200 iterations.

cd /home/z/my-project
LOG=finetuning/datasets/llm_verify_pseudo.log
PROG=finetuning/datasets/llm_verify_pseudo_progress.json
SCRIPT=finetuning/scripts/llm_verify_pseudo.mjs
TOTAL=1391

iteration=0
MAX_ITER=300

while [ $iteration -lt $MAX_ITER ]; do
  iteration=$((iteration + 1))

  # Check if already complete
  if [ -f "$PROG" ]; then
    DONE=$(python3 -c "import json; print(json.load(open('$PROG')).get('done',0))" 2>/dev/null || echo 0)
    echo "[$(date +%H:%M:%S)] iter=$iteration done=$DONE/$TOTAL" >> $LOG
    if [ "$DONE" -ge "$TOTAL" ]; then
      echo "[$(date +%H:%M:%S)] COMPLETE — all $TOTAL entries verified." >> $LOG
      break
    fi
  fi

  # Run node script (foreground within this wrapper; wrapper itself is backgrounded)
  echo "[$(date +%H:%M:%S)] starting node iteration $iteration (done so far: ${DONE:-0})" >> $LOG
  node "$SCRIPT" >> "$LOG" 2>&1
  EXIT_CODE=$?
  echo "[$(date +%H:%M:%S)] node exited with code $EXIT_CODE" >> $LOG

  # Brief pause before restart
  sleep 3
done

echo "[$(date +%H:%M:%S)] runner finished after $iteration iterations" >> $LOG
