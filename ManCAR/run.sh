
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.6"}
MASTER_PORT=${MASTER_PORT:-12346}


ori_weight_vals=(1.0)
pl_weight_vals=(0.5)     
trigger_num_train_vals=(4)
item_per_trigger_train_vals=(7) 
step_vals=(5)
COMMON_ARGS="--model_name ManCAR \
             --batch_size 512 --eval_batch_size 1024 \
             --early_stop 5   --num_workers 8 --verbose 0 --dataset CDs_and_Vinyl"

LOG_ROOT="logs/CDs_and_Vinyl"
mkdir -p "${LOG_ROOT}"

    
for pl in "${pl_weight_vals[@]}"; do

  SUBDIR="${LOG_ROOT}/pl_${pl}"
  mkdir -p "${SUBDIR}"
      for trigger in "${trigger_num_train_vals[@]}"; do
        for item in "${item_per_trigger_train_vals[@]}"; do
              for step in "${step_vals[@]}"; do
                LOG_FILE="${SUBDIR}_tri_${trigger}-item_${item}_step_${step}.log"
                printf ">>> ori=%s -> %s\n" \
                      "$ori" "$LOG_FILE"

                # ---------------------------- CORE PATCH ----------------------------
                if [[ "$pl" != "0.0" && "$pl" != "0" ]]; then
                  # force re‑run: optionally backup the old file
                  [[ -f "$LOG_FILE" ]] && mv "$LOG_FILE" "${LOG_FILE}.old$(date +%s)"
                else
                  # pl==0 : keep previous result if log file is non‑empty
                  [[ -s "$LOG_FILE" ]] && { echo "    – already finished, skipping."; continue; }
                fi
                # -------------------------------------------------------------------

                torchrun \
                  --nproc_per_node 1 \
                  --nnodes  "${WORLD_SIZE}" \
                  --node_rank "${RANK}" \
                  --master_addr "${MASTER_ADDR}" \
                  --master_port "${MASTER_PORT}" \
                  main.py \
                  ${COMMON_ARGS} \
                  --gpu 5 \
                  --log_file     "${LOG_FILE}" \
                  --trigger_num_train "${trigger}" \
                  --item_per_trigger_train "${item}" \
                  --reason_step "${step}"
        done
      done
    done
  done
