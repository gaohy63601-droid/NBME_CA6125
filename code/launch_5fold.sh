#!/bin/bash
# Launch 5-fold finetune in parallel on GPUs 1,2,4,5,7
# Usage: ./launch_5fold.sh <round_tag> [extra_csv]
#   round_tag: e.g. "r1" (no pseudo) or "r2" (with pseudo)
#   extra_csv: optional path to pseudo_train.csv
set -e
cd /raid/yiren/ghy/motion_transfer/medical/nbme_baseline
TAG="${1:?need round tag, e.g. r1}"
EXTRA="$2"
EPOCHS="${3:-5}"
AWP_START="${4:-2}"
BACKBONE=ckpt/mlm_backbone

GPUS=(1 2 4 5 7)
LOGDIR=logs/${TAG}
mkdir -p $LOGDIR

EXTRA_FLAG=""
if [[ -n "$EXTRA" ]]; then
    EXTRA_FLAG="--extra_csv $EXTRA"
fi

for f in 0 1 2 3 4; do
    GPU=${GPUS[$f]}
    SESS="nbme_${TAG}_f${f}"
    tmux kill-session -t $SESS 2>/dev/null || true
    tmux new-session -d -s $SESS -c /raid/yiren/ghy/motion_transfer/medical/nbme_baseline
    CMD="CUDA_VISIBLE_DEVICES=$GPU HF_HOME=/raid/yiren/hf_cache/huggingface /raid/yiren/conda_envs/ragnarok/bin/python -u train.py --epochs $EPOCHS --awp_start_ep $AWP_START --fold $f --backbone $BACKBONE --ckpt_name ${TAG}_fold${f}.pt $EXTRA_FLAG 2>&1 | tee $LOGDIR/fold${f}.log"
    tmux send-keys -t $SESS "$CMD" C-m
    echo "launched $SESS on GPU $GPU"
done
echo "all 5 folds launched. monitor: tmux ls | grep nbme_${TAG}"
