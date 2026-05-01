#!/bin/bash
# Train PubMedBERT-large 5-fold on NBME train_split.
# Output: ckpt/pubmed_fold{0..4}.pt  (KEEP existing xl_fold*.pt untouched)
set -e
# Edit WORKDIR / PYTHON to match your environment.
WORKDIR="${WORKDIR:-$(pwd)}"
PYTHON="${PYTHON:-python}"
cd "$WORKDIR"

BACKBONE=ckpt/pubmed_backbone
TAG=pubmed
EPOCHS="${1:-5}"
AWP_START="${2:-2}"

GPUS=(0 1 3 4 7)   # any 5 idle GPUs
LOGDIR=logs/${TAG}
mkdir -p $LOGDIR

for f in 0 1 2 3 4; do
    GPU=${GPUS[$f]}
    SESS="nbme_${TAG}_f${f}"
    tmux kill-session -t $SESS 2>/dev/null || true
    tmux new-session -d -s $SESS -c "$WORKDIR"
    CMD="CUDA_VISIBLE_DEVICES=$GPU $PYTHON -u train_encoder.py --epochs $EPOCHS --awp_start_ep $AWP_START --fold $f --model_name microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract --backbone $BACKBONE --ckpt_name ${TAG}_fold${f}.pt 2>&1 | tee $LOGDIR/fold${f}.log"
    tmux send-keys -t $SESS "$CMD" C-m
    echo "launched $SESS on GPU $GPU"
done
echo "all 5 folds launched. monitor: tmux ls | grep nbme_${TAG}"
