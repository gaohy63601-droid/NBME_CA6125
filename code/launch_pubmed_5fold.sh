#!/bin/bash
# Train PubMedBERT-large 5-fold on NBME train_split.
# Output: ckpt/pubmed_fold{0..4}.pt  (KEEP existing xl_fold*.pt untouched)
set -e
cd /raid/yiren/ghy/motion_transfer/medical/nbme_baseline

BACKBONE=ckpt/pubmed_backbone
TAG=pubmed
EPOCHS="${1:-5}"
AWP_START="${2:-2}"

GPUS=(0 1 3 4 7)   # 5 idle GPUs (avoid 6 which has root daemon, avoid 2/5 which had issues earlier)
LOGDIR=logs/${TAG}
mkdir -p $LOGDIR

for f in 0 1 2 3 4; do
    GPU=${GPUS[$f]}
    SESS="nbme_${TAG}_f${f}"
    tmux kill-session -t $SESS 2>/dev/null || true
    tmux new-session -d -s $SESS -c /raid/yiren/ghy/motion_transfer/medical/nbme_baseline
    CMD="CUDA_VISIBLE_DEVICES=$GPU HF_HOME=/raid/yiren/hf_cache/huggingface /raid/yiren/conda_envs/ragnarok/bin/python -u train.py --epochs $EPOCHS --awp_start_ep $AWP_START --fold $f --model_name microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract --backbone $BACKBONE --ckpt_name ${TAG}_fold${f}.pt 2>&1 | tee $LOGDIR/fold${f}.log"
    tmux send-keys -t $SESS "$CMD" C-m
    echo "launched $SESS on GPU $GPU"
done
echo "all 5 folds launched. monitor: tmux ls | grep nbme_${TAG}"
