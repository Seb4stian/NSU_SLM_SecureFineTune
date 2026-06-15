#!/bin/bash
# Master script: Train both models and generate responses
# Run inside screen so it persists if VS Code disconnects

set -e

export HF_HUB_CACHE="/mnt/cache"
export HF_DATASETS_CACHE="/mnt/cache"
export HF_HOME="/mnt/cache"

LOGFILE="/mnt/cache/training_pipeline.log"

echo "========================================" | tee -a "$LOGFILE"
echo "Pipeline started at $(date)" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"

# ── Step 1: Train onlyRejectingQuestions ──────────────────────────────────────
echo "" | tee -a "$LOGFILE"
echo "[STEP 1/4] Training model: onlyRejectingQuestions" | tee -a "$LOGFILE"
echo "Started at $(date)" | tee -a "$LOGFILE"

accelerate launch --num_processes=4 --multi_gpu /mnt/cache/train_onlyRejectingQuestions.py 2>&1 | tee -a "$LOGFILE"

echo "[STEP 1/4] COMPLETED at $(date)" | tee -a "$LOGFILE"

# ── Step 2: Train onlyGoodExamples ───────────────────────────────────────────
echo "" | tee -a "$LOGFILE"
echo "[STEP 2/4] Training model: onlyGoodExamples" | tee -a "$LOGFILE"
echo "Started at $(date)" | tee -a "$LOGFILE"

accelerate launch --num_processes=4 --multi_gpu /mnt/cache/train_onlyGoodExamples.py 2>&1 | tee -a "$LOGFILE"

echo "[STEP 2/4] COMPLETED at $(date)" | tee -a "$LOGFILE"

# ── Step 3: Generate responses for onlyRejectingQuestions ─────────────────────
echo "" | tee -a "$LOGFILE"
echo "[STEP 3/4] Generating responses: onlyRejectingQuestions" | tee -a "$LOGFILE"
echo "Started at $(date)" | tee -a "$LOGFILE"

python /mnt/cache/generate_responses_onlyRejectingQuestions.py 2>&1 | tee -a "$LOGFILE"

echo "[STEP 3/4] COMPLETED at $(date)" | tee -a "$LOGFILE"

# ── Step 4: Generate responses for onlyGoodExamples ──────────────────────────
echo "" | tee -a "$LOGFILE"
echo "[STEP 4/4] Generating responses: onlyGoodExamples" | tee -a "$LOGFILE"
echo "Started at $(date)" | tee -a "$LOGFILE"

python /mnt/cache/generate_responses_onlyGoodExamples.py 2>&1 | tee -a "$LOGFILE"

echo "[STEP 4/4] COMPLETED at $(date)" | tee -a "$LOGFILE"

# ── Done ─────────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"
echo "ALL STEPS COMPLETED at $(date)" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "Output files:" | tee -a "$LOGFILE"
echo "  - /home/azureuser/cloudfiles/code/Users/ing.eduardo.castro/ValidationDataset_MiniCPM-1B-sft-bf16-LoRA_onlyRejectingQuestions.jsonl" | tee -a "$LOGFILE"
echo "  - /home/azureuser/cloudfiles/code/Users/ing.eduardo.castro/ValidationDataset_MiniCPM-1B-sft-bf16-LoRA_onlyGoodExamples.jsonl" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "HuggingFace repos:" | tee -a "$LOGFILE"
echo "  - Edcastro/MiniCPM-1B-sft-bf16-edcastr_JavaScript-onlyRejectingQuestions" | tee -a "$LOGFILE"
echo "  - Edcastro/MiniCPM-1B-sft-bf16-edcastr_JavaScript-onlyGoodExamples" | tee -a "$LOGFILE"

# ── Step 5: Copy all scripts to ablatingAnalysis folder ──────────────────────
echo "" | tee -a "$LOGFILE"
echo "[STEP 5/5] Copying Python scripts to ablatingAnalysis folder" | tee -a "$LOGFILE"

mkdir -p /home/azureuser/cloudfiles/code/Users/ing.eduardo.castro/ablatingAnalysis

cp /mnt/cache/train_onlyRejectingQuestions.py /home/azureuser/cloudfiles/code/Users/ing.eduardo.castro/ablatingAnalysis/
cp /mnt/cache/train_onlyGoodExamples.py /home/azureuser/cloudfiles/code/Users/ing.eduardo.castro/ablatingAnalysis/
cp /mnt/cache/generate_responses_onlyRejectingQuestions.py /home/azureuser/cloudfiles/code/Users/ing.eduardo.castro/ablatingAnalysis/
cp /mnt/cache/generate_responses_onlyGoodExamples.py /home/azureuser/cloudfiles/code/Users/ing.eduardo.castro/ablatingAnalysis/
cp /mnt/cache/run_pipeline.sh /home/azureuser/cloudfiles/code/Users/ing.eduardo.castro/ablatingAnalysis/

echo "[STEP 5/5] COMPLETED - All scripts copied to ablatingAnalysis/" | tee -a "$LOGFILE"
