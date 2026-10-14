#!/usr/bin/env bash
# Reproduce the paper.
#
#   bash reproduce.sh analyse          every table and figure from the per-image records in results/
#   bash reproduce.sh run <exp>        rerun one experiment; <exp> is one of:
#                                      e2 vsr e3 e1 paraphrase sanity t05 t07 fixlip all
#
# Each experiment writes results/<timestamp>_<name>/ (config.json + results.jsonl); the analysis
# scripts read the latest run of each name. Set PYTHON to choose the interpreter (default: python).
# GPU: needed for Qwen2.5-VL-3B (about 16 GB, e.g. a T4); CLIP and SigLIP also run on CPU, slowly.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
ENCODERS=(clip siglip)
MODELS=(clip siglip qwen)
ALL_CHECKPOINTS="openai/clip-vit-base-patch16 google/siglip-base-patch16-224 Qwen/Qwen2.5-VL-3B-Instruct"

latest() { ls -d results/*_"$1" 2>/dev/null | sort | tail -1; }
have() { [ -n "$(latest "$1")" ]; }
step() { printf '\n==== %s\n' "$*"; }

need_aro() {
  if [ ! -f data/aro/visual_genome_relation.json ]; then
    cat <<'EOF'
ARO VG-Relation annotations not found. Download visual_genome_relation.json from the ARO
release (https://github.com/mertyg/vision-language-models-are-bows) and place it at
data/aro/visual_genome_relation.json. Visual Genome images are then fetched automatically.
EOF
    exit 1
  fi
}

run_e2() {
  step "E2 What's Up (all models)"
  $PY scripts/download_whatsup.py
  for m in "${MODELS[@]}"; do
    $PY scripts/e2_whatsup.py --config "configs/e2_whatsup_${m}.json" --name "e2_whatsup_${m}"
  done
}
run_vsr() {
  step "VSR replication (all models)"
  $PY scripts/e2_vsr.py --models $ALL_CHECKPOINTS
}
run_e3() {
  step "E3 hard negatives (all models)"
  need_aro
  for m in "${MODELS[@]}"; do
    $PY scripts/e3_hardneg.py --config "configs/e3_${m}.json" --name "e3_${m}"
  done
}
run_e1() {
  step "E1 Flickr30k Entities (encoders; downloads the ~4.4 GB image zip)"
  for m in "${ENCODERS[@]}"; do
    $PY scripts/e1_flickr.py --config "configs/e1_${m}.json" --name "e1_${m}"
  done
}
run_paraphrase() {
  step "Paraphrase stability (CLIP)"
  $PY scripts/download_whatsup.py
  $PY scripts/e4_paraphrase.py --n 100 --name e4_paraphrase_clip
}
run_sanity() {
  step "Sanity checks (CLIP)"
  $PY scripts/download_whatsup.py
  $PY scripts/e5_sanity.py --n 100 --name e5_sanity_clip
  $PY scripts/e5_task_conditioning.py --n 40 --name e5_task_conditioning_clip
}
run_t05() {
  step "Background-averaged term (all models)"
  $PY scripts/download_whatsup.py
  $PY scripts/t05_context_average.py run --models $ALL_CHECKPOINTS
}
run_t07() {
  step "Substitute-lexicon sensitivity (all models)"
  need_aro
  for m in "${MODELS[@]}"; do
    $PY scripts/e3_hardneg.py --config "configs/e3_${m}.json" --substitutes contrast --no-first-order --name "e3_${m}_contrast"
  done
}
run_fixlip() {
  step "FIxLIP comparison (encoders; slow: minutes per image)"
  $PY scripts/setup_fixlip.py
  $PY scripts/download_whatsup.py
  for m in "${ENCODERS[@]}"; do
    $PY scripts/e2_whatsup.py --config "configs/e2_fixlip_${m}.json" --name "e2_fixlip_${m}"
  done
}

analyse() {
  mkdir -p results
  if have e2_whatsup_qwen; then
    step "E2 What's Up"
    $PY scripts/analyze_e2.py $(for m in "${MODELS[@]}"; do latest "e2_whatsup_${m}"; done)
  fi
  if have e2_vsr_qwen2.5; then
    step "VSR"
    $PY scripts/analyze_e2.py $(latest e2_vsr_clip) $(latest e2_vsr_siglip) $(latest e2_vsr_qwen2.5)
  fi
  if have e3_qwen; then
    step "E3 failure prediction"
    $PY scripts/analyze_e3.py $(for m in "${MODELS[@]}"; do latest "e3_${m}"; done)
    step "E3 text-only control (loads the text encoders)"
    $PY scripts/e3_text_control.py "$(latest e3_clip)" --model openai/clip-vit-base-patch16
    $PY scripts/e3_text_control.py "$(latest e3_siglip)" --model google/siglip-base-patch16-224
    step "By part of speech";              $PY scripts/analyze_e6.py
  fi
  if have e2_whatsup_clip && have e3_clip; then
    step "Operator / substitute stability"; $PY scripts/analyze_e4.py
  fi
  if have e2_whatsup_clip; then
    step "Held-out deletion faithfulness";  $PY scripts/analyze_faithfulness.py
  fi
  if have e1_clip; then
    step "E1 phrase localisation";         $PY scripts/analyze_e1.py
  fi
  if have t05_ctx_qwen2.5; then
    step "Background-averaged term";       $PY scripts/t05_context_average.py analyse
  fi
  if have e3_qwen_contrast; then
    step "Substitute-lexicon sensitivity"; $PY scripts/t07_lexicon.py
  fi
  if have e2_fixlip_clip; then
    step "FIxLIP comparison";              $PY scripts/compare_fixlip.py
  fi
  step "Cost";                             $PY scripts/analyze_cost.py
  if have e2_whatsup_qwen && have e2_vsr_qwen2.5 && have e3_qwen; then
    step "Figures and appendix tables (paper_coling/figures, paper_coling/tables)"
    $PY scripts/coling_figures.py
    $PY scripts/coling_tables.py
  fi
  echo
  echo "Analysis summaries are also written as summary.md / summary.json inside each run directory."
}

case "${1:-}" in
  analyse|analyze) analyse ;;
  run)
    case "${2:-}" in
      e2) run_e2 ;; vsr) run_vsr ;; e3) run_e3 ;; e1) run_e1 ;; paraphrase) run_paraphrase ;;
      sanity) run_sanity ;; t05) run_t05 ;; t07) run_t07 ;; fixlip) run_fixlip ;;
      all) run_e2; run_vsr; run_e3; run_e1; run_paraphrase; run_sanity; run_t05; run_t07; run_fixlip ;;
      *) echo "usage: bash reproduce.sh run {e2|vsr|e3|e1|paraphrase|sanity|t05|t07|fixlip|all}"; exit 1 ;;
    esac ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
