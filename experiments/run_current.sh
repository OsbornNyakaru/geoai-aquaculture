#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 8 — NoPE / permutation-invariant SET encoder, CAPACITY-REMOVING.
#   RESPONSE_05 idea #2. Drop the positional embedding entirely (seq.pos_encoding=none):
#   with bidirectional attention + masked-mean-pool the network becomes a permutation-
#   invariant SET encoder over the observed monthly band-vectors. It can no longer memorize
#   WHICH slot a value sat in — the ultimate deletion of the position channel. The Sentinel-1
#   aquaculture signature (persistent low VH/VV backscatter; temporal median) is a set
#   statistic, so most real signal should survive.
#
#   Context: iter5 relative-time (remove start) WON +0.0128 — start IS shifted train-vs-test.
#   iter7 duration-norm (remove length) LOST −0.0064 — length is already distribution-matched,
#   so there was no shift to remove. NoPE tests the strongest form: remove positional identity
#   ALTOGETHER. Two-tailed — order also carries pond-vs-rice fill/drain timing, so this can go
#   either way — but regardless of public score it is the ideal DIVERSE second private-LB finalist
#   (fails on different rows than the champion). pos_encoding=none is the ONLY variable vs the
#   0.8908 champion; pos_encoding=learned reproduces it bit-for-bit. Held at prevalence_target 0.649.
#
#   DECISION RULE: upload submission_seq_nope.csv, gate vs 0.8908.
#     > 0.8908           -> order was pure nuisance -> NEW CHAMPION; re-baseline.
#     within ~0.01       -> lock as the DIVERSE finalist; next = cross-view invariance objective (iter9).
#     craters (< ~0.87)  -> order carries real signal -> abandon pure NoPE; try the middle ground
#                           (set + a single explicit "duration=L" token) OR go to iter9.
# =====================================================================
set -euo pipefail

python run_pipeline.py --full --model seq --name seq_nope

echo "=== done. Upload submissions/submission_seq_nope.csv (realized pos-rate 0.649) and paste the LB score ==="
