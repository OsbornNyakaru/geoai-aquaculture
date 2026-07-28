#!/usr/bin/env bash
# =====================================================================
# REPRODUCE THE SUBMITTED FINALISTS — one command, for reviewers.
#
# WHY THIS FILE EXISTS
#   `run_pipeline.py --model` defaults to `gbdt` (run_pipeline.py:78), the SUPERSEDED baseline.
#   A reviewer running a bare `python run_pipeline.py --full` would reproduce the ~0.826-0.878
#   gradient-boosted model, NOT our submission, and would reasonably conclude the solution does
#   not reproduce.
#
#   This is not hypothetical. In the EY Biodiversity Challenge (same platform, same rulebook
#   template), only 2 of the top 10 finishers survived post-challenge code review -- the rest were
#   eliminated for "failing to submit code within the deadline or having reproducibility concerns."
#   Our standing is 65% leaderboard + 35% code review of the top 5: a LARGER weight against a
#   SMALLER field. See gemini_loop/RESEARCH_08_EY.md.
#
# STAGE 1 reproduces the single model (fast, ~1 run).
# STAGE 2 reproduces FINALIST #1, `champion_archblend4`, which needs 9 runs and is what actually
#   scored 0.899643. Pass --quick to skip stage 2.
#
# OPERATING POINT: this runs `calibration.compliance_mode: legal` -- calibration fit on TRAINING
#   out-of-fold predictions only, then a LITERAL 0.5 cut, with genuine probabilities in both
#   columns. The rules state: "Setting a probability threshold is strictly forbidden. Your binary
#   target should be based on the default threshold of 0.5." An earlier revision of this project
#   used a prevalence pin that violated that; it is retained ONLY as `compliance_mode: pinned` so
#   the historical anchors in experiments/anchors.tsv stay reproducible, and its output must not
#   be submitted. See REPORT.md section 7.
# =====================================================================
set -euo pipefail

QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

cat <<'BANNER'
=====================================================================
 Reproducing the SUBMITTED finalists: from-scratch temporal Transformer

   FINALIST #1  champion_archblend4  public LB 0.899643   (stage 2)
   FINALIST #2  seq_a_xview          public LB 0.889686   (stage 1)

 STAGE 1 fingerprints to verify in the log below:
   seq relative_time ON: observed window left-aligned to t_rel=0
   seq cross-view invariance ON: lambda=1
   seq input width: 24 channels/month
   LEGAL calibration ... slope=1.576 ... realized test pos-rate 0.548
   t_star        0.5000        <- LITERAL. Not fitted. This is the compliance claim.
   final_oof   ~ 0.97440
   oof_auc     ~ 0.98943

 STAGE 2 fingerprints:
   mean pairwise rho = 0.9524
   POOLED: 4 members | test pos-rate 0.5670

 TWO HONEST NOTES FOR THE REVIEWER
 1. final_oof is deliberately NOT a proxy for leaderboard performance here -- our best-LB model
    has our LOWEST OOF, and local OOF (~0.975) has been ANTI-correlated with the LB (~0.89).
    That is the designed covariate shift doing what it was designed to do. See README.
 2. We do not set torch deterministic algorithms for the GPU seq path, and our MEASURED
    seed-to-seed leaderboard spread is 0.0191. Expect run-to-run variation on that order. A
    reproduction that lands within ~0.02 of the stated score IS a successful reproduction; one
    that matches to 4 decimals would be surprising. See REPORT.md section 4.
=====================================================================
BANNER

# Confirm the committed config really is the champion BEFORE spending the compute.
python - <<'PY'
import sys
# Use the pipeline's own loader rather than re-reading the file here. It sets
# encoding="utf-8" explicitly (src/utils.py:36); a bare open() picks up the platform
# default, which is cp1252 on Windows and crashes on the non-ASCII in config.yaml.
# Re-implementing the read is also how this check would silently drift from the pipeline.
from src.utils import load_config
cfg = load_config()
s, c = cfg["seq"], cfg["calibration"]
want = {
    "seq.K":                        (s["K"], 2),
    "seq.relative_time":            (s["relative_time"], True),
    "seq.pos_encoding":             (s["pos_encoding"], "learned"),
    "seq.consistency_lambda":       (float(s["consistency_lambda"]), 1.0),
    "seq.pooling":                  (s.get("pooling", "mean"), "mean"),
    "seq.dropout":                  (float(s["dropout"]), 0.2),
    "seq.channels.drop_bands":      (list(s.get("channels", {}).get("drop_bands") or []), []),
    # THE COMPLIANCE ASSERTION. If this is not 'legal', the run produces a rules-violating
    # submission and the script must refuse rather than silently emit one.
    "calibration.compliance_mode":  (c.get("compliance_mode", "legal"), "legal"),
}
bad = {k: v for k, (v, exp) in want.items() if v != exp}
for k, (v, exp) in want.items():
    print(f"  {'OK ' if v == exp else 'BAD'} {k:32s} = {v!r:10} (expected {exp!r})")
if bad:
    sys.exit("\nCommitted config is NOT the champion; refusing to run. Offending keys: "
             + ", ".join(bad))
print("\nCommitted config verified as the champion, in RULES-COMPLIANT mode.\n")
PY

# ---- STAGE 1: finalist #2, the single model. ----
echo "=== STAGE 1/2: single model (finalist #2) ==="
python run_pipeline.py --full --model seq --name seq_a_xview

# Compliance is auditable from the CSV alone, without trusting any of our code.
python - <<'PY'
import pandas as pd, sys
df = pd.read_csv("submissions/submission_seq_a_xview.csv")
ok = (df["TargetF1"].astype(int) == (df["TargetRAUC"] >= 0.5).astype(int)).all()
uniq = df["TargetRAUC"].nunique()
print(f"\n  COMPLIANCE AUDIT (readable straight off the submission file)")
print(f"    TargetF1 == (TargetRAUC >= 0.5) on every row : {ok}")
print(f"    distinct TargetRAUC values                   : {uniq} / {len(df)}")
print(f"    realized positive rate                       : {df['TargetF1'].mean():.4f}")
if not ok:
    sys.exit("  FAILED: the binary column is not the literal 0.5 cut of the probability column.")
PY

if [ "$QUICK" = "1" ]; then
  echo; echo "=== --quick: stopping before stage 2. ==="; exit 0
fi

# ---- STAGE 2: finalist #1. Needs every archblend4 member + its seed replicates. ----
echo
echo "=== STAGE 2/2: champion_archblend4 (finalist #1) -- 8 more runs ==="
python run_pipeline.py --full --model seq --name seq_a_reltime --set seq.consistency_lambda=0
python run_pipeline.py --full --model seq --name seq_a_reltime_s7 --set seed=7 --set seq.consistency_lambda=0
python run_pipeline.py --full --model seq --name seq_a_nope --set seq.consistency_lambda=0 --set seq.pos_encoding=none
python run_pipeline.py --full --model seq --name seq_a_l3   --set seq.consistency_lambda=3
for SD in 7 13 21 29; do
  python run_pipeline.py --full --model seq --name "seq_a_xview_s${SD}" --set seed=$SD
done

python tools/arch_blend.py \
  --members seq_a_reltime seq_a_nope seq_a_l3 seq_a_xview \
  --name champion_archblend4

python - <<'PY'
import pandas as pd, sys
df = pd.read_csv("submissions/submission_champion_archblend4.csv")
ok = (df["TargetF1"].astype(int) == (df["TargetRAUC"] >= 0.5).astype(int)).all()
print(f"\n  COMPLIANCE AUDIT (finalist #1)")
print(f"    TargetF1 == (TargetRAUC >= 0.5) on every row : {ok}")
print(f"    realized positive rate                       : {df['TargetF1'].mean():.4f}")
if not ok:
    sys.exit("  FAILED: the binary column is not the literal 0.5 cut of the probability column.")
PY

echo
echo "=== Done. ==="
echo "  finalist #1  submissions/submission_champion_archblend4.csv   (LB 0.899643)"
echo "  finalist #2  submissions/submission_seq_a_xview.csv           (LB 0.889686)"
echo "  Compare the printed fingerprints against the banner above."
