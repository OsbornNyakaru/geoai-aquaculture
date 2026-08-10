#!/usr/bin/env bash
# =====================================================================
# REPRODUCE THE SUBMITTED FINALISTS — one command, for reviewers.
#
# WHY THIS FILE EXISTS
#   `run_pipeline.py --model` defaults to `gbdt`, the SUPERSEDED baseline. A reviewer running a
#   bare `python run_pipeline.py --full` would reproduce the ~0.826-0.878 gradient-boosted model,
#   NOT our submission, and would reasonably conclude the solution does not reproduce.
#
#   This is not hypothetical. In the EY Biodiversity Challenge (same platform, same rulebook
#   template), only 2 of the top 10 finishers survived post-challenge code review -- the rest were
#   eliminated for "failing to submit code within the deadline or having reproducibility concerns."
#   Our standing is 65% leaderboard + 35% code review of the top 5: a LARGER weight against a
#   SMALLER field. See gemini_loop/RESEARCH_08_EY.md.
#
# ⚠️ READ THIS IF YOU ARE COMPARING AGAINST config/config.yaml
#   The committed config defaults reproduce the ORIGINAL 24-channel champion. Both designated
#   finalists are BUILT ON TOP of it with explicit `--set` flags, shown below and passed here.
#   In particular `seq.channels.permanence=true` with a SINGLE threshold `cdf_taus=[-21.0]` is
#   what makes the 25-channel permanence model; it is deliberately NOT the config default, so
#   that the historical anchors in experiments/anchors.tsv stay reproducible from the same file.
#   Every flag that differs from the committed default is passed explicitly on the command line
#   in this script -- there are no hidden defaults behind either finalist.
#
# OPERATING POINT: runs `calibration.compliance_mode: legal` -- Platt calibration fit on TRAINING
#   out-of-fold predictions only, then a LITERAL 0.5 cut, genuine probabilities in both columns.
#   The rules state: "Setting a probability threshold is strictly forbidden. Your binary target
#   should be based on the default threshold of 0.5." An earlier revision used a prevalence pin
#   that violated that; it survives ONLY as `compliance_mode: pinned` so historical anchors stay
#   reproducible, and its output must never be submitted. See REPORT.md section 8.
#
# STAGE 1 reproduces FINALIST #1 (5 runs + a pool).  STAGE 2 reproduces FINALIST #2 (8 runs + a
#   pool). Pass --quick to stop after stage 1.
# =====================================================================
set -euo pipefail

QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

# The permanence champion = committed config + these two flags. Used by every stage-1 run.
PERM="--set seq.channels.permanence=true --set seq.channels.cdf_taus=[-21.0]"

cat <<'BANNER'
=====================================================================
 Reproducing the SUBMITTED finalists: from-scratch temporal Transformer

   FINALIST #1  champion_perm_seedavg5   public LB 0.899882   (stage 1)
                = 5 seeds x 5 folds of the 25-channel permanence model, legally pooled
   FINALIST #2  champion_archblend4      public LB 0.899643   (stage 2)
                = 4 architecture variants, legally pooled

 WHY BOTH FINALISTS ARE POOLED ARTIFACTS, AND WHY NEITHER IS OUR BEST PUBLIC SCORE
   Our measured seed-to-seed leaderboard sd is 0.0191. Three separate single-seed "records"
   (0.906492, 0.913263, 0.912759) ALL collapsed to ~0.8995 when averaged over 5 seeds. We
   therefore designate pooled, low-variance artifacts for the 721-row private slice rather than
   the best number we ever saw on the 309-row public slice. See REPORT.md section 4.

 STAGE 1 fingerprints to verify in the log below:
   seq relative_time ON: observed window left-aligned to t_rel=0
   seq cross-view invariance ON: lambda=1
   seq input width: 25 channels/month        <- 24 + the permanence indicator
   t_star        0.5000        <- LITERAL. Not fitted. This is the compliance claim.
   Pairwise rank correlation between seeds: mean ~0.95
   POOLED: 5 seeds | test pos-rate ~0.58

 STAGE 2 fingerprints:
   seq input width: 24 channels/month        <- archblend4 predates the permanence channel
   mean pairwise rho = 0.9524
   POOLED: 4 members | test pos-rate 0.5670

 TWO HONEST NOTES FOR THE REVIEWER
 1. final_oof is deliberately NOT a proxy for leaderboard performance here -- our best-LB models
    have our LOWEST OOF, and local OOF (~0.975) has been ANTI-correlated with the LB (~0.90).
    That is the designed covariate shift doing what it was designed to do. See README.
 2. We do not set torch deterministic algorithms for the GPU seq path, and our MEASURED
    seed-to-seed leaderboard spread is 0.0191. Expect run-to-run variation on that order. A
    reproduction that lands within ~0.02 of the stated score IS a successful reproduction; one
    that matches to 4 decimals would be surprising. See REPORT.md section 4.
=====================================================================
BANNER

# Confirm the committed config really is the champion BASE before spending the compute.
python - <<'PY'
import sys
# Use the pipeline's own loader rather than re-reading the file here. It sets
# encoding="utf-8" explicitly (src/utils.py); a bare open() picks up the platform default,
# which is cp1252 on Windows and crashes on the non-ASCII in config.yaml.
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
    # The transductive terms (iteration 41) must be OFF for the designated finalists; with both
    # disabled the pipeline is bit-for-bit the champion (verified against a pristine checkout).
    "seq.transduct.enable":         (bool((s.get("transduct") or {}).get("enable", False)), False),
    "seq.distill.enable":           (bool((s.get("distill") or {}).get("enable", False)), False),
    # THE COMPLIANCE ASSERTION. If this is not 'legal', the run produces a rules-violating
    # submission and the script must refuse rather than silently emit one.
    "calibration.compliance_mode":  (c.get("compliance_mode", "legal"), "legal"),
}
bad = {k: v for k, (v, exp) in want.items() if v != exp}
for k, (v, exp) in want.items():
    print(f"  {'OK ' if v == exp else 'BAD'} {k:32s} = {v!r:10} (expected {exp!r})")
if bad:
    sys.exit("\nCommitted config is NOT the champion base; refusing to run. Offending keys: "
             + ", ".join(bad))
print("\nCommitted config verified as the champion base, in RULES-COMPLIANT mode.\n")
PY

# Reusable compliance audit: readable straight off a submission CSV, trusting none of our code.
audit () {
  python - "$1" "$2" <<'PY'
import pandas as pd, sys
path, tag = sys.argv[1], sys.argv[2]
df = pd.read_csv(path)
ok = (df["TargetF1"].astype(int) == (df["TargetRAUC"] >= 0.5).astype(int)).all()
print(f"\n  COMPLIANCE AUDIT ({tag})")
print(f"    file                                         : {path}")
print(f"    TargetF1 == (TargetRAUC >= 0.5) on every row : {ok}")
print(f"    distinct TargetRAUC values                   : {df['TargetRAUC'].nunique()} / {len(df)}")
print(f"    realized positive rate                       : {df['TargetF1'].mean():.4f}")
if not ok:
    sys.exit("  FAILED: the binary column is not the literal 0.5 cut of the probability column.")
PY
}

# ---- STAGE 1: FINALIST #1 -- champion_perm_seedavg5 (5 seeds, legally pooled). ----
echo "=== STAGE 1/2: champion_perm_seedavg5 (finalist #1) -- 5 runs + pool ==="
for SD in 42 7 13 21 29; do
  python run_pipeline.py --full --model seq $PERM --set seed=$SD --name "perm_single_s${SD}"
done
python tools/seed_average.py --variant perm_single --name champion_perm_seedavg5
audit "submissions/submission_champion_perm_seedavg5.csv" "finalist #1"

if [ "$QUICK" = "1" ]; then
  echo; echo "=== --quick: stopping before stage 2. ==="; exit 0
fi

# ---- STAGE 2: FINALIST #2 -- champion_archblend4 (4 architecture variants, legally pooled). ----
# NOTE these members predate the permanence channel and are reproduced WITHOUT $PERM, at the
# 24-channel width they were actually submitted at.
echo
echo "=== STAGE 2/2: champion_archblend4 (finalist #2) -- 4 runs + pool ==="
python run_pipeline.py --full --model seq --name seq_a_reltime --set seq.consistency_lambda=0
python run_pipeline.py --full --model seq --name seq_a_nope --set seq.consistency_lambda=0 --set seq.pos_encoding=none
python run_pipeline.py --full --model seq --name seq_a_l3   --set seq.consistency_lambda=3
python run_pipeline.py --full --model seq --name seq_a_xview

python tools/arch_blend.py \
  --members seq_a_reltime seq_a_nope seq_a_l3 seq_a_xview \
  --name champion_archblend4
audit "submissions/submission_champion_archblend4.csv" "finalist #2"

echo
echo "=== Done. ==="
echo "  finalist #1  submissions/submission_champion_perm_seedavg5.csv  (LB 0.899882)"
echo "  finalist #2  submissions/submission_champion_archblend4.csv     (LB 0.899643)"
echo "  Compare the printed fingerprints against the banner above."
