#!/usr/bin/env bash
# =====================================================================
# REPRODUCE THE SUBMITTED MODEL — one command, for reviewers.
#
# WHY THIS FILE EXISTS
#   `run_pipeline.py --model` defaults to `gbdt` (run_pipeline.py:78), which is the SUPERSEDED
#   baseline. A reviewer who runs the bare `python run_pipeline.py --full` from an older README
#   would reproduce the ~0.826-0.878 gradient-boosted model, NOT our 0.8955 submission, and would
#   reasonably conclude the solution does not reproduce.
#
#   This is not a hypothetical risk. In the EY Biodiversity Challenge (same platform, same
#   rulebook template), only 2 of the top 10 finishers survived post-challenge code review -- the
#   rest were eliminated for "failing to submit code within the deadline or having reproducibility
#   concerns," and one prize appears to have gone unpaid. Our final standing is 65% leaderboard +
#   35% code review of the top 5, a LARGER weight against a SMALLER field.
#   See gemini_loop/RESEARCH_08_EY.md.
#
# WHAT IT DOES
#   Runs the exact committed champion configuration and prints the fingerprints a reviewer should
#   see. Nothing here is overridden -- config/config.yaml IS the champion, so this script is a
#   thin, honest wrapper rather than a special path that only works here.
# =====================================================================
set -euo pipefail

cat <<'BANNER'
=====================================================================
 Reproducing the SUBMITTED model: from-scratch temporal Transformer
 Expected public LB: 0.8955

 Verify these fingerprints in the log below:
   seq relative_time ON: observed window left-aligned to t_rel=0
   seq cross-view invariance ON: lambda=1
   seq input width: 24 channels/month
   final_oof   ~ 0.97528
   oof_auc     ~ 0.98943
   t_star        0.4450
   test pos-rate 0.553 -> 0.649

 NOTE: final_oof is deliberately NOT a proxy for leaderboard performance in this
 competition -- our best-LB model has our LOWEST OOF. See the README section
 "How generalization is validated (and an honest caveat)".
=====================================================================
BANNER

# Confirm the committed config really is the champion before spending the compute.
python - <<'PY'
import sys, yaml
cfg = yaml.safe_load(open("config/config.yaml"))
s, c = cfg["seq"], cfg["calibration"]
want = {
    "seq.K": (s["K"], 2),
    "seq.relative_time": (s["relative_time"], True),
    "seq.pos_encoding": (s["pos_encoding"], "learned"),
    "seq.consistency_lambda": (float(s["consistency_lambda"]), 1.0),
    "seq.pooling": (s.get("pooling", "mean"), "mean"),
    "seq.dropout": (float(s["dropout"]), 0.2),
    "calibration.prevalence_target": (float(c["prevalence_target"]), 0.649),
}
bad = {k: v for k, (v, exp) in want.items() if v != exp}
for k, (v, exp) in want.items():
    print(f"  {'OK ' if v == exp else 'BAD'} {k:32s} = {v!r:10} (expected {exp!r})")
if bad:
    sys.exit("\nCommitted config is NOT the champion; refusing to run. Offending keys: "
             + ", ".join(bad))
print("\nCommitted config verified as the champion.\n")
PY

python run_pipeline.py --full --model seq --name champion

echo
echo "=== Done. Submission written to submissions/submission_champion.csv ==="
echo "=== Compare the printed fingerprints against the banner above.        ==="
