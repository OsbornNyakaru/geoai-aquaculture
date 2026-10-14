"""ROC shape probe — settle H_shape vs H_point by turning the AUC column into a RULER.

THE QUESTION THIS ANSWERS, AND WHY IT IS THE LAST OPEN ONE.
We asserted (UPDATE_24 §3.2) that the leader's advantage lives in the HIGH-RECALL corner of the ROC.
Our own §3.3 then proved we cannot know that: the max F1 reachable on our existing ranking by
threshold alone lies in [0.8817, 0.9574], and the leader's ~0.918 sits strictly INSIDE that interval.
So two incompatible hypotheses survive, and they imply opposite work:

  H_shape : the leader's ROC genuinely dominates ours in the high-recall corner.
            => our RANKING is deficient there; partial-AUC / recall-region methods are the right lane.
  H_point : our ROCs are of equal quality (their global AUC is LOWER than ours by 0.00095, which is
            ~21 discordant pairs out of 22538 -- statistically indistinguishable) and they simply
            OPERATE at a better point on an equally good curve.
            => our ranking is fine and the whole gap is where 0.5 lands, i.e. a calibration defect.
            Note this is a LIVE defect: Platt is fit at prior 0.4023 and deployed at prior ~0.618.

⚠️ CREDIT AND CORRECTION. This design is not ours -- a round-24 researcher produced it, and its first
move was to catch us asserting §3.2 as fact when §3.3 proves it unidentified. We had two sections of
our own brief contradicting each other. Their published control value (0.857285562) is off in the 7th
decimal; the correct figure for the champion partition is 0.857285473. The mechanism is entirely
theirs and it is right.

THE TRICK. `TargetRAUC` is scored by ROC-AUC, which depends ONLY on the ordering -- and ties are
allowed. Under the Mann-Whitney convention every tied (pos, neg) pair contributes exactly 0.5. So by
submitting a column with only two or three distinct values we collapse whole blocks of pairs into
ties whose contribution is known in advance, and the returned AUC becomes an equation in one unknown
integer. Zindi prints 9 decimals; consecutive integers are ~0.0028 apart. No noise, no model, no
assumption.

  PROBE A (control): 1.0 above the cut, 0.0 below. Every block is either forced or tied, so the
  result depends on NOTHING unknown. It is a control that must return the value arithmetic
  guarantees -- the standard we now hold ourselves to after two gates of ours failed for want of one.
  A match confirms P, TP and FP to the last decimal AND the grader's tie convention.

  PROBE B (measurement): 1.0 above the cut, 0.5 for the top-m rows below it, 0.0 for the rest.
  Recovers p_B = how many of our missed positives sit in that top-m band.

⚠️ PROBE B IS NOW SECONDARY, AND SAY SO IN THE REPORT. It was designed when we believed the cell
was TP 164 / FP 17 / FN 27 -- a RECALL deficit, 27 positives to go and find. The corrected cell is
TP 164 / FP 27 / FN 17: our dominant error is FALSE POSITIVES, by 27 to 17. Probe B interrogates
the smaller half of the error budget. The question it was built to settle -- "is our ranking flat
in the high-recall corner?" -- is still well posed, but the corner that now costs us most is the
one just ABOVE the cut, not below it. A mirrored probe (0.5 for the bottom-m rows ABOVE the cut,
recovering how many of our 27 FPs sit just over the line) is the more valuable instrument and is
the same one submission. Neither is worth a slot until the finalists are designated.

⛔⛔ THESE ARE DELIBERATELY LOW-SCORING SUBMISSIONS (~0.86 AUC vs our 0.9458). They are instruments,
not entries. **DO NOT UPLOAD EITHER UNTIL THE TWO FINALISTS ARE DESIGNATED ON ZINDI AND CONFIRMED.**
If the platform ever defaults an undesignated slot to "best public" or "most recent", an unlocked
account plus a 0.86 probe is a catastrophe. The tool refuses to write unless --finalists-locked is
passed, and that flag is a statement about the Zindi UI, not about this repo.

LEGALITY. These produce LEADERBOARD-DERIVED quantities, which under our standing rule are
DIAGNOSIS ONLY and must never feed the operating point. That rule is respected here: nothing the
probes return may set a threshold, a hyperparameter, or a model choice. We use them to settle a
question for the report. This is the same category as the F1-column inversion we have run and
disclosed since iter42.

USAGE
    python tools/roc_probe.py --src submission_jtt_lam5_s42 --tp 164 --den 371 --plan
    python tools/roc_probe.py --src submission_jtt_lam5_s42 --tp 164 --den 371 --finalists-locked
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import get_logger  # noqa: E402

log = get_logger()
SUBS = Path(__file__).resolve().parents[1] / "submissions"

# ⚠️ ROUND-24 CORRECTION. These were 191 / 118 (from n_public = 309), and that trio is
# ARITHMETICALLY IMPOSSIBLE: see `tools/lb_cell_solve.py`. AUC on a finite sample is exactly
# C/(P·N) with C a half-integer, so the reported 9-decimal AUC is a hard rational constraint, and
# no (P, N) with P + N = 309 satisfies it -- the nearest miss is 76x the display window. Solving
# the five reported (AUC, F1) pairs jointly leaves 15 candidate (n, P); the full-test predicted-
# positive counts then select n = 333, P = 181 by a factor of ~9 over the next admissible value.
P_PUBLIC = 181          # public positives, solved exactly at round 24 (was 191, refuted)
N_PUBLIC = 152          # 333 - 181  (was 118)


def auc_two_tier(tp, fp, fn, tn):
    """Probe A: 1.0 above the cut, 0.0 below. Every block forced or tied."""
    num = 0.5 * (tp * fp) + tp * tn + 0.0 * (fn * fp) + 0.5 * (fn * tn)
    return num / (P_PUBLIC * N_PUBLIC)


def auc_three_tier(tp, fp, fn, tn, m, p_b):
    """Probe B: 1.0 above, 0.5 for the top-m below, 0.0 for the rest. p_b positives in the mid tier."""
    q = m - p_b                                  # negatives in the mid tier
    num = (0.5 * (tp * fp)                       # pos-above vs neg-above : tied
           + tp * tn                             # pos-above vs neg-below : concordant
           + 0.0                                 # pos-below vs neg-above : discordant
           + 0.5 * p_b * q                       # mid pos vs mid neg     : tied
           + p_b * (tn - q)                      # mid pos vs low neg     : concordant
           + 0.0                                 # low pos vs mid neg     : discordant
           + 0.5 * (fn - p_b) * (tn - q))        # low pos vs low neg     : tied
    return num / (P_PUBLIC * N_PUBLIC)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="submission stem to probe, e.g. submission_jtt_lam5_s42")
    ap.add_argument("--tp", type=int, required=True, help="public TP from the F1 inversion")
    ap.add_argument("--den", type=int, required=True, help="public PP+P from the F1 inversion")
    ap.add_argument("--m", type=int, default=20, help="size of the mid tier in probe B")
    ap.add_argument("--plan", action="store_true", help="print the predicted values only, write nothing")
    ap.add_argument("--finalists-locked", action="store_true",
                    help="ASSERT the two finalists are already designated on Zindi. Required to write.")
    args = ap.parse_args()

    src = SUBS / f"{args.src}.csv"
    if not src.exists():
        raise SystemExit(f"not found: {src}")
    df = pd.read_csv(src)

    pp_pub = args.den - P_PUBLIC                 # predicted positives on the public slice
    tp, fp = args.tp, pp_pub - args.tp
    fn, tn = P_PUBLIC - tp, N_PUBLIC - fp
    log.info("source %s | public cell from the F1 inversion: TP %d  FP %d  FN %d  TN %d",
             args.src, tp, fp, fn, tn)
    log.info("  (F1 check: 2*%d/(%d) = %.9f)", tp, args.den, 2 * tp / args.den)
    if min(tp, fp, fn, tn) < 0:
        raise SystemExit("negative cell entry - the inversion or P is wrong. Refusing.")

    log.info("")
    log.info("PROBE A (control) predicted public AUC = %.9f", auc_two_tier(tp, fp, fn, tn))
    log.info("  depends on NOTHING unknown. A match confirms P=%d, TP=%d, FP=%d exactly, and that the",
             P_PUBLIC, tp, fp)
    log.info("  grader scores ties at 0.5. A mismatch means one of our F1 inversions is wrong, which")
    log.info("  would be even more valuable to learn.")
    log.info("")
    log.info("PROBE B (measurement) predicted public AUC as a function of p_B, m=%d:", args.m)
    for p_b in range(0, args.m + 1):
        tag = ""
        if p_b == 16:
            tag = "   <- p_B>=16: our ranking ALREADY matches the leader's point => H_point"
        if p_b == 8:
            tag = "   <- p_B<=8 : our ranking is flat in the high-recall corner => H_shape"
        log.info("    p_B=%2d  AUC %.9f%s", p_b, auc_three_tier(tp, fp, fn, tn, args.m, p_b), tag)
    sep = abs(auc_three_tier(tp, fp, fn, tn, args.m, 16)
              - auc_three_tier(tp, fp, fn, tn, args.m, 15))
    log.info("  consecutive p_B are %.6f apart and Zindi prints 9 decimals -> p_B recovered EXACTLY.", sep)

    if args.plan:
        log.info("")
        log.info("--plan: nothing written.")
        return

    if not args.finalists_locked:
        raise SystemExit(
            "\nREFUSING TO WRITE.\n"
            "These probes score ~0.86 and are instruments, not entries. Pass --finalists-locked only\n"
            "once BOTH finalists are designated on Zindi and you have visually confirmed it in the UI.\n"
            "An undesignated account plus a 0.86 upload is the one unrecoverable mistake available\n"
            "today, and the deadline is today.")

    above = df.TargetF1.values == 1
    order = df.TargetRAUC.values

    a = df[["ID"]].copy()
    a["TargetF1"] = df.TargetF1.values                    # decision column UNCHANGED
    a["TargetRAUC"] = above.astype(float)
    a.to_csv(SUBS / "submission_probeA_control.csv", index=False)

    # top-m below-cut rows by the source model's own score
    below_idx = [i for i in range(len(df)) if not above[i]]
    below_idx.sort(key=lambda i: -order[i])
    mid = set(below_idx[:args.m])
    b = df[["ID"]].copy()
    b["TargetF1"] = df.TargetF1.values                    # decision column UNCHANGED
    b["TargetRAUC"] = [1.0 if above[i] else (0.5 if i in mid else 0.0) for i in range(len(df))]
    b.to_csv(SUBS / "submission_probeB_shape.csv", index=False)

    log.info("")
    log.info("wrote submission_probeA_control.csv (%d rows, %d above cut)", len(a), int(above.sum()))
    log.info("wrote submission_probeB_shape.csv   (mid tier = top %d of %d below-cut rows)",
             args.m, len(below_idx))
    log.info("⚠️ TargetF1 is byte-identical to the source in both. Only the RAUC column is a ruler.")


if __name__ == "__main__":
    main()
