# PROJECT_STATE — single source of truth (portable across cloud accounts)

> **What this file is.** The one document you carry to any cloud account. It lives in the
> git repo, so a fresh Colab/Kaggle account gets it automatically on `git pull`. It holds
> everything: how to resume anywhere, the current champion, every experiment + output + LB
> score, what improved, what declined, the lessons, and the next action.
>
> **It is updated EVERY session.** If you're reading this on a new account, it is current as
> of the "Last updated" line below. Supporting files (`experiments/LB_LOG.md`,
> `gemini_loop/AGENT_BRIEF.md`, `RUN_ON_CLOUD.md`, `RUN_ON_KAGGLE.md`) go deeper; this file
> summarizes and points to them.

- **Competition:** GeoAI Aquaculture Pond Identification (Zindi / FAO / ITU)
- **Repo:** `OsbornNyakaru/geoai-aquaculture` (private) · branch `main`
- **Deadline:** 2026-08-16 · **Submissions:** max 5/day (manual upload to Zindi; no API)
- **Last updated:** 2026-08-07 (iter38: mean_min pooling = 0.912759 (+0.0063, clears gate) — 3rd single-seed-42 candidate, iter39 SEED-CONFIRMS it + runs the free tree-lane gate; tree infra mapped, adversarial_cv.py built) · **🔬 iter38: `mean_min` pooling (mean + temporal LOW-TAIL of hidden state) = 0.912759, beats mean_std/moments — the low tail (permanent-low-scatterer pond signal) matters more than spread. Single-seed-42 (soft & vhsq both washed out before) → iter39 seed-confirms vs perm seed-avg 0.899882; if real, iter40 builds the n-invariant low-quantile (p10/p25) robust form. TREE LANE: infra is ~80% built (windowed features + n-invariant gating + CatBoost ensemble all exist; iter30's 0.995→0.70 was the covariate shift, not windowing). Built `tools/adversarial_cv.py` = free GO/NO-GO gate (test-like-holdout vs random-holdout); run before spending any tree submission. Remaining tree build (gated on GO): monotone constraints, IQR/L-scale features, feature-shift removal, early-stop.** · **⚠️ DISCIPLINE NOTE: the iter35 c_repl_vhsq 0.913263 "NEW BEST" did NOT survive seed-confirmation (5-seed avg 0.899512 ≈ plain permanence seed-avg 0.899882). It was seed-42 public luck. Seed-robust ceiling remains ~0.900. Single-channel feature tweaks (adds AND replacements) all wash out → next gains must be STRUCTURAL (pooling / trees / architecture). Free MODE-A diagnostic: test masking is MAR (tent-shaped month coverage) but real covariate shift exists on the SAR level (mean_VH/VH² KS not closed by windowing). shift_diagnostics MODE-B bug FIXED. Finalists stay {perm seed-avg 0.899882, archblend4 0.899643}.** · **🏆 c_repl_vhsq = 0.913263 = NEW BEST EVER (+0.0068 vs 0.906492, first >0.9065). Drops the duplicate VV missing-indicator (R=1, info-free) and adds VH² in its place → width stays 25. VH² was toxic as a 26th ADD (iter34 −0.018) but WINS as a REPLACEMENT → width was the enemy, not the coordinate; the "replace-not-add" cell + the dispersion/2nd-moment axis are both confirmed. ⚠️ single-seed-42 → iter37 seed-confirms (5 seeds + seed-avg vs 0.899882) + tests if SWA stacks. Soft permanence (0.899958) & rice-gate replacement (0.899173) HURT — dropped. BUILT `tools/shift_diagnostics.py` (FREE, legal): MODE-A MAR-vs-seasonal dropout check (gates ALL distributional features) + MODE-B adv-AUC/label-AUC feature screen (submission-free ranker, replaces anti-correlated OOF). Deep research (gemini_loop/research_md/) converges w/ our 8 agents; adds the Linear→GELU→mean-pool edit + IQR/MAD dispersion.** · **🔬 ROUND-17 RE-OPENS (literature-backed, we WILL explore): (1) temporal DISPERSION via L-scale/GMD not std (unbiased U-stat, THE rice-vs-pond separator — our std test was n-biased); (2) TREES done right on windowed-CV (leader's CatBoost lane; our 0.995→0.70 was fixable shift-leak, and trees can use the relative/rank features our mean-pool provably can't); (3) OPTICAL greenness axis (c_repl_ricegate already tests it — get its LB); (4) moment/quantile POOLING; (5) corrected ROCKET PPV; (6) SWA/SWAD (BUILT). FREE checks gate all: MAR-vs-seasonal dropout, windowed-CV harness, adv-AUC pre-screen. Finalists {c_perm_single 0.906492, archblend4 0.899643}. Saerens OFF (gate FAILED = conditional shift). Full detail: scratchpad/research17_digest.md, UPDATE_17.md.** · **🔑 iter35 GATE RESULT (real data, preds_c_perm_single): FAIL — KS D=0.186, p=0.000 → the shift has a real CONDITIONAL component, Saerens/MLLS UNSAFE, do NOT apply. π_t est MLLS 0.578 / BBSE 0.559 (agree, ~0.56-0.58 < believed 0.65; current pos-rate already 0.550 → upside ~nil regardless). Gate saved a bad submission. IMPLICATION: permanence single-feature bet is fragile on private → archblend4 is a REAL finalist hedge. Capacity-neutral arms RAN, LB pending (OOF blind): soft 0.97585 / vhsq 0.97411 / ricegate 0.97442. Upload c_perm_soft + c_repl_vhsq (paired vs 0.906492). seed-avg reproduced (pooled pos-rate 0.5816 = the 0.899882 finalist).** · **🟢 the offline LABEL-SHIFT GATE (`tools/label_shift_gate.py`, 0 submissions, validated on synthetic label-vs-conditional shift) decides whether the LEGAL Saerens prior-shift correction (~+0.010..+0.019 on the F1 half, capacity-free) is safe — adv-AUC 0.89 was the WRONG instrument (Agent 6 rebutted Agent 3); the mixture goodness-of-fit test is the right one. Run it on any preds bundle; if PASS, `--emit-submission` writes the Saerens-corrected file (AUC unchanged, only F1 moves). · REAL single-τ perm seed dist: 42=0.906492, 29=0.900715, 13=0.891730, 21=0.878575, 5-seed-avg=**0.899882** — seed-avg is +0.0055 OVER member mean = a REAL pooling LEVEL gain (perm seeds less rank-corr than base) → `champion_perm_seedavg5_st`=0.899882 is the MEASURED robust FINALIST (tied w/ archblend4 0.899643, zero seed luck).** · **🔬 ROUND-16 (all 8 agents): iter34's OOF↑/LB↓ is a SHIFT-CARRIER divergence tax (Ben-David d_HΔH + Δλ*), NOT capacity noise — cure is CAPACITY-NEUTRAL. Three untested theory-backed moves: (1) SOFT permanence σ(0.5·(−21−VH)) = optimal rank-1 LLR, kills n=5 quantization [Agent 4 TOP]; (2) CHANNEL REPLACEMENT — VH/VV missing-indicators are identical (R=1) so drop one & add VH²/rice-gate at const width 25 = the untested 2×2 cell [Agents 1&2]; (3) LOW-RANK bottleneck k=3 [Agent 1, iter36]. Hard-τ scan KILLED (τ=−21 on physical optimum). Offline submission-free screen derived: ADD IFF (2·labelAUC−1)>κ(2·advAUC−1), ADV≤0.56 & LABEL≥0.75. Saerens/BBSE likely UNSAFE (adv-AUC 0.89 = covariate not pure-label shift). 4 agents (SWA/Saerens/finalist/analogous) pending session-limit reset. All code for (1)(2) built+width-verified (25 ch), iter35 staged.** · **⚠️ iter34 RESULT: adding a 2nd channel to the permanence champion HURTS — pondband 0.900373 (−0.0061), vhsq 0.894697 (−0.0178), vvperm 0.891166 (−0.0153); ricegate not yet uploaded (out of slots, low prior). Clean mechanism: every arm's OOF ROSE but every LB FELL → extra Linear params overfit the adv-AUC-0.89 shift. Pos-rate direction predicted the sign (all 3 losers moved AWAY from true ~0.65). OOF anti-correlated again (highest-OOF pondband still lost). NEXT: capacity-NEUTRAL single-τ scan {−22,−20.5,−20} + the REAL single-τ seed-avg (finalist).** · **⚠️ iter33 RESULT: `champion_perm_archblend4` = 0.892939 (bottom bucket). The permanence +0.010 and the pooling +0.010 are SUBSTITUTES, not complements — both monetize operating-point disagreement at the 0.5 cut, and a shared strong feature collapses it. −0.0067 vs base archblend4, −0.0136 vs the single c_perm_single (0.9065). STOP pooling permanence. `champion_archblend4` (0.899643) stays the ensemble finalist; `c_perm_single` (0.906492) is the single-model champion. iter34 grows the SINGLE model: 4 second-feature candidates (VV permanence, pond-band occupancy, VH², SARxoptical rice-gate) from UPDATE_15, one at a time on the permanence base, seed 42 directional → iter35 seed-confirm the winner. UPDATE_15 (round-15 deep-research brief) shipped.** · **🏆 THE FEATURE LANE IS OPEN, AND FEATURE-SELECTED. `c_perm_single` (champion + ONE per-month VH permanence indicator `1[VH_dB(t)<−21]`) = 0.906492 — highest public score ever. Monotone τ selection: 1-τ 0.9065 > 4-τ 0.9016 > 6-τ 0.8987 (the signal is ONE physically-privileged cut ~−21 dB; extra thresholds add noise). Seed-confirmed: 4-τ seed-avg 0.8969 = +0.010 vs champion seed-avg. 🔬 ROUND-15 RESEARCH IN PROGRESS (8 subagents → UPDATE_15 for Claude+Gemini deep research). cross_pol `VH−VV` TOXIC (affine-spanned). cross_pol (`VH−VV`) isolated + TOXIC (−0.0228), dropped. First feature win in the Transformer; caught by ONE-CHANGE-AT-A-TIME isolation (combined c_permxpol = 0.8788 would have masked it). GATE: single seed-42 (our lucky seed) → iter32 seed-confirms before it's a finalist. Direction: PERMANENCE. iter30 CatBoost FAILED (catblend5 −0.0136, standalone 0.698); ensemble lane stays closed but features are alive.** · **🏆 THE LEGAL BOARD (pre-permanence): `champion_archblend4` (legal) = 0.899643 — highest ELIGIBLE score AND our current finalist #1; `seq_a_xview` (legal) = 0.889686. `c_perm` 0.901605 supersedes these IF seed-confirmed. Every artifact scored under the prevalence pin is INELIGIBLE and must not be designated.** · **🔑 GOING LEGAL MADE THE ENSEMBLE WORK: pinned archblend4−champion = −0.0009 (nothing); legal = +0.0100. The pin overwrote every member's operating point to 0.649, so pooling could only average ORDER (ρ=0.9524, nothing to gain). Under a literal 0.5 cut it also averages CALIBRATION — member pos-rates spanned 0.534–0.586 — which the pin was discarding. iter18's 'pooling is marginal' verdict was an artifact of the operating point.** · Historical (illegal) champion 0.8955 single seed / reliable ≈0.8865; historical leading finalist `champion_archblend4` = 0.894643. · **Loop state: 🏁 ARCHITECTURE SEARCH CLOSED. iter24 ran and FAILED, significantly: `champion_gbdtblend5` = 0.879123, −0.0155 vs archblend4 on a strongly PAIRED comparison (4/5 shared members, identical 309 rows) → ≥2.5σ, not noise. The result INVERTS its own hypothesis: the GBDT was equally decorrelated as ROCKET (ρ 0.849 vs 0.850) but ~4× stronger (−0.011 vs −0.040 LB), and its blend cost NEARLY TWICE AS MUCH (−0.0155 vs −0.0090). Cross-model-class blending is now closed with n=2 across maximally different families. THE LAW: under a rank-only metric with a PINNED threshold, a decorrelated member's reordering near the cut costs more level than its independence buys back — only rank-twins (ρ>0.93, same class) pool without loss, and those buy variance, never level. METHODOLOGICAL LESSON: the screen said `g_gbdt votes=1/2 → HOLD` and was RIGHT; it was overridden by extending iter18's "read the matrix not the screen" rule beyond its derivation (within-class pooling, where members are equally competent). Do NOT upload `champion_xview_gbdt` (½ weight on the member that just cost −0.0155 at ⅕). · **iter25 (2026-07-27) then reopened ONE narrow lane from round-11 research + a local Phase-A audit that cost 0 submissions: the shift lives in the VALUES (adv-AUC 0.8915 masked+left-aligned) and a 2-D screen (A=shift vs T=signal) names VV and blue as free deletions — with the SAR literature independently rejecting VV. Capacity-REDUCING, the only class that has ever won. Tempered: the shift is DISTRIBUTED (max single-band A 0.59 vs joint 0.89), so this cannot collapse it. iter25 RAN 2026-07-28: gate PASSED (ATCF1 15/15, DIS 5/5) and `c_dropvv` CLEARED THE PRE-COMMITTED RULE — 2/2 votes, ATC-F1 0.8977 vs the champion's 5-seed range [0.7196, 0.8601], i.e. above its BEST draw. `c_dropvvblue` also cleared but is a ρ=0.9841 twin whose extra band (blue) is itself a HOLD with a negative margin. iter26 UPLOADED IT: **LB = 0.884217, −0.0113 PAIRED — the screen was WRONG IN SIGN.** Feature-space deletion CLOSED. The finding is about the INSTRUMENT: all 7 original anchors sit at an identical 24-channel width, so ATC-F1 was only ever certified WITHIN that family; `c_dropvv` (22 ch) was the first REPRESENTATION change and adding it as an 8th anchor drops ρ +0.964 → +0.738 as the single discordant pair — while the gate still reads 17/18 PASS, so **the gate does not catch out-of-family failure.** The ratio battery is also a representation change: do NOT screen it with ATC-F1. Second consecutive ~0.015 over-prediction by me, both on out-of-family candidates — the 3× ATC-F1 discount is calibrated in-family and is not enough outside it.** · **🔑 NEW 2026-07-28 — THE PUBLIC-LB LEADER (`sdv`) DESCRIBED THEIR APPROACH ON THE FORUM AND IT SAYS WE HAVE BEEN SEARCHING A FLAT AXIS: they are at ~0.94 with PLAIN CATBOOST ("the model isn't the bottleneck"), they tried regime-mimicking validation and it "barely correlated", and their named lever is RATIO/RELATIVE FEATURES over absolute values. Our ≈0.05 gap is therefore entirely in FEATURES, not model class or ensembling — the last 8 iterations searched architecture. We have NEVER tested a cross-band ratio; `VH − VV` (the dB log cross-pol ratio) is queued and unrun. NOTE this does not contradict "amplitude is the signal": we tested WITHIN-SERIES TEMPORAL RANK (destroys level), not a CROSS-BAND ratio at fixed t (preserves level, cancels per-period gain drift). We conflated two different transformations. · NEXT ACTION: upload c_dropvv, then iter26 = ratio-feature battery screened on the free 2-D A/T screen, AND the Phase-Two writeup in parallel — 35% of the top-5 rubric, still does not exist, deadline 2026-08-16.**
- **🚨 READ FIRST if you are a fresh session — three corrections, one of them fatal to the ledger:**
  1. **SEED VARIANCE IS 0.0191, MEASURED (2026-07-22).** The champion configuration, changing *only*
     the RNG seed, scored **0.8955 (seed 42)** vs **0.8764 (seed 7)**. **Nine of our eleven recorded
     verdicts have effect sizes smaller than that** — including relative-time (+0.0128) and the
     cross-view win (+0.0047) that made this model "champion". **Do not trust any A/B in the ledger
     below ±0.019 unless it was seed-paired.** The ±0.01 floor we operated under for fifteen
     iterations came from row-count theory, never from a measurement.
  2. **Pretrained models are LEGAL** — TabPFN and Presto were rejected on a misreading of the rules.
  3. **The metric is RANK-ONLY** — the LB is blind to calibration.
  See §6, `experiments/LB_LOG.md` (the seed section), and `gemini_loop/RESEARCH_07.md`.

---

## 1. Resume in 60 seconds (any account)

1. `git pull` the repo (see §2 for the per-platform loop).
2. Read this file top-to-bottom — you're now caught up.
3. **Current next action: `Run all` → iter39 (SEED-CONFIRM mean_min + FREE tree-lane gate + MODE-B screen) on Colab.**
   iter38 found mean_min pooling = 0.912759 (single-seed-42); iter39 seed-confirms it (5 seeds + seed-avg vs
   0.899882) — ≥0.9059 = real seed-robust win. iter39 also runs the two FREE offline gates: `adversarial_cv.py`
   (tree-lane GO/NO-GO) and the fixed MODE-B feature screen. If mean_min confirms → iter40 = n-invariant
   low-quantile pooling (robust form of min). If adversarial_cv = GO → build shift-robust CatBoost lane. Also
   upload pending champion_replvhsq_swa_seedavg5 (SWA level test). Finalists {perm seed-avg 0.899882,
   archblend4 0.899643} until something confirms above them; Saerens OFF.
   Single-channel tweaks are tapped; iter38 tests STRUCTURAL dispersion pooling (mean_std / mean_min /
   moments — already implemented) on the permanence champion, seed 42 directional vs 0.906492. ≥0.9125 →
   seed-confirm + build the unbiased L-scale/GMD pooling (iter39). Flat → the built-in std is biased (window
   artifact) so iter39 tries L-scale; if that's flat too, the Transformer dispersion axis is tapped → pivot
   to the TREE lane (the biggest remaining ceiling: windowed-CV harness + shift-robust CatBoost + n-invariant
   feature bank; trees can use the ratio/rank features our mean-pool can't, and the leader's ~0.94 is on
   CatBoost). ALSO pending (already generated, 1 upload each): champion_replvhsq_swa_seedavg5 (SWA level test)
   + re-run `tools/shift_diagnostics.py --mode screen` for the fixed MODE-B feature ranking. Finalists stay
   {perm seed-avg 0.899882, archblend4 0.899643}; Saerens OFF.
   iter37 runs `tools/shift_diagnostics.py` (free: MAR-vs-seasonal dropout + adv-AUC/label-AUC feature
   screen) then seed-confirms c_repl_vhsq at 5 seeds + seed-avg (vs 0.899882) and tests whether SWA stacks
   on it. Read: seedavg ≥ 0.9059 → real seed-robust champion & new finalist #1. Then iter38 = adv-AUC-
   SCREENED replacements (L-scale/IQR dispersion — the properly-unbiased forms the research prefers over
   VH²; VV-permanence) + (if MODE-A/windowed-CV pass) the shift-robust CatBoost lane + the GELU-pool edit.
   Finalists now: **champion_replvhsq_seedavg5 (if confirmed) + champion_archblend4 0.899643** (decorrelated,
   Clark E[max]); fallback perm seed-avg 0.899882 + archblend4. Saerens OFF (gate FAILED).

   [superseded] iter36 SWAD — folded into iter37 arm 2 (SWA-on-champion). iter35 gave the win above.
   ROUND-17 (8 agents, open posture) is done → 6 literature-backed RE-OPENS (details: scratchpad/
   research17_digest.md). **Do the FREE offline checks first (0 submissions):** (F1) MAR-vs-SEASONAL dropout
   diagnostic — subsample train to random 5-consec-month windows, compare permanence/L-scale distributions
   vs full-12; if seasonal, ALL distributional features (incl. permanence) are compromised — this gates
   everything. (F2) WINDOW-LENGTH-MATCHED (4-6mo) CV harness — makes offline decisions correlate w/ LB &
   enables the tree lane. (F3) adv-AUC(train-vs-test) pre-screen of each candidate feature. **Then submissions
   (seed-avg, paired vs 0.906492), ranked:** iter36 SWAD (BUILT) → (S1) L-scale/GMD dispersion of VH/VV as
   static feature [TOP: we tested dispersion with std=n-biased; L-scale is the exactly-unbiased U-stat and the
   literature's #1 pond-vs-rice separator] → (S2) optical greenness replacement (+ get c_repl_ricegate LB) →
   (S3) shift-robust CatBoost on windowed CV [the leader's ~0.94 lane; our 0.995→0.70 was shift-leak + wrong-
   regime CV, both fixable; trees can use the relative/rank features our mean-pool PROVABLY can't] → (S4)
   moment/quantile pooling head → (S5) corrected ROCKET PPV bank. **Finalists {c_perm_single 0.906492 +
   archblend4 0.899643}** (Clark E[max]: decorrelated + fat tail; NOT two permanence variants). **Saerens
   stays OFF** (gate FAILED = conditional shift). UPDATE_17.md + research16/17 digests hold the full brief.

   [superseded] iter35 (CAPACITY-NEUTRAL levers) — RAN; gate=FAIL, arms pending LB.
   iter34 SETTLED that adding a channel HURTS; round-16 research says WHY (shift-carrier divergence tax,
   not capacity) and that the cure is capacity-NEUTRAL. iter35 arms (all width 25, seed 42 directional,
   paired vs c_perm_single 0.906492): (1) `c_perm_soft` — soft permanence σ(0.5·(−21−VH)), Agent-4 top
   pick, best odds to clear; (2) `c_repl_vhsq` / `c_repl_ricegate` — drop the duplicate VV missing-
   indicator (R=1, info-free) and add VH²/rice-gate at CONSTANT width (the untested 2×2 cell); (3)
   `champion_perm_seedavg5_st` — the REAL single-τ 5-seed avg finalist (the 0.8969 upload was the STALE
   4-τ file; expect ≥0.90). Upload priority: seed-avg finalist → c_perm_soft → better replacement arm.
   ≥0.9125 → capacity-neutral WIN, seed-confirm iter36. Within noise → lane flat, FINALIZE.
   iter35 ALSO runs `tools/label_shift_gate.py` on the champion bundle (0 subs) → prints PASS/FAIL for the
   Saerens F1 lever. If PASS: emit the corrected finalist (`--emit-submission`), upload = likely +0.010-0.019.
   **iter36 if needed:** low-rank k=3 bottleneck (Agent 1) or SWAD tail-averaging (Agent 5), both need a build.
   **Deadline-bound:** DESIGNATE finalists (`champion_perm_seedavg5_st` 0.899882 + `champion_archblend4`
   0.899643 — real, tied, decorrelated; OR swap in the lucky 0.906492 single seed as the risky E[max] leg,
   pending UPDATE_16 Q5) + Phase-Two writeup. UPDATE_16.md ready to feed Claude/Gemini deep research.

   [prior] iter32 (SEED-CONFIRM the permanence win) — DONE, confirmed.
   🏆 **THE FEATURE LANE OPENED (iter31, 2026-07-29).** Adding per-month VH permanence indicators
   `1[VH_dB(t)<τ]` as Transformer channels scored **c_perm = 0.901605 — our BEST public score ever**,
   above `champion_archblend4` (0.899643), as a single legal model. Isolated one-at-a-time, `VH−VV`
   cross_pol was TOXIC (−0.0228) → dropped. This is the first feature win inside the Transformer, and
   the user's "one change at a time on one direction" discipline is what caught it (the combined
   `c_permxpol` = 0.8788 would have looked like a loss). **Direction now: PERMANENCE.**
   **⚠️ Gate:** c_perm is single-seed-42 (our lucky seed); +0.0119 is at the ~0.013 resolution. iter32
   seed-confirms it — `c_perm` at 5 seeds vs the champion at the same 5 seeds → `champion_perm_seedavg5`
   vs `champion_seedavg5`. **≥ +0.006 seed-avg → real win, new finalist**; within noise → seed-42 luck.
   Plus two one-at-a-time τ-selection probes (`c_perm_single` τ=−21, `c_perm_wide` 6 thresholds).
   0 new code (τ grid is a config override). **iter30 (CatBoost) FAILED** (catblend5 −0.0136, standalone
   0.698 — OOF-AUC illusion); ensemble/model-class lane stays closed, but FEATURES are now alive.
   **Still deadline-bound regardless:** DESIGNATE finalists on Zindi + the Phase-Two writeup (35%, unbuilt). Optional single
   remaining probe with any real thesis: VH-CDF permanence + `VH−VV` as CHANNELS in the Transformer
   (the model that transfers) — but it is an unscreenable representation change and "expect small".

   **🏆 2026-07-28 — WE ARE LEGAL *AND* AT OUR BEST SCORE EVER.** `champion_archblend4` rebuilt
   through the compliant path = **0.899643**, above every artifact we have ever submitted
   (previous best `c_meanmin` 0.898566, illegal). Going legal made it **+0.005 BETTER** than its own
   pinned version. Finalist #2 is `seq_a_xview` (legal) = 0.889686.

   **⛔ DESIGNATE MANUALLY.** Zindi's default is your two best *public* scores — and several of
   those are the INELIGIBLE pinned artifacts. Do this with days to spare, not on 2026-08-16.

   **🔑 The pin was suppressing the ensemble** (see LB_LOG iter28). Pinned archblend4−champion =
   −0.0009; legal = +0.0100. The pin overwrote every member's operating point to 0.649, so pooling
   could only average ORDER (ρ=0.9524 — nothing there). A literal 0.5 cut also averages
   CALIBRATION, where the members genuinely disagreed (pos-rates 0.534–0.586). **This means several
   pin-era ensemble rules are now UNVERIFIED and must be re-derived before use** — including "gate
   members on level gap, not correlation" and the rank-correlation go/no-go, which printed SKIP at
   ρ=0.9524 and was wrong.

   **The one cheap probe worth running:** if pooling now buys level through *calibration* diversity,
   a LARGER pool should help. Test `archblend6` (add `seq_a_k4`, `seq_a_base` — whose legal pos-rates
   0.534/0.530 sit at the edge of the current spread) against archblend4's 0.899643. One submission,
   and it directly tests whether the pin-era "weak members drag" rule survives. Do NOT let this
   displace the writeup.

   **✅ 2026-07-28 — WE ARE NOW RULES-COMPLIANT, AND IT COST 0.0058.** The prevalence pin was an
   explicit rules violation (verified verbatim on the live rules page: *"Setting a probability
   threshold is strictly forbidden... default threshold of 0.5"*, plus *"Zindi will need the raw
   probabilities"* — our RAUC column was emitting uniform RANKS). Both are fixed:
   `calibration.compliance_mode: legal` = Platt fit on TRAINING OOF only, literal 0.5 cut, real
   probabilities in both columns. Measured PAIRED on the LB: pinned 0.8955 -> **legal 0.889686**,
   delta **-0.0058**, which is BELOW our own 0.006 suggestive threshold. Our reliable pinned level
   was 0.8865, so the legal single seed is *above* it. The pin was adding 104 positives that were
   ~49% correct — coin flips. The +0.07 it was credited with came from iteration 02 on the
   SUPERSEDED GBDT and was never re-measured on the transformer.

   **⚠️ ALL SIX PREVIOUSLY SUBMITTED ARTIFACTS ARE UNUSABLE.** Every one was produced by the
   illegal path, so **none can be designated as a finalist.** The legal board has exactly one
   entry: `seq_a_xview` (legal) = **0.889686**. Rebuilding a legal `champion_archblend4` — which
   was finalist #1 — is the top priority.

   **⚠️ A BUG WAS CAUGHT BEFORE UPLOAD, and its lesson generalizes.** The blenders fed normalized
   RANKS into Platt; a rank transform maps OOF and test to uniform SEPARATELY and destroys the
   train-vs-test level difference, collapsing the positive rate onto the train prior (0.402/0.422
   observed vs 0.53-0.60 for individual models). Root cause is conceptual: our "always
   rank-average, never probability-average" law was derived UNDER THE PIN, where the pin
   re-derived the cut so only ORDER mattered. Under a literal 0.5 cut the LEVEL is the entire
   TargetF1 column. Fixed via `calibrated_pool()`. **The recurring failure mode is extending a
   rule past the regime it was derived in — this is the third instance (iter24, iter26, here).**

   Superseded next action (still live, lower priority): iter28 = the RATIO-FEATURE battery, tested ON THE LB (the offline screen is
   no longer trustworthy for representation changes — see the ⛔ below), and START the Phase-Two
   writeup in parallel.**

   **⛔ 2026-07-28 — THE OFFLINE SCREEN IS CERTIFIED WITHIN ONE FAMILY ONLY. READ BEFORE SCREENING
   ANYTHING.** iter26 uploaded `c_dropvv` on a 2/2 SUBMIT call with an ATC-F1 margin of +0.0902 (1.57
   seed-sd, above the champion's best-of-five seed draw). **LB came back 0.884217 — negative, −0.0113
   paired.** The screen was wrong in **sign**, the first time a SUBMIT call was followed and failed.
   Cause: all 7 original anchors are architecture/objective variants at an **identical 24-channel input
   width**, so ATC-F1 was only ever certified *within that family*. `c_dropvv` (22 channels) was the
   first **representation** change. Adding it as an 8th anchor drops ρ **+0.964 → +0.738**, and it is
   the single discordant pair — **yet the gate still reads 17/18 PASS, so the gate does not catch this.**
   **The ratio-feature battery is a representation change too.** Screening it with ATC-F1 repeats this
   error verbatim. Test ratios on the LB, one variable, seed-paired — or re-certify the estimator
   against `c_dropvv` first. Feature-space **deletion is now CLOSED** (cost: 1 submission).

   **Measurement noise, not submissions, is now the binding constraint.** ~78 subs and 19 days remain,
   but seed sd is 0.0191 and **averaging cannot fix it**: at seed rank-corr 0.9511 the variance-reduction
   factor is (1+0.9511·4)/5 = **0.961**, so 5-seed pooling moves sd only to 0.0187. **Effects below
   ~0.02 are unmeasurable on the public slice by any construction we have.** The leader is ≈+0.05 ahead.
   Hunt that lever; stop buying 0.005s.

   **🔑 Read this before anything else — the public-LB leader (`sdv`) posted their approach
   (forum, 14–16 Jul 2026, post-reset) and it reframes the whole project.** They sit at ~0.94 with
   **plain CatBoost** — *"the model isn't the bottleneck"* — so our ≈0.05 gap is **entirely in the
   features**, not the model class and not the ensembling that iterations 18–25 spent themselves on.
   Three actionable specifics: (a) their named lever is **relative/ratio-style features**, which
   *"survive the shift far better than absolute values"* — **we have never tested a cross-band ratio**,
   and `VH − VV` (the dB log cross-pol ratio) is queued and unrun; (b) they tried **regime-mimicking
   validation and it "barely correlated"**, so do not spend days rebuilding CV; (c) **the two scored
   columns are independent** and they optimize each separately (see `RESEARCH_11.md` §1a).
   **Do not read (a) as contradicting "amplitude is the signal."** Our refutation tested
   *within-series temporal rank*, which destroys level. A *cross-band ratio at fixed t* preserves level
   while cancelling per-period multiplicative gain drift. Two different transformations; we conflated
   them for ten iterations.

   iter25 (ran 2026-07-28) closed the band-deletion screen with a rare positive: gate PASSED and
   **`c_dropvv` cleared the pre-committed rule** — 2/2 votes, ATC-F1 **0.8977** against the champion's
   5-seed range **[0.7196, 0.8601]**, i.e. above its *best* draw. Uploading it (paired, seed 42 both
   sides). Committed prediction ≈0.900; **≥0.9075** = confident win, **≤0.8835** = confident loss.

   The Phase-A audit (`tools/shift_audit.py`, ~3 min, zero submissions) that produced it closed one
   lane and opened another:
   - **CLOSED:** the per-band missing-indicator deletion three agents predicted. Indicators alone
     separate masked-train from test at adv-AUC **0.4758** (below chance) and add **+0.0028** over
     values — because `apply_mask` already applies S2 dropout at rates measured off test, so the
     distribution was matched by construction. No submission spent.
   - **OPENED:** the shift lives in the VALUES (adv-AUC **0.8915** masked+left-aligned, vs 0.965–0.976
     for Presto on raw pixels). The 2-D screen (A=shift, T=signal) names **VV** (A 0.5907, T 0.7801 —
     top shift-carrier, dominated by VH's 0.8302) and **blue** (A 0.5344, T 0.5963) as free deletions.
     **The SAR literature independently rejects VV** (wind-sensitive; threshold drifts 2.6 dB/yr vs
     VH's 2.1; VH's histogram is cleanly bimodal — Ottinger 2017/2019, Li 2018). Capacity-REDUCING.
   - **Temper expectations:** max single-band A is 0.59 vs a joint 0.89, so the shift is DISTRIBUTED
     and band deletion cannot collapse it. VV's T is 0.0001 below the median — a knife-edge.
   Paste back the logged `seq input width` per candidate (MUST be 22/23/20, not 24), the gate, the
   three SCREEN lines, and the arch_blend diag rows.

4. **Then: the PHASE-TWO WRITEUP.** The architecture search is finished — iter24 closed
   the last model class, and cross-class blending is closed with n=2. No modelling lane remains that
   is sized to the ~0.010 measurement floor. **35% of the final score for a top-5 finish is the
   reproducibility/novelty review, and that deliverable does not exist yet.** With the 2026-08-16
   deadline this is now, by a wide margin, the highest-expected-value work available. Raw material to
   REUSE (not rewrite): `README.md`, `JOURNEY.md`, `experiments/reproduce_champion.sh`, and the full
   ledger in `experiments/LB_LOG.md`. Lead with what is genuinely novel and already evidenced in-repo:
   the **measured 0.0191 seed variance** and the intellectual honesty of voiding nine of our own
   verdicts with it; the **offline LB-predicting validator** (`tools/offline_validate.py` — ATC-F1
   retro-fit to known-LB anchors, permutation-null gate, seed-noise floor) that let ~14 iterations be
   screened at zero submission cost; the **rank-only metric proof** and the prevalence pin; the
   **pinned-threshold ensemble law** from iter22/24; and the **adversarial-AUC 0.97 shift evidence**
   proving the OOF↔LB gap is irreducible covariate shift, not leakage.
   **Endgame task, do NOT leave to the last day:** designate the two finalists MANUALLY on Zindi. The
   default is the two best *public* scores, which is wrong here — the public max (0.8955, and
   c_meanmin 0.8986) are lucky single-seed draws. See §3 for the pair and the open question.

   **iter18 result — the architecture ensemble is MARGINAL.** Cross-architecture rank-corr
   ρ=0.9395 (≈ the 0.9511 seed baseline): the four tied transformer variants are the same model
   class, so they barely decorrelate and the blend lands at the member mean (variance-only, like
   seed-avg). Lesson: pooling variants can't buy *level*; only a different model class can. So the
   plan is representation first (iter19 pooling), then a decorrelated MiniRocket/CropNet member
   (iter20). Round-09 deep research (Claude + Gemini) is triaged in `gemini_loop/RESPONSE_09.md`;
   the three-way convergence on dispersion pooling is why iter19 leads.

   **iter17 result — the Presto lane is DEAD, for 0 submissions.** All four configs returned
   adversarial AUC **0.965–0.976** on the frozen embeddings (>0.9 ⇒ the encoder *encodes* the
   designed temporal shift rather than normalizing it). ATC-F1 put Presto **0.044–0.059 LB below
   champion**; both configs HOLD. Its OOF (0.967–0.969) is already below champion's 0.975. That
   closes the foundation-model / model-class frontier. **But it was worth every second:** a
   near-perfect train/test separator existing in a general-purpose, label-free representation of the
   raw pixels **independently proves the shift is real and large** — the ~0.975 OOF vs ~0.89 LB gap
   is mostly irreducible covariate shift, and our champion already carries the right response
   (masking views + relative time + cross-view invariance = shift-invariance machinery). Presto lost
   *because* it faithfully re-encodes the raw signal, shift included.

   **Why the grand ensemble now.** The seed-average bought variance reduction but no level (0.8865 ==
   the single-seed mean 0.8859), because seeds are 95.1% rank-correlated. The one remaining cheap
   shot at *level* is to pool across **different architectures**, which may be decorrelated where
   seeds are not. The go/no-go is free — the cross-architecture rank-correlation matrix
   (`tools/arch_blend.py`): ρ ≈ 0.95 ⇒ no gain (behaves like seed-avg); ρ < ~0.90 ⇒ pooling gains
   level with bounded downside. The screen cannot resolve this (ATC-F1 seed sd 0.0576 ⇒ ±0.0094 LB,
   coarser than any ensemble gain), so the matrix is the instrument.

   **iter16 result: seed-averaging scored 0.8865 against a predicted ~0.886.** The variance model
   is confirmed to 0.0006. But there was **no ensemble gain** — we bought variance reduction, not
   level, because the seeds are 95.1% rank-correlated so only ~5% of the error is independent.
   **Finalist decision (revised):** designate **`champion_seedavg5` + `seq_a_xview`**, not
   xview + NoPE. Shrunk true-quality estimates are 0.8865 vs 0.8911 — a tie inside our resolution
   floor — but the two carry *different risk profiles* (consensus vs point estimate), whereas
   xview and NoPE differ by 0.0038 and are two draws of the same thing.

   **iter15 settled the measurement question. The screen's resolution is ≈0.010–0.013 LB**, derived
   two independent ways that agree: ATC-F1's seed sd (0.0576) converts to ±0.0094 LB via the anchor
   fit `LB = 0.1628·ATCF1 + 0.7714`, and the directly measured champion seed spread is 0.0191
   (sd ≈0.013). Seed rank-correlation is **0.9511** — ~95% of our ordering is reproducible, ~5% is
   RNG, and that 5% moves the LB by 0.019.

   **🚨 THE STANDING RULE FROM HERE: stop running small A/B probes.** Only effects **> ~0.010 LB**
   are measurable *in principle* with our budget. The only two effects ever measured above that
   floor are the GBDT→Transformer swap (+0.05) and detrend (−0.05) — **both model-class changes**.
   Every architectural tweak, loss term, pooling variant, positional reframe and regularization knob
   sits below it. The two fundable directions are **(a) variance reduction** (seed averaging) and
   **(b) a model-class change** = the **Presto lane** (`RESEARCH_07.md` §5e).
   In iter15 the new seed-noise guard downgraded two **2/2-vote** candidates to HOLD because their
   margins were inside estimator seed noise — it caught exactly the iter14 mistake.

   **Superseded — kept for the record:**
   iter15 measures the estimator seed-floor across five champion seeds, screens `c_meanmin` (the
   *lower* tail — the pond literature's actual detector, which we never tested), and pools the five
   champion seeds into one rank-averaged submission via the new `tools/seed_average.py`.
   **Expect the seed-averaged public score to be BELOW 0.8955, and want it to be.** 0.8955 is the
   better of two draws from a distribution with sd ~0.013 and is probably an upward fluctuation.
   Judge the pooled submission against the **mean** of our single-seed scores (~0.886), not the max.
   Chasing the public number under this much variance is what produces a shake-up.

   **Superseded — kept for the record:**
   **(a) UPLOAD `submissions/submission_c_dropout3.csv` to Zindi** and paste the LB. The file was
   already written by the iter13 run — no rerun needed. This is our **first screen-approved
   submission** and the first LB spend since iter10.
   **(b) `Run all` → iter14** (regularization sweep, **0 submissions**) and paste back the three
   tables. Independent of (a); do not serialize them.

   **iter13: the screen FIRED for the first time.** `c_dropout3` (dropout 0.2 → **0.3**) cleared
   both certified estimators — **ATC-F1 +0.0165, DIS +0.0029, 2/2 → SUBMIT**. It is *exactly
   parameter-neutral*, the most on-thesis knob in the repo under our own "less fit transfers better"
   law — and it had **never been touched in twelve iterations**. We spent those twelve on
   architecture while the plainest regularization knob sat at its default.
   ⚠️ The DIS margin is **tiny** (+0.0029): a 2/2 by the rule, not a resounding one. Estimator
   deltas are **not** on the LB scale.
   Also from iter13: **`c_compact` genuinely ran and FAILED** (−0.0053/−0.0252) — the 24→14 deletion
   two research agents derived *independently* does not work, so independent agent convergence is
   **not** evidence. **`c_meanmax_l0` flipped to −0.0847** vs +0.0838 with λ=1, so the upper-tail
   statistic *depends on* cross-view invariance rather than competing with it. **`c_k3` ≈ 0** — K=2
   confirmed optimal from three points instead of two.

   **iter12 (first screen): all five candidates HELD, 0 submissions spent.** The rule earned its
   keep — ATC-F1 liked `mean_max` (+0.0838) but DIS disagreed (−0.0301), which under the old blind
   regime would have been a wasted submission. Three findings:
   **(a) Amplitude is the PRIMARY SIGNAL** — rank-replacement collapsed OOF 0.9753 → 0.857/0.865 on
   both seeds (ATC-F1 −0.1703). The pond discriminator is "persistently LOW backscatter", an
   absolute level. The rank/ordinal family is now closed *with evidence*. (This does **not**
   resurrect the old "amplitude is toxic" law, which remains unevidenced — *removing* amplitude
   being catastrophic is a different claim.)
   **(b) `c_compact` was never tested** — a config-path bug (`seq.compact_missing` vs
   `seq.channels.compact_missing`) meant the flag never reached the model; the run was bit-identical
   to the champion and the screen scored the no-op as a 0.0000 tie. Fixed; the pipeline now logs the
   **actual** input width. Re-tested in iter13.
   **(c) DIV failed at ρ = −0.857** (2/15) — *lower* fold-diversity goes with *higher* LB, the
   opposite of hypothesis H1. H1 is not supported.

   **iter11 PASSED (2026-07-22) — the measurement constraint is broken.** Retro-fit on the 7 anchors:
   **ATC-F1 15/15 concordant, ρ=+0.964** and **DIS 5/5, ρ=+1.000 (n=4)** both cleared, while the
   *original* pre-committed estimator **ATC FAILED at ρ=−0.429** and the naive control MARG failed at
   −0.321. The pre-repair iter11 would have failed outright; the two that cleared are exactly what
   round 07 added. And the two failures **confirm the rank-only proof** — both measure confidence,
   which the LB cannot see, so both anti-correlate.
   **→ SCENARIO A is live:** screen offline, submit only where **≥2 cleared estimators** beat the
   champion. Treat DIS as a second vote only (n=4, exact null p≈0.042); ATC-F1 is the solid one.
4. **iter11 was REPAIRED on 2026-07-22 before its first run** (`RESEARCH_07.md`). Three bugs would
   have wasted the whole spend: the DIS estimator was **unscoreable** (only the champion got a second
   seed, but ≥3 variants are needed for a rank correlation — now fixed by seeding detrend/k4/reltime);
   the gate could raise `KeyError`; and a loose glob admitted `_smoke` files as seed replicates.
   The **gate itself was replaced**: exact permutation nulls at n=7 show ρ>0.7 passes by chance at
   p=0.044 (~9% familywise over 3 estimators) **and** can reject a *perfect* validator (ρ=0.643),
   because 4 of the 7 anchors sit inside the noise band so their measured order is itself noise. It
   now scores concordance on pairs with |ΔLB| > 0.01 (exact null p=0.0048), with ρ reported descriptively.
5. **After iter11, branch on the gate** — the two-scenario decision tree is in `RESEARCH_07.md` §6.
   Zero-submission work that runs either way: the partial-S2 missingness audit, iterated adversarial
   channel attribution, posting the two-column legality question, and building the Presto/TabPFN lane.
6. **RULE FACTS (verified 2026-07-22):** (a) we **designate 2 finalist submissions** (default = 2 best
   public) → the hedge is usable; designate manually before close. (b) **100-total cap**: ≈20 used,
   ≈80 left. (c) final score = **65% LB + 35% rubric** (top-5, reproducibility/innovation) → prep a
   reproduction README at endgame; the ITU Cropland precedent (same organiser family, 40% report
   weight) shows this channel is real and low-variance. (d) **Pretrained models are LEGAL** — see §6.

---

## 2. How to run on ANY cloud account (the portable loop)

The design: **code lives in GitHub, data + secret live in the account.** Each iteration the
coding agent edits `experiments/run_current.sh` + `config/config.yaml` and pushes; you pull and
**Run all**; the notebook itself never changes. Zindi submission stays manual (5/day).

**You always need three things on a new account:**
1. **The code** — via `git pull` of the private repo (needs a GitHub token, below).
2. **The data** — `Train.csv`, `Test.csv`, `SampleSubmission.csv` (from Zindi; keep PRIVATE —
   rules allow only supplied data).
3. **A GitHub PAT** — fine-grained token, repo `geoai-aquaculture`, **Contents: Read-only**,
   short expiry, stored as a secret named **`GH_PAT`**. (Read-only + single-repo + short expiry
   because the token gets embedded in `.git/config` on the VM.)

### Google Colab
- Drive: put the 3 CSVs in `MyDrive/geoai-data/` (top-level, exact name — the notebook reads it).
- Colab Secret `GH_PAT` (🔑 sidebar, enable notebook access).
- **Runtime ▸ Change runtime type ▸ T4 GPU.**
- Open `colab_run.ipynb` → **Runtime ▸ Run all.** Cell 1 pulls; Cell 4 runs `run_current.sh`;
  Cell 5 downloads `submission_*.csv`.
- Full details: `RUN_ON_CLOUD.md`.

### Kaggle Notebooks
- Upload the 3 CSVs as a **Private Dataset** (e.g. `geoai-aqua-data`); attach via **Add Input**.
- **Add-ons ▸ Secrets** → add `GH_PAT`; **Internet: On**; **Accelerator: GPU T4 x2**.
- Paste the cells from `RUN_ON_KAGGLE.md` (git pull → deps → data → `bash run_current.sh` → FileLink).
- Kaggle gives ~30 GPU-hrs/week; a full seq run is a few minutes.

### The loop each iteration (all accounts)
`git pull` → **Run all / `bash experiments/run_current.sh`** → download the newest
`submission_*.csv` → upload on Zindi → **paste the public LB score back to the agent** (it goes
into `experiments/LB_LOG.md`, the reward signal) → agent stages the next experiment + pushes.

---

## 3. Current status

- **Champion model:** from-scratch temporal Transformer (attention over observed months via
  `src_key_padding_mask`, per-band missing-indicator channels, masked-mean-pool), **K=2**
  masking-augmented training views, **relative-time reframing ON**, **cross-view invariance objective
  (λ=1.0)**, operating point held at **realized pos-rate 0.649**.
- **Champion config** (`config/config.yaml`, LIVE — this is the reverted, exact 0.8955 state):
  `seq.K: 2`, `seq.relative_time: true`, `seq.pos_encoding: learned`, `seq.consistency_lambda: 1.0`,
  all `seq.channels.*: false`, `seq.tta.enable: false`, `calibration.prevalence_target: 0.649`.
- **Best public LB: 0.8955** (0.8780 → +0.0128 relative-time → 0.8908 → +0.0047 cross-view invariance).

### 🔴 THE BOARD RESET 2026-07-28 — every row below is INELIGIBLE

All six artifacts were produced by the **rules-violating** prevalence-pin path (threshold tuning +
a non-probability RAUC column). **None of them can be designated as a finalist.** They are retained
only as the historical record and as the anchors the offline validator was retro-fitted to.

**The legal board:**

| legal artifact | public LB | status |
|---|---|---|
| **`champion_archblend4`** (legal) | **0.899643** | 🏆 **FINALIST #1** — best public score ever recorded here, and eligible |
| `seq_a_xview` (legal, seed 42) | 0.889686 | ✅ **FINALIST #2** (default; single seed, so higher variance) |
| `champion_seedavg5` (legal) | — | built, not scored — pos-rate 0.5835; largely redundant with archblend4 |

**⚠️ Several ensemble rules were derived UNDER THE PIN and are now unverified**, including "gate
members on level gap, not correlation" and the rank-correlation go/no-go (which printed SKIP at
ρ=0.9524 and was wrong). Do not apply them without re-deriving under a literal 0.5 cut.

### Historical artifact board (ILLEGAL — reference only)

| artifact | public LB | what it is | seed luck? |
|---|---|---|---|
| `c_meanmin` | 0.898566 | min-pool variant, single seed | ⚠️ yes |
| `seq_a_xview` | 0.8955 | champion, seed 42 | ⚠️ **known lucky draw** |
| **`champion_archblend4`** | **0.894643** | 4 transformers, seed+arch pooled | ✅ **none — lowest variance** |
| `champion_seedavg5` | 0.886530 | 5 champion seeds pooled | ✅ none |
| `champion_rocketblend5` | 0.885661 | ⅕ ROCKET (ρ0.87 member) | ✅ none |
| `champion_gbdtblend5` | 0.879123 | ⅕ GBDT | ✅ none |

**Finalist #1 = `champion_archblend4`, settled.** Highest reliable public *and* the lowest-variance
construction; nothing from iterations 18–24 challenged it.

**Finalist #2 is the one open decision (due before 2026-08-16).** The hedge thesis that favoured
`rocketblend5` — "buy a genuinely independent component to insure against a shared private-slice
failure of the transformer cluster" — is **weakened by iter24**, which showed foreign members cost
real level (−0.009 and −0.016) rather than trading level for variance. Candidates:
- `champion_rocketblend5` (0.8857) — the only true diversity hedge; costs ≈−0.009 expected.
- `champion_seedavg5` (0.8865) — no level cost, but largely redundant with archblend4 (a subset of
  its pooling).
- `c_meanmin` (0.8986) — highest public after the lucky champion, but ρ=0.9928 to xview → a rank-twin
  that hedges **nothing**; its public edge is itself a single-seed draw.
Decide near the deadline, not now; the reasoning above is the input.
- **Target RECALIBRATED 2026-07-22.** Forum scores of 0.953 / "0.98+" were posted **before the
  25 Jun data reset**, i.e. earned on the **leaked** data — ignore them. The live competitive band is
  the "90s club" (thread dated 14 Jul, post-reset): roughly **0.90–0.95**. We sit just below the bar,
  **not 0.033 behind it.** The gap is smaller and more winnable than earlier revisions of this file claimed.
- **Loop state: iter11 STAGED (offline validator, 0 submissions), awaiting a Colab run.** Both
  structural lanes measured closed — positional (dnorm −0.006, NoPE +0.001) and objective (λ=3 −0.003,
  so λ=1 is an interior optimum). Diverse finalist (NoPE 0.8917) provisionally locked — **but** if any
  lane produces a model within ~0.01 of champion with different errors, it replaces NoPE as finalist #2
  (NoPE is a near-clone and buys little private-LB variance hedge).
- **Research round 07 done** → `gemini_loop/RESEARCH_07.md`, which carries the rule correction, the
  rank-only proof, the two-scenario decision tree, and the fixes applied to iter11.

### Confirmed data facts (verified on the live Zindi site 2026-07-22)

- **TRAIN: 1,821 rows × 12 FULL months**, ~40% positive. **TEST: 1,030 rows × only 4/5/6 CONSECUTIVE
  months**, rest `-9999`; test positive rate believed ~0.65 (which is exactly our tuned prevalence).
- Bands: S1 **VH/VV** always present when a month is observed; **10 S2 optical bands may be missing
  per-band due to cloud**. **lat/lon REMOVED** — these are isolated patches with no spatial context.
- **The shift is TEMPORAL BY DESIGN:** train and test are different time periods; conditions "change
  across seasons and years." Public LB = **30%** of test (~309 rows), private = **70%** (~721 rows).
- **25 Jun 2026 data reset** after a leak (new train = old train + old test *with labels*; new test
  issued; lat/lon stripped). **Our first submission was 9 Jul**, so all 7 LB anchors post-date the
  reset and are mutually comparable — the iter11 retro-fit is valid.
- **Open question, unanswered on the forum:** does "month 01" in train mean the same calendar month
  as "month 01" in test? If not, several seasonal ideas are unworkable and relative-time's win is
  even better explained.

---

## 4. Full experiment ledger (every run, output, LB, verdict)

Metric = **0.6·F1 + 0.4·ROC-AUC**. "OOF" = local cross-val combined (⚠️ **proven BLIND / often
anti-correlated** — never used for selection). "LB" = Zindi public (~309 rows) = ground truth.

### Phase 1 — GBDT ensemble + prior correction (pre-transformer)
| Change | Operating point | LB | Verdict |
|---|---|---|---|
| GBDT ensemble, inherited train prior | pos 0.40 | 0.7140 | baseline |
| + base-rate/prior correction | pos 0.50 | 0.7561 | ✅ +0.042 |
| + prior correction (swept) | pos ~0.65 | **0.8260** | ✅ GBDT peak |
| prior 0.70 / 0.75 / 0.80 | — | 0.8216 / 0.8166 / 0.8037 | prior lever saturated |
| WIF + EVI features | pos 0.50 | 0.7509 | ❌ reverted (train AUC 0.83, no transfer) |

### Phase 2 — from-scratch temporal Transformer (the breakthrough)
| Change | Realized pos-rate | LB | Verdict |
|---|---|---|---|
| Temporal Transformer | 0.593 | 0.8776 | ✅ |
| Temporal Transformer | 0.627 | 0.8732 | |
| **Temporal Transformer** | **0.649** | **0.8780** | 👑 **CHAMPION** |
| Temporal Transformer | 0.672 | 0.8733 | |

→ +0.05 over the GBDT peak **despite identical OOF** — the finding that defines this competition.

### Phase 3 — improvement attempts, 2026-07-20 (all LOST; champion held)
| # | Experiment (only variable vs champion) | OOF | LB | Verdict |
|---|---|---|---|---|
| 2 | + GBDT rank-average blend (0.7 seq / 0.3 GBDT, ρ=0.85) | 0.952 | 0.8705 | ❌ −0.0075 |
| 3 | + `per_cell_detrend` input channels | 0.979 | **0.8266** | ❌ −0.0514 |
| 4 | seq masking views K=2 → K=4 | **0.984** | 0.8665 | ❌ −0.0115 |

Also verified this session: **Step-1 `prevalence_target 0.649` mechanism works** (holds any run
at the exact champion pos-rate → clean isolation); the Colab env **reproduces faithfully** (blend
landed exactly between its components).

### Phase 4 — capacity-CONSTRAINT direction (round-04 research, in progress)
Round-04 Deep Research triaged in `gemini_loop/RESPONSE_04.md`. Rejected proven dead-ends
(Saerens-EM prior; Zou-threshold/EVI index projection). Shifting from capacity *expansion* to
*constraint*: test capacity-neutral, structural changes one at a time.
| # | Experiment (only variable vs champion) | OOF | LB | Verdict |
|---|---|---|---|---|
| 5 | relative-time reframing (`seq.relative_time`: left-align window to t_rel=0) | 0.9811 | **0.8908** | ✅ **NEW CHAMPION** (+0.0128; first win, capacity-neutral structural reframe) |
| 6 | MC temporal-dropout TTA on champion (`seq.tta`: mask 1-2 active months, 8 views, soft-vote) | — | 0.8885 | ❌ −0.0023 (within noise, did not beat champion; reverted) |
| 7 | duration-normalized fractional positions (`seq.pos_encoding: dnorm`; share [0,1] frame across L) | 0.9789 | 0.8844 | ❌ −0.0064 (length already matched → no shift to remove; reverted) |
| 8 | NoPE / permutation-invariant SET encoder (`seq.pos_encoding: none`; drop positional embedding) | 0.9789 | 0.8917 | ➖ TIE +0.0009 (position is neutral; LOCKED as diverse finalist) |
| 9 | cross-view invariance objective (`seq.consistency_lambda: 1.0`; penalize logit var across K views) | 0.9753 | **0.8955** | ✅ **NEW BEST** +0.0047 (reduced overconfidence; edge of noise) |
| 10 | cross-view invariance strength probe (`consistency_lambda: 3.0`) | 0.9727 | 0.8921 | ❌ −0.0034 (λ=1.0 is an interior optimum; reverted; objective lane CLOSED) |
| — | research round 06 → `RESPONSE_06.md` (both reports triaged) | | | ✅ done |
| 11 | **offline LB-predicting validator** (ATC · seed-disagreement · control) retro-fit to 7 known-LB anchors | | **0 subs** | **staged** |
| 12 | queued: dispersion pooling `mean ⊕ std` (Ottinger permanence/low-std physics) | | | not yet run |
| 13 | queued: focal loss γ=3 / FLSD-53, keep λ=1, refit δ | | | not yet run |
| — | gated on iter11 PASS: fold-ensemble deletion → group-DRO → VH−VV → AUC surrogate | | | gated |
| 18–23 | ensemble/model-class lanes — full results in `experiments/LB_LOG.md` | | 0.8946 / 0.8986 / 0.8857 | archblend4 = leading finalist; ROCKET decorrelated but −0.040 weak |
| 24 | **GBDT as decorrelated member** (ρ 0.8734, ATC-F1 −0.0110 LB) → `champion_gbdtblend5` | 0.98319 | **0.879123** | ❌ **−0.0155 vs archblend4, PAIRED (4/5 shared members) → ≥2.5σ, significant.** Stronger member cost ~2× more than ROCKET's → cross-class blending CLOSED, n=2 |
| — | endgame: designate finalists MANUALLY on Zindi (default = 2 best public, which is NOT what we want) · Phase-Two reproducibility/novelty writeup | | | **next, after iter24's LB** |

**The design compass (refined through iter7):** it is not "never change the model" — it is *added
capacity* (extra model/channels/augmentation) and *robustness moves* (TTA) that don't transfer. A
capacity-neutral structural reframe helps **only when it deletes a channel that is actually SHIFTED
train-vs-test.** Relative-time removed window START (calendar month = shifted) → +0.0128 WON.
Duration-norm removed window LENGTH (matched by augmentation = NOT shifted) → −0.0064 LOST. Before
proposing any reframe, ask first: *is this channel actually shifted?* NoPE (iter8) removes positional
identity entirely — a bigger, two-tailed bet, and the diverse finalist regardless of its public score.

---

## 5. Progress & declines — the narrative

**What moved us UP (0.714 → 0.891, +0.177 total):**
1. **Prior/base-rate correction** (+0.11 to the GBDT peak 0.826): the test set is far more
   positive (~65%) than train (~40%). Now saturated.
2. **GBDT → from-scratch Transformer** (+0.05 to 0.878): attention over *only observed months*
   transfers across the designed domain shift where flattened GBDT aggregates over-fit the source.
3. **Relative-time reframing** (+0.013 to 0.891, 2026-07-21): left-align each observed window to
   t_rel=0 so positional embeddings encode relative step, not calendar month — kills the calendar-
   specific spectral memorization the covariate shift punishes. Capacity-neutral; broke a 10-day plateau.
4. **Cross-view invariance objective** (+0.005 to 0.8955, 2026-07-21): penalize logit variance across
   a row's K=2 masked views (L=BCE+λ·Var). Reduced the model's overconfidence (its diagnosed weakness)
   and improved transfer. Objective-level, capacity-neutral. **iter10 then showed λ=1.0 is an INTERIOR
   OPTIMUM** — λ=3.0 de-saturated further (t\* 0.4450→0.3400, delta 1.30→0.725) with `oof_auc` intact
   at 0.9896, yet scored 0.8921. So the mechanism is real but bounded: *some* de-saturation transfers,
   more does not, and the failure is not ranker collapse. Lane closed at λ=1.0.

**What DECLINED (Phase 3 — everything we tried after 0.878):**
- Blend −0.0075, detrend −0.0514, K=4 −0.0115. Pattern: **every attempt that ADDED something
  (a model, input channels, more augmentation) lost.** The detrend result specifically
  **disproves** the "remove per-series level → better transfer" thesis for this model.

**Why we paused (then resumed):** public LB ≈309 rows → **~±0.01 noise**. Single-submission A/B
**cannot resolve** small (+0.005) gains; only large effects or breakages are detectable. So we
stopped guessing toggles inside the noise band and ran a research round. The output — relative-time
reframing — was a *large* effect (+0.013, above noise), which is exactly the class of change worth a
submission. Lesson: don't probe inside the noise; hunt changes big enough to clear it.

---

## 6. Lessons & DEAD ENDS (do not retry)

**Hard lessons (2026-07-20, refined 2026-07-21):**
1. **Added *capacity* hurts; capacity-neutral *structure* helps — but ONLY if it deletes a SHIFTED
   channel.** Extra model / channels / augmentation all lost (−0.008 to −0.051); robustness moves
   (TTA) land within noise. Relative-time reframing (remove window START = calendar month, which is
   shifted train-vs-test) WON +0.013. Duration-norm (remove window LENGTH, which augmentation already
   distribution-matches → NOT shifted) LOST −0.006. Compass: reframe the coordinate/inductive-bias to
   delete a channel that is *actually shifted*, never its capacity, never a matched/informative channel.
2. **OOF is anti-correlated**, not merely blind — highest-OOF run (K=4, 0.984) = 2nd-worst LB;
   the 0.8908 winner's OOF (0.9811) was *lower* than the old champion's (0.9827).
3. **Measurement resolution is the binding constraint** — 309-row public LB, ±0.01 noise. Only
   probe changes plausibly large enough to clear it; don't A/B inside the noise band.
4. **🚨 CROSS-MODEL-CLASS BLENDING DOES NOT WORK HERE — closed with n=2 (iter22, iter24).** Both
   foreign members we could build (random-kernel ROCKET, mature tree ensemble) reached ρ≈0.85
   decorrelation, and **both dragged the blend below the pure-transformer pool**: rocketblend5 −0.0090,
   gbdtblend5 −0.0155 vs `archblend4`. The iter22 repair ("decorrelated is necessary but not
   sufficient — the member must also be competent") is itself **REFUTED**: the GBDT *was* competent
   (−0.011 vs ROCKET's −0.040, equally decorrelated) and its blend cost **nearly twice as much**. The
   correct, stronger statement:
   > Under a rank-only metric with a **pinned threshold**, a decorrelated member's reordering of rows
   > *near the cut* costs more level than its independence buys back. Only near-rank-twins (ρ>0.93,
   > i.e. the same model class) pool without loss — and those buy variance reduction, never level.
   Corollary: "blend level ≈ weighted mean of member levels" is **false** under rank-averaging; do not
   size a member's weight from that arithmetic (it is what mispredicted iter24 by 0.013).
5. **Do not extend a rule past its derivation.** iter18's *"for ensemble calls read the correlation
   matrix, not the screen"* was derived for pooling **within** the transformer class, where all
   members are equally competent and decorrelation is the only free variable. iter24 applied it to a
   blend containing a **weaker foreign member** and overrode a correct `votes=1/2 → HOLD`. Where member
   competence is the deciding variable, the screen is the right instrument. Related: **ATC-F1's
   MAGNITUDE is unreliable (~3× overstated, iter19)** — trust its sign only. A gate phrased as
   "ATC-F1 within 0.02 LB" silently trusts the magnitude; corrected by 3×, iter24's −0.011 reading is
   ≈−0.033 ≈ ROCKET's −0.040, which predicts the observed drag exactly.
6. **The 0.019 seed lottery is a TRANSFORMER property, not a task property** (iter24). Seed
   rank-correlation: Transformer **0.9511** vs GBDT **0.9795** — the GBDT's seed-independent error
   component is ~2.4× smaller. Our whole "seed noise voids the ledger" finding is a statement about
   the from-scratch Transformer under this data regime, not about the dataset. (n=1 seed pair vs 10,
   so directional.) This survives iter24's failure — it is a property of the *member*, measured
   independently of the blend — and is worth stating in the Phase-Two writeup.

**Do not re-propose (tried & failed, or rule-illegal):** **GBDT+seq blending — now CLOSED PROPERLY
(iter24, LB 0.879123, −0.0155 paired), superseding the old "badly constructed, not proof" caveat;
the iter2 blend WAS badly constructed, and the clean re-measurement with a pinned prevalence, a
two-level rank-blend and seed-pooled members still lost** · cross-class blending generally (ROCKET
iter22 −0.0090, GBDT iter24 −0.0155) · `per_cell_detrend` and the
additive-channel family (`deltas`/`indices`/`rank`, now low-prior) · K>2 augmentation · BBSE/EM
prior estimation · WIF / fixed-threshold water features · temperature scaling ·
importance-weighting / DANN for TRAINING (ESS collapse @ adversarial AUC 0.99) · OOF meta-stacking ·
group-KFold / "it's leakage" (the gap is designed covariate shift, proven leak-free) · VH−VV as a
replacement channel *(removed 2026-07-22: `(VH,VV)→(VH,VH−VV)` is an invertible linear map feeding a
linear layer, so the model can already represent it)*.

> **⚠️ TabPFN and pretrained/foundation models are NO LONGER on this list.** They were listed as
> "rule-banned" — that was **wrong**. See the corrected constraints below.

**Constraints (never violate) — CORRECTED 2026-07-22 from the live rules page:**
- Only the supplied datasets. **No external DATA.**
- **PRETRAINED MODELS ARE ALLOWED** — verbatim: *"You may use pretrained models as long as they are
  openly available to everyone."* Every doc before 2026-07-22 wrongly said "train from scratch, no
  pretrained models," and TabPFN was rejected on that false basis. TabPFN v2, Presto, Prithvi, Clay,
  SatMAE are all **legal**. (Caveat to verify: Zindi also says "custom packages in your submission
  notebook will not be accepted" — confirm what that means operationally before relying on a pip dep.)
- AutoML banned. Open-source, seeded, reproducible only.
- `TargetF1` scored at hard 0.5 (prior/prevalence shift allowed, threshold tuning not).
- **100 submissions total**, ≤5/day. Final score = **65% LB + 35% code review** of the top 5.

**⚠️ THE METRIC IS RANK-ONLY.** After the prevalence pin the predicted-positive count is fixed at
P̂ = 0.649·n, so F1 = 2·TP/(P̂+P) is monotone in precision@k, and AUC is rank-only by definition.
**The LB sees only how the model ORDERS the 1030 test rows — it is blind to calibration.** This
reframes the "de-saturation" story told about iter9/iter10 (cross-view invariance must have won by
changing the *ranking*, not by reducing overconfidence) and demotes any change whose mechanism is
purely calibrative (focal loss, temperature). It also means our prevalence instrument is saturated:
Lipton et al. show the F1-optimal cut is F1\*/2, and our `t_star = 0.445` ≈ 0.89/2 already.

**Measurement protocol (quantified 2026-07-22).** Combined-metric SE ≈ **0.012** on the 309-row
public LB (≈0.008 on the 721-row private). But a **paired** delta between two ρ≈0.9 variants of our
own model has SE ≈ **0.006**. So: unpaired/cross-team needs ≥0.012; our own A/B is *confident* at
≥0.012 and *suggestive* at ≥0.006; below 0.006 is unmeasurable. Expected |public − private| drift
for a single model ≈ 0.012.

---

## 7. Key files map

| File | Role |
|---|---|
| **`PROJECT_STATE.md`** | ← this file. Master state, portable across accounts. Updated every session. |
| `experiments/LB_LOG.md` | Reward ledger — paste each submission's Zindi LB here. |
| `gemini_loop/AGENT_BRIEF.md` | Standing directive for the coding agent (rules, queue, meta-lessons). |
| `gemini_loop/UPDATE_06.md` | **Current** research brief → paste into Claude Fable Deep Research (05/04 = prior rounds). |
| `JOURNEY.md` / `JOURNEY.docx` | Plain-English story of the whole project (regenerate the docx via `tools/make_journey_docx.py`). |
| `experiments/run_current.sh` | The one experiment the notebook runs each iteration (agent edits + pushes). |
| `config/config.yaml` | Single source of truth for all pipeline settings. |
| `colab_run.ipynb` / `RUN_ON_KAGGLE.md` | The pull-run loop for Colab / Kaggle. |
| `run_pipeline.py` | End-to-end: CSVs → CV → calibration → `submission.csv`. |
| `src/seq_model.py` | The champion Transformer. `src/calibration.py` | Fixed-0.5 + prevalence lever. |

---

## 8. Update protocol (keep this file current)

At the end of every session (or whenever an LB score comes in), the coding agent updates:
- **§3 Current status** (champion, best LB, loop state, next action) and the header "Last updated".
- **§4 ledger** (new rows) and **§5 narrative** (if progress/decline changed).
- **§6 lessons** (if a result adds a dead-end or overturns one).
This file is the thing you carry between accounts — it must always reflect reality. If it and a
supporting file ever disagree, trust the most recent LB score in `experiments/LB_LOG.md`.
