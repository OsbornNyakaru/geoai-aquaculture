# RESPONSE_19_CHATGPT — triage of the ChatGPT Deep Research report (REJECTED in full)

Source: `C:\Users\ADMIN\Downloads\deep-research-report.md`, produced against the **UPDATE_18**
brief. Triaged 2026-08-11, at iter43-pending. **Verdict: reject in full.** One idea in it was
unspent (AWEI); it is adjudicated below and closed on measured evidence, not on style.

This file exists so iter44 does not re-litigate any of it.

---

## 1. The report is not grounded in this competition

It is not a weak report; it is a report about a different problem. Enumerated against verified
facts:

| Report claim | Verified reality | Source |
|---|---|---|
| "public leaderboard compromised by a test-set ordering leak"; "top scores near 1.0" | Leader is **0.9301**. No leak exists; no community post says otherwise. | Zindi LB |
| "we emphasize cross-validation over public scores" | **Exactly inverted.** Local OOF ~0.975 is ANTI-correlated with LB ~0.90. Selecting on OOF is the single most expensive error available to us. | `LB_LOG.md` passim; `README` |
| U-Net / ResNet-50 encoders / segmentation masks / "morphological filters" / "remove predicted islands smaller than some area" | Data is **1821 tabular rows x 12 months x 12 bands** of point time series. No imagery, no pixels, no spatial extent. Nothing to segment; nothing to morphologically filter. | `src/data.py` |
| Table 1: eight experiments, logistic reg 0.60 -> U-Net 0.99 | **Fabricated.** Not one row corresponds to any of the 43 logged iterations. | `experiments/LB_LOG.md` |
| "ponds are only ~15% of samples" | Train prior is **0.4023**. | `PROJECT_STATE.md` |
| "No experiment used early fusion of Sentinel-1 (SAR) data" | SAR **is** the champion. The permanence channel is `1[VH_dB < -21]`. | iter31/32, `src/seq_model.py` |
| "post-hoc threshold tuning could be explored ... it informs our private score tuning" | **RULES VIOLATION.** Hard 0.5 cut, `calibration.compliance_mode: legal`. Acting on this sentence would disqualify the entry. | competition rules; `reproduce_champion.sh` gate |
| Metric = 60% F1 + 40% ROC-AUC | Correct. One of the few. | rules |

Citations offered without URLs and unverified from here: Greenstreet et al. 2023 (percentile
stacking, F1~0.95), Liang et al. 2024 (MAFU-Net, DIAS index, F1 90.7%), Ferriby et al. 2021,
KGDNet. Given the fabrication rate in the surrounding text, treat all four as unconfirmed.

## 2. The one unspent idea: AWEI — CLOSED, do not test

Report recommendation #2 is "add NDWI, MNDWI, AWEI, and the DIAS index."

**Already always-on** in the seq encoder (`src/seq_model.py:203-219`, `_index_channels`):
NDWI, MNDWI, NDVI, VV-VH, SDWI. Recommendation #1 (percentile / temporal stacking) is likewise
already what the 24-channel encoder is.

**Genuinely absent from the seq path:** `awei_nsh` and `awei_sh`. They exist only in
`src/features.py:129-130`, the superseded GBDT lane. So the report did surface one untested
channel. It is nevertheless closed, on measured evidence:

    AWEI_nsh = 4*(G - SWIR1) - (0.25*NIR + 2.75*SWIR2)
    AWEI_sh  = BLUE + 2.5*G - 1.5*(NIR + SWIR1) - 0.25*SWIR2

Both are **pure affine functions of bands the encoder already receives**. The encoder's first
layer is `self.proj = nn.Linear(in_dim, d)` (`src/seq_model.py:378`). An affine channel is
exactly spanned by that projection: it adds **zero representational capacity** and adds one more
column of parameters to overfit an adv-AUC-0.89 covariate shift with.

We have measured this exact case. **iter31: `VH-VV` cross-pol — also affine-spanned — added as a
channel scored -0.0228**, the largest single-feature loss in the ledger
(`LB_LOG.md:535-536`; row 31 line 362). That is also precisely why NDWI / MNDWI / NDVI / SDWI DO
earn their slots: ratios and logs are **non**-affine, so they are not spanned.

Conclusion: adding AWEI to the seq stack is a rerun of a known -0.023. **Do not spend a slot.**

"DIAS index" could not be verified and its band formula is unknown. If it ever surfaces with a
definition: it is worth testing **iff** it is non-affine in {blue, green, red, re1, re2, re3,
nir, nira, swir1, swir2, VH, VV}. If it is affine, the same closure applies without a run.

## 3. Process lesson

The report was generated from UPDATE_18 yet invented a leak, a data modality, and an experiment
history. **Screening rule for the Round-19 Claude and Gemini reports:** check the returned
experiment history against `experiments/LB_LOG.md` first. If a report's account of what we have
already run does not match the ledger, discard the whole report rather than mining it for
fragments — an ungrounded generator's "novel idea" is not evidence, and the cost of chasing one
at iter44 (our last experimental slot, deadline 2026-08-16) is the competition.

## 4. Net effect on the plan

**None.** iter43's three uploads proceed unchanged; iter44 remains reserved for the Round-19
Claude/Gemini returns against `gemini_loop/UPDATE_19.md`, which is correctly scoped (95.8% of the
remaining gap to the leader is the F1 term; the live clue is ARM T's high-F1 members).
