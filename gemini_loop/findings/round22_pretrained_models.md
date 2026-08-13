# Round 22 — Pretrained models vs. our input shape (REFUTATION ATTEMPT)

## STATUS: COMPLETE

## HEADLINE

**The target claim is FALSE on both halves, but the practical conclusion barely moves.**

1. *"No pretrained geospatial FM fits our input shape"* — **REFUTED at code level.** Presto and
   Galileo both do. Our 12 bands map 1:1 onto Presto's S1+S2 slots (VH,VV,blue,green,nir,nira,
   re1,re2,re3,red,swir1,swir2 -> VV,VH,B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12). Missing modalities,
   12 timesteps, variable per-row masking, constant month and constant lat/lon are all
   OFFICIALLY SUPPORTED, three of them with published evidence.
2. *"Presto is dead"* — **the kill does not stand.** iter17 killed it with (a) adversarial AUC,
   (b) ATC-F1, (c) OOF. Round 18 RETIRED adv-AUC ("BACKWARDS, Spearman +0.68/+1.00"); iter25/26
   proved ATC-F1 invalid OUT-OF-FAMILY, and frozen-Presto embeddings are the most out-of-family
   thing this project ever screened; OOF is blind by our own standing rule. **Presto was never
   submitted — it has never been measured.**
3. **BUT:** the correct reading is *"the instruments were bad"*, not *"Presto would have won"*.
   Given a banked 0.910837, 2–3 slots and 3 days, the recommendation is to run fine-tuned Presto
   **offline only**, for the code-review writeup. See "IF WE COULD ONLY DO ONE THING".
4. The brief's hunch that generic TS foundation models might fit better is **wrong** — they are
   channel-independent, which structurally destroys the cross-band-at-fixed-t signal (VH<−21 dB
   permanence, VH−VV, NDVI) that our best features are built from.

**Target claim under attack:** "No pretrained geospatial foundation model fits our input shape, so
pretrained weights are useless to us. In particular Presto is dead."

**Our shape:** 1817 train rows x (12 bands x 12 months). Test 1030 rows, each with only 4-6
CONTIGUOUS observed months. No lat/lon. No absolute date. No imagery, no patches — one pixel's
multi-band monthly time series. Metric 0.6*F1(hard 0.5) + 0.4*AUC. Weights legal, external DATA
illegal, AutoML forbidden.

---

## VERDICT TABLE

| Model | Input required | Our shape fits? | Needs lat/lon | Needs abs. dates | Frozen-embedding path | Fits ~25 min Colab | Compliance risk |
|---|---|---|---|---|---|---|---|
| **Presto** (402k) | (N,T,17) pixel series + mask + DW + latlons + month; T in 1..24 | **YES — near-exact.** 12/12 bands map; ERA5/SRTM/DW officially maskable; adapter already written in `src/presto_features.py` | Arg is mandatory, but a **constant** is author-sanctioned (S2-Agri100 precedent) | **NO** — `month` defaults to 0, int or per-row tensor | **YES** — `eval_task=True` -> 128-d mean-pooled | **YES** (frozen: seconds; fine-tune: minutes) | LOW (MIT, weights legal) |
| **Galileo-Tiny** (5.3M) | 4 streams (space-time/space/time/static) + 4 masks + months + patch_size | **YES, likely better** — same band groups, `patch_size` can be 1, unused streams fully masked | **NO — no latlon arg exists** | **NO** (months arg, constant OK) | **YES** | YES | LOW-MED (licence unchecked) |
| Prithvi-EO-2.0 (300/600M) | T x (H,W) images, 3D conv over (t,h,w) cubes | **NO** — needs spatial extent | no | yes (temporal PE) | yes | no (600M) | n/a |
| SatMAE / SatMAE++ | image patches (fMoW) | **NO** — imagery | no | yes | yes | marginal | n/a |
| Clay v1.5 (768-d) | 256x256 chips + wavelengths + GSD + **lat/lon** + **timestamp** | **NO** — fails on all three axes | **YES** | **YES** | yes | no | n/a |
| TerraMind | image tiles, 9 modalities, dual-scale early fusion | **NO** — imagery | no | varies | yes | no | n/a |
| DOFA | image + wavelength-conditioned dynamic weights | **NO** — imagery | no | no | yes | marginal | n/a |
| AnySat | multi-scale image tiles, JEPA | **NO** — spatial encoders | no | yes | yes | no | n/a |
| SITS-Former | small spatial patches of S2 SITS | **NO** — imagery | no | yes | yes | yes | n/a |
| MOMENT | univariate, zero-padded to **512**, patch 8 | **Surgery only** — pad 12->512 (2.3% occupancy); channel-independent | no | no | yes | yes | LOW |
| Mantis (8M) | univariate, fixed length, ViT, **classification-native** | **Surgery only** — channel-independent + PCA adapter | no | no | **yes, frozen is its selling point** | yes | LOW |
| Chronos | univariate forecasting, tokenized | **NO** — task + univariate | no | no | partial | yes | n/a |
| TimesFM | univariate decoder-only forecasting | **NO** — task | no | no | partial | yes | n/a |
| Lag-Llama | univariate probabilistic forecasting, long lags | **NO** — task + length 12 | no | no | no | yes | n/a |
| Moirai | any-variate forecasting, long context | **NO** — task | no | no | partial | yes | n/a |
| UniTS | multi-task incl. classification, long-context patching | Marginal | no | no | yes | yes | LOW |
| **TabPFN v2** | table <=10k rows x <=500 cols, NaN native | **YES — 1817x144 is its design centre** | no | no | in-context, no training at all | **YES (fastest here)** | **MED — AutoML question, see N16.** DATA risk is the LOWEST of all (synthetic pretraining corpus) |

**Reading the table:** only 3 rows are green — Presto, Galileo, TabPFN v2. The EO models die on
*imagery patches*; the forecasting FMs die on *task*; MOMENT/Mantis die on *channel independence*
(they can be forced to run, but not to see our signal).

---

## EARLY LEAD TO VERIFY: `construct_single_presto_input`

Status: **CONFIRMED EXISTS** (documented in repo README, exported as `presto.construct_single_presto_input`).
NOT in `single_file_presto.py` (read that file — it has only Attention/Mlp/LayerScale/Block/Encoder/
Decoder/PrestoFineTuningModel/FinetuningHead/Presto + get_sinusoid_encoding_table /
get_month_encoding_table / month_to_tensor). NOT in `presto/utils.py` either (read it — only
update_data_dir, seed_everything, initialize_logging, timestamp_dirname).
**FOUND: `presto/dataops/utils.py`.** `presto/__init__.py` contains
`from .dataops.utils import construct_batch_presto_input, construct_single_presto_input`.
**The lead is TRUE and it is exactly what it was hoped to be: an official partial-band /
partial-modality entry point.** Full detail in N2. The prior run's dying agent was right.

---

## RAW NOTES (append-only, newest at bottom)

### N1. Presto README (https://github.com/nasaharvest/presto/blob/main/README.md) — verbatim-ish
Five inference inputs:
- `x`: `[batch_size, num_timesteps, bands]`, normalized bands.
- `latlons`: `[batch_size, 2]` — latitude/longitude. **REQUIRED ARG.**
- `month`: starting month, int OR tensor (so per-row start months are allowed).
- `dynamic_world`: `[batch_size, num_timesteps]`; **"when unavailable, fill with the value `9`"**
  to tell the model to ignore it. -> DW is OPTIONAL by an explicit sentinel. GOOD FOR US.
- `mask`: same shape as `x`; `mask[i,j,k]==1` => ignore `x[i,j,k]`. Per-element, per-row.
- README quote: "3 of the input tensors (`x`, `dynamic_world`, `mask`) can be generated using
  `presto.construct_single_presto_input`"
- README quote: "The number of timesteps passed is optional, and can be any value between 1 and 24
  (2 years of data)." -> 12 timesteps is squarely in range. GOOD FOR US.

Implication so far: partial-band + partial-time input is an OFFICIALLY SUPPORTED path. The two open
blockers are (a) latlons required, (b) month required. Both need code-level checking.

### N2. LEAD VERDICT — `construct_single_presto_input` IS REAL
Located: `presto/dataops/utils.py`, imported in `presto/__init__.py` as
`from .dataops.utils import construct_batch_presto_input, construct_single_presto_input`.
Read via raw.githubusercontent. Both functions:
- take OPTIONAL `s1`, `s2`, `era5`, `srtm` tensors, each with an accompanying band-name list;
- validate consistent timestep counts across whatever was provided;
- map provided bands into the canonical `BANDS` layout via `BANDS.index()`;
- **set mask=0 only for the bands you supplied — everything else stays masked (=1)**;
- optionally compute NDVI when both B8 and B4 are present;
- default `dynamic_world` to the `class_amount` sentinel (=9, "ignore") when not provided;
- normalize via `S1_S2_ERA5_SRTM.normalize()` when `normalize=True` (default).
`construct_batch_presto_input` is the batched twin: `(batch, timesteps, bands)`.
=> **The "we have no ERA5/SRTM/DynamicWorld" objection is REFUTED at the code level.** Missing
channel groups are an officially-supported first-class case, not a hack.

### N3. Presto canonical band layout (`presto/dataops/pipelines/s1_s2_era5_srtm.py`)
BANDS order: VV, VH, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12, temperature_2m,
total_precipitation, elevation, slope, NDVI. NORMED_BANDS drops B9.
BANDS_GROUPS_IDX (OrderedDict): S1, S2_RGB, S2_Red_Edge, S2_NIR_10m, S2_NIR_20m, S2_SWIR, ERA5,
SRTM, NDVI. **Masking/tokenization is per GROUP, not per raw band** — important for point 4.
Normalization: S1 (VV,VH) add 25.0 divide 25.0; S2 divide 10000.0 (i.e. raw reflectance scaled);
ERA5 [-272.15, 0.0]/[35.0, 0.03]; SRTM /[2000.0, 50.0]; NDVI /1.0. Default num_timesteps = 12.

### N4. OUR ACTUAL COLUMNS (read `data/raw/Train.csv` + `Test.csv` headers directly)
12 bands x 12 months, band names:
`VH, VV, blue, green, nir, nira, re1, re2, re3, red, swir1, swir2`
Missing sentinel in Test.csv is **-9999** (both int and float columns).
Mapping to Presto BANDS is essentially EXACT:
  VV->VV, VH->VH, blue->B2, green->B3, red->B4, re1->B5, re2->B6, re3->B7, nir->B8,
  nira->B8A, swir1->B11, swir2->B12.
That is 12 of Presto's 13 non-auxiliary bands — we are missing only B9 (which Presto DROPS from
NORMED_BANDS anyway). **We have a 100% match on the bands Presto actually normalizes for S1+S2.**
Only ERA5 (2), SRTM (2), NDVI (1, derivable from nir+red) and DynamicWorld are absent — and NDVI
is auto-computed by construct_*_presto_input because we have both B8 and B4.
S2 units: our reflectance values are ~1100-2600 integers => already the /10000 scale Presto expects.
S1 units: our VV/VH are ~-20 to -36 => **dB**, which is exactly what Presto's (x+25)/25 assumes.

### N5. !!! THE TEAM HAS ALREADY BUILT A PRESTO LANE !!!
`run_presto.py` (170 lines) exists at repo root, plus `src/presto_features.py::embed`,
`tools/fetch_presto.py`, and a vendored `vendor/presto/presto_core.py`.
Docstring: "frozen pretrained encoder + a ~129-parameter logistic head", `Presto.load_pretrained().encoder`,
all params frozen, 5-fold CV on the head, `--month-mode {const,true}` where const = "calendar
identity DELETED". It already applies `_mask_views` so train is shown 4-6 months like test.
Its stated go/no-go is the ADVERSARIAL AUC on embeddings.
=> So the conclusion "Presto is dead" may be an EMPIRICAL result from this script, not a shape
argument. Must find the run output before claiming anything. NEXT: read src/presto_features.py
and search logs for the adversarial AUC number.

### N6. `src/presto_features.py` — the adapter ALREADY EXISTS and is already correct
Read in full. `to_presto(cube, sar_units='db')` builds `(N,T,17)` x + `(N,T,17)` mask with
Presto's 1=masked convention; `OUR_TO_PRESTO = [1,0,2,3,4,5,6,7,8,9,10,11]` (our VH-first ->
Presto VV-first, asserted by `_check_band_order`); ADD_BY/DIV_BY hardcoded to match N3 exactly;
NDVI derived at index 16 with `mask[NDVI] = max(mask[B8], mask[B4])`; ERA5+SRTM indices [12,13,14,15]
masked everywhere. `embed()` calls
`encoder(x, dynamic_world=dw, mask=mask, latlons=latlons, month=month, eval_task=True)` in batches
of 256 and returns **(N, 128)**.
It sets `latlons = torch.zeros(n, 2)` — the docstring calls latlons "MANDATORY and NOT maskable".
`month_mode='const'` sets month=0 for all rows (relative-time reframing); `'true'` uses
`argmax(mask[:,:,0]==0)` = first month with VV observed.
Its docstring already records the round-07 finding: **"Presto masks at TOKEN-GROUP granularity, so
losing `red` alone drops the whole S2_RGB token for that month."**

### N7. CODE VERIFICATION of the `mask_tokens` question (task point 4) — BOTH sides confirmed
(a) `single_file_presto.py`, `Encoder.mask_tokens` — **the uniform-mask assert is REAL**, verbatim:
```python
@staticmethod
def mask_tokens(x, mask):
    summed = mask.sum(dim=(1, 2))  # summed tells me the number of masked elements per batch idx
    assert summed.max() == summed.min(), f"{summed.max()}, {summed.min()}"
```
=> the single-file revision CANNOT batch rows with different observed-month counts. Our 4/5/6-month
test windows would trip this on nearly every batch.
(b) `presto/presto.py`, `Encoder.mask_tokens` — **NO assert; variable mask counts ARE handled**, verbatim:
```python
@staticmethod
def mask_tokens(x, mask):
    mask = mask.bool()
    sorted_mask, indices = torch.sort((~mask).int(), dim=1, descending=True, stable=True)
    x = x.gather(1, indices[:, :, None].expand_as(x))
    x = x * sorted_mask.unsqueeze(-1)
    max_length = sorted_mask.sum(-1).max()
    x = x[:, :max_length]
    updated_mask = 1 - sorted_mask[:, :max_length]
    return x, indices, updated_mask
```
It stable-sorts unmasked tokens to the front, truncates to the batch max, and returns an
`updated_mask` used downstream as an attention mask. **So the answer to task point 4 is: the assert
exists in one revision and is GONE in the maintained one. Use `presto/presto.py`. No padding hack,
no per-row batches needed.** `tools/fetch_presto.py` in our repo already documents and acts on this.

### N8. `presto/presto.py` Encoder.forward — verbatim signature
```python
def forward(self, x, dynamic_world, latlons, mask=None, month=0, eval_task=True)
```
- `latlons` is POSITIONAL and has NO default => structurally mandatory. It is consumed by a static
  `cartesian()` (geodetic -> unit-sphere xyz) then a linear `latlon_embed`. There is no
  latlon-dropout / latlon-mask argument in the signature. **Task point 2 answer: YES it is required,
  and the only code-level path is to pass a CONSTANT (our repo passes zeros).**
- `month` default 0, `Union[torch.Tensor, int]` => per-row start months allowed, and a constant is
  legal. **Task point 3 answer: an absolute date is NOT required; month=0 for every row is a
  supported call.**
- `eval_task=True` -> single mean-pooled embedding over valid tokens (**dim = `embedding_size`,
  default 128**). `eval_task=False` -> full token sequence + indices + updated mask.
  **Task point 5 answer: YES, a documented frozen-feature path exists** (also demonstrated in the
  repo's `downstream_task_demo.ipynb`).

---

## !!! THE ACTUAL KILL — AND WHY IT NO LONGER STANDS !!!

### N9. What actually killed Presto (read `experiments/LB_LOG.md` iter17, line ~1803; `PROJECT_STATE.md` ~344)
It was **not** a shape argument. It was an offline screen, and **zero submissions were spent**:

| config | adv-AUC on embeddings | ATC-F1 vs champ | OOF combined | verdict |
|---|---|---|---|---|
| c_presto_const | 0.9757 | −0.0444 LB | 0.9672 | HOLD (1/2) |
| c_presto_true  | 0.9668 | −0.0589 LB | 0.9693 | HOLD (1/2) |

Three reasons were given: (1) adv-AUC > 0.9 ("the encoder is *encoding* the shift"); (2) ATC-F1 puts
it 0.044–0.059 LB below champion; (3) its OOF 0.967–0.969 is below champion's 0.975.

### N10. **ALL THREE INSTRUMENTS HAVE SINCE BEEN RETIRED BY THIS PROJECT'S OWN LATER WORK.**
This is the refutation, and it comes from our own files, not from literature.

1. **adv-AUC is DEAD as a selection criterion.** `PROJECT_STATE.md` (round-18 block): *"adv-AUC is
   DEAD as a selection criterion — Spearman vs realized transfer is +0.68 (transforms) / +1.00
   (modalities), i.e. BACKWARDS; a mild synthetic shift scores adv-AUC 0.9955 at 0.0046 AUC cost
   while our REAL shift is 0.9670, so the statistic saturates before it measures anything."*
   The iter17 go/no-go was **precisely** an adv-AUC > 0.9 rule. By the project's own round-18
   finding, Presto's 0.9757 is **non-evidence, and if anything reads positive.**
2. **ATC-F1 is not valid out-of-family.** iter25/26: *"all 7 original anchors sit at an identical
   24-channel width, so ATC-F1 was only ever certified WITHIN that family; `c_dropvv` (22 ch) was
   the first REPRESENTATION change and adding it as an 8th anchor drops ρ +0.964 → +0.738 ... the
   gate does not catch out-of-family failure."* A frozen 128-d Presto embedding + logistic head is
   the **most out-of-family candidate the project has ever screened** — further outside the anchor
   family than `c_dropvv` was. The −0.0444/−0.0589 ATC-F1 reads were produced by an instrument the
   project later proved invalid in exactly this regime.
3. **OOF is blind.** `MEMORY.md`: *"OOF is blind, Zindi LB is ground truth"*; LB_LOG repeatedly:
   *"OOF anti-correlated again (highest-OOF pondband still lost)"*. A 0.008 OOF deficit is not a
   kill by this project's own standing rule.

**Conclusion on the target claim.** The proposition *"no pretrained geospatial foundation model
fits our input shape"* is **FALSE and refuted at code level** — Presto fits our shape almost
exactly (N4: 12/12 band match; N7: variable masking supported; N8: 12 timesteps supported, constant
month legal). The proposition *"Presto is dead"* rests on an offline screen whose three instruments
were all subsequently retired. **Presto was never submitted. It has never actually been measured.**

### N11. What was ALSO never tried (from `PROJECT_STATE.md` line ~131, our own words)
*"pretrained per-pixel foundation models fine-tuned (legal per rules; **Presto was frozen-only in
iter17 — fine-tuning untested**)"*. And the frozen head that WAS tried was a **~129-parameter
logistic regression on a mean-pooled 128-d vector** — which throws away the entire token sequence
(`eval_task=False` path) and every per-band missingness indicator the champion relies on.
So the measured evidence covers ONE point in the space: frozen + mean-pool + linear.

### N12. PRESTO PAPER (arXiv:2304.14065, read via ar5iv HTML) — three published facts that each
### remove one of our supposed blockers
- **Parameter count 402k (encoder), embedding size 128.** Confirms N8.
- **LAT/LON — published precedent for NOT having it.** On S2-Agri100, which lacks per-pixel
  coordinates, the authors *"used the location of the central pixel"* for all samples and
  *"Presto remained performant"*. => **Feeding a single constant lat/lon to every row is a
  published, author-sanctioned degradation, not an abuse.** Our repo's `latlons = zeros(n,2)` is
  the same move (0,0 is in the Gulf of Guinea, but since it is CONSTANT across all rows it
  contributes an identical additive token to every sample and carries zero discriminative signal —
  it can only shift the encoder off its pretraining manifold, not leak or distort ranking).
- **TIMESTEPS — 3 is enough.** The paper reports Presto *"remained performant when receiving only 3
  input timesteps"*. Our test rows have 4–6. Not a blocker.
- **DYNAMIC WORLD — removable.** *"negligible performance differences when Dynamic World input was
  removed."* Confirms the `9` sentinel is real and cheap.
- **n ~ 2000 evidence (task point 6).** CropHarvest binary tasks, F1: **Kenya 0.861 (TIML 0.838),
  Brazil 0.888 (TIML 0.835), Togo 0.760 (TIML 0.732)** — all three beat the prior SOTA
  meta-learning baseline. These are exactly the small-n (order 10^3 labelled points), single-pixel,
  multi-band, 12-timestep, binary land-use tasks our competition is. **This is the closest published
  analogue to our problem that exists, and Presto wins it.**
  CAVEAT: the paper section I fetched did not state per-task label counts; I have NOT verified the
  exact n for Togo/Kenya/Brazil and am not asserting it.
- Fine-tuned regression: fuel moisture RMSE 25.28 (baselines 23.84–28.75 — i.e. Presto is NOT best
  here), algae blooms RMSE 0.815 (baselines 0.850–1.249 — best). Honest reading: Presto is
  competitive, not dominant, outside its crop-type home turf.

### N13. GALILEO (nasaharvest/galileo, ICML 2025) — the successor, and it is a BETTER shape fit
Read `src/galileo.py`. `Encoder.forward(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months,
patch_size, input_resolution_m, exit_after, token_exit_cfg, add_layernorm_on_exit)`.
- Four data streams with four independent masks: space-time, space, time, static.
  **Mask convention: 0 = keep, 1 = masked, 2 = decode.** No modality is mandatory — absence is
  expressed purely through the mask. Same graceful-degradation story as Presto, generalized.
- **NO lat/lon argument at all.** Position embeddings derive from spatial patch coordinates and a
  `input_resolution_m` ground-sample-distance ratio, not absolute geolocation.
  => **Galileo removes the one hard blocker Presto has.** (INFERRED FROM CODE — file `src/galileo.py`,
  `Encoder.forward`; I did not find a doc statement saying "latlon not required".)
- `months` is still an input (temporal encoding) — same constant-month workaround applies.
- `patch_size` ≥ 1, so a 1x1 "patch" (= our single pixel) is representable. Galileo's own benchmark
  suite explicitly includes **pixel time series classification** and **CropHarvest**, which is the
  1x1 pixel-timeseries regime.
- Galileo-Tiny is **5.3M params** — 13x Presto but still trivially embeddable on a Colab GPU.
- NOT YET VERIFIED: the exact band lists (`SPACE_TIME_BANDS`, `TIME_BANDS`, `SPACE_BANDS`,
  `STATIC_BANDS` live in `src/data/dataset.py`, not read), and whether the released checkpoint's
  S1/S2 band set matches our 12. Treat the "better fit" claim as promising-but-unverified on bands.

### N14. Galileo bands — VERIFIED, and they are the SAME layout as Presto
`src/data/dataset.py`: `SPACE_TIME_BANDS = EO_SPACE_TIME_BANDS + ["NDVI"]`, with
`SPACE_TIME_BANDS_GROUPS_IDX` = S1, S2_RGB (B2,B3,B4), S2_Red_Edge (B5,B6,B7), S2_NIR_10m (B8),
S2_NIR_20m (B8A), S2_SWIR (B11,B12), NDVI. **Identical group structure to Presto's
BANDS_GROUPS_IDX**, minus ERA5/SRTM which Galileo moves into the separate TIME_BANDS (ERA5, TC,
VIIRS) / SPACE_BANDS (SRTM, DW, WC) / STATIC_BANDS streams — i.e. streams we simply mask out
entirely. `Normalizer` class does shift/divide with optional mean±k*std mode.
=> **Our existing `src/presto_features.py::to_presto` band mapping transfers to Galileo almost
verbatim** (drop indices 12–15, keep the 12 bands + NDVI as the space-time stream). This is the
single cheapest new lane available: the adapter is 90% written.

---

## TABULAR: TabPFN v2

### N15. TabPFN v2 capability facts
- Hollmann et al., **Nature 2025**, "Accurate predictions on small data with a tabular foundation
  model". Weights on HF: `Prior-Labs/TabPFN-v2-clf`.
- **Official envelope: up to 10,000 samples and 500 features.** Ours: 1817 rows x 144 features.
  **Squarely inside — this is literally TabPFN's design centre.**
- **Native missing values.** Per the implementation: missing values are encoded as `-2.0`
  (Inf as `2.0`, -Inf as `4.0`), features are mean-imputed, and **an additional "missingness"
  channel is concatenated to the features before the linear embedding**. So TabPFN carries a
  missingness indicator internally — the same device our champion carries by hand.
  (Source: implementation description; I have NOT read PriorLabs/TabPFN source myself — flagged in
  CAVEATS.)
- Inference is **in-context**: one forward pass over (train set + test row), no gradient training.
  On 1817x144 this is seconds-to-a-couple-of-minutes on a Colab GPU. **Fits the compute budget with
  room to spare** — this is the only candidate here that costs far LESS than 25 minutes.

### N16. IS TabPFN AutoML? — the strongest version of the objection, then the defence
**THE OBJECTION (steelmanned — do not wave this away):**
1. TabPFN comes out of the Freiburg AutoML lab (Hutter et al.); its predecessor was published and
   marketed in AutoML venues, and the Nature paper's headline comparison is *against AutoML systems*
   (AutoGluon), positioning it as a drop-in replacement for them. A reviewer who knows the field
   will recognise the lineage instantly.
2. Its selling point is *"no hyperparameter tuning needed"* — it INTERNALISES model selection.
   Functionally it occupies the same slot in a workflow that an AutoML system occupies: you hand it
   a table and a target and it returns predictions with no modelling decisions. If the rule's INTENT
   is "the entrant must make and defend the modelling choices", TabPFN arguably defeats that intent
   even if it defeats no literal clause.
3. The ecosystem ships `AutoTabPFNClassifier` (post-hoc ensembling over preprocessing/config
   portfolios) in `tabpfn-extensions`. That wrapper **IS** AutoML by any definition. If we import
   TabPFN at all, a reviewer may not distinguish which class we called.
4. Even plain `TabPFNClassifier` defaults to `n_estimators > 1`, doing internal ensembling over
   feature/class permutations. That is test-time augmentation, but it is an *automated ensemble*,
   and someone determined to read it as AutoML has a hook.
**THE DEFENCE:**
1. Plain `TabPFNClassifier` performs **no search of any kind** — no hyperparameter optimisation, no
   model selection, no architecture search, no pipeline search, no validation-driven choice. It is
   a single fixed pretrained transformer evaluated in one forward pass. That is *definitionally* a
   pretrained model used for inference, which the rules explicitly permit — the same legal category
   as Presto.
2. The AutoML functionality is a **separate package and a separate class**. Not importing
   `tabpfn_extensions` is a bright, auditable line: `pip freeze` and the import list prove it.
3. `n_estimators` permutation-ensembling is deterministic given a seed and is a fixed property of
   the released model's default inference procedure, not a search over configurations we chose.
**VERDICT: DEFENSIBLE BUT NOT FREE.** It is a live compliance risk that costs writeup space and
reviewer goodwill in a contest that is 35% code review. If we run it, we must (a) use only
`TabPFNClassifier` with stated defaults, (b) never install `tabpfn-extensions`, (c) state the
AutoML question and its answer explicitly in the writeup rather than hoping nobody asks.
**COMPLIANCE ON DATA — and this is TabPFN's one clean win:** TabPFN v2 is pretrained
**entirely on synthetic data generated from structural causal models**. There is no external
real-world corpus in it at all, so the "external data" clause cannot even be engaged. On the data
axis TabPFN is the *safest* candidate in this document — cleaner than Presto (pretrained on 21.5M
real Sentinel pixel series) or Galileo (11 real EO datasets).

---

## WIDER SURVEY

### N17. Imagery-patch EO models — ALL DISQUALIFIED, and the killing constraint is the same one
- **Prithvi-EO-2.0** (IBM/NASA/Jülich, arXiv:2412.02732). ViT-MAE with **3D patch embeddings: a 3D
  convolution dividing the input into non-overlapping cubes of size (t, h, w)** over a sequence of
  T images of size (H, W). **KILLED BY: requires spatial extent (h, w > 1).** We have h=w=1; the 3D
  conv and 2D positional embeddings degenerate. Also 300M/600M params.
- **Clay v1.5.** Input = image chips (156x256 / 256x256 at 10 m nominal scale) PLUS metadata:
  **wavelengths, GSD, latitude/longitude, and time step (week/hour)**; the position encoding is
  scaled by GSD and *combined with* lat/lon and time step. Output 768-d.
  **KILLED THREE TIMES: needs image chips, needs lat/lon, needs absolute timestamps.** Worst fit here.
- **SatMAE / SatMAE++.** MAE over fMoW image patches with temporal + spectral positional encodings
  attached to image tokens. **KILLED BY: imagery patches.**
- **TerraMind** (dual-scale early fusion over nine modalities), **DOFA** (wavelength-conditioned
  dynamic weight generator), **AnySat** (scale-adaptive *spatial* encoders + JEPA). All three are
  image-tile models; their flexibility story is about spatial scale and modality count, not about
  running with no spatial dimension. **KILLED BY: imagery patches.**
- **SITS-Former.** Consumes small spatial patches of a Sentinel-2 SITS, not a single pixel.
  **KILLED BY: imagery patches.**
- **Galileo** is the exception among 2025-era EO models precisely because it inherits Presto's
  pixel-timeseries lineage (N13/N14): `patch_size` can be 1 and its benchmark suite explicitly
  includes pixel time series classification and CropHarvest.
**So: exactly two EO foundation models in the current landscape are shape-compatible with a single
pixel's time series — Presto and Galileo — and they are from the same lab (NASA Harvest).** That is
why the target claim felt true. It is nonetheless false: two is not zero.

### N18. General time-series foundation models — a shared, decisive architectural mismatch
- **MOMENT** (arXiv:2402.03885). Fixed input length **512**; shorter series are **left-padded with
  zeros**, with an `input_mask` marking padding. **Channel-independent**: multivariate series are
  processed one channel at a time along the batch dimension, and *"for classification, every channel
  is processed independently and all embeddings are concatenated before the classification head."*
  Patch length 8 => 64 patches.
- **Mantis** (Feofanov et al., arXiv:2502.15637, 2025). ViT-based, contrastively pretrained,
  **purpose-built for time-series CLASSIFICATION**, beats other FMs both frozen and fine-tuned, best
  calibration error. Pretrained **univariately**; multivariate handled channel-independently plus
  proposed **adapters (e.g. PCA)** to cut memory and model channel dependence. A v2 exists
  (arXiv:2602.17868, MantisV2, synthetic data + test-time strategies).
- **Chronos** — univariate probabilistic FORECASTING via value tokenization. No classification head,
  no multivariate. **KILLED BY: task and channel count.**
- **TimesFM** — univariate decoder-only FORECASTING (later versions add covariates). **KILLED BY: task.**
- **Lag-Llama** — univariate probabilistic FORECASTING whose inductive bias is *lag features* at
  horizons far longer than 12. **KILLED BY: task and length.**
- **Moirai** — any-variate FORECASTING, masked encoder, genuinely multivariate. Still forecasting,
  still long-context. **KILLED BY: task.**
- **UniTS** — multi-task including classification; the least-bad generic candidate, same long-context
  patching regime.
**THE SHARED KILLER, architectural not incidental:**
1. **Length 12 against a native context of 512 with patch length 8.** Our entire series is ~1.5
   patches; zero-padding puts real data at 2.3% occupancy — a regime no published evaluation covers.
   (MOMENT justifies 512 because *"a large majority of classification datasets have time series
   shorter than 512"* — but the UCR/UEA sets it pads are typically 100–1000 long, not 12.)
2. **Channel independence destroys exactly where our signal is.** Our discriminative structure is
   CROSS-BAND AT A FIXED TIMESTEP: the VH < −21 dB permanence indicator (our single biggest feature
   win, +0.010 LB), the VH−VV dual-pol gate, NDVI = f(nir, red). MOMENT and Mantis encode each of
   our 12 bands as an independent 12-point univariate series and only concatenate at the head.
   **A linear head on concatenated per-channel embeddings cannot represent a product of two
   channels.** Presto/Galileo tokenize by (band-group x timestep), preserving cross-band interaction
   inside the attention. This is decisive.
3. **Missingness is not native.** MOMENT's `input_mask` marks *padding*, not scattered mid-series
   missingness; our 4–6-month contiguous windows are only expressible by treating the observed run
   as the whole series and discarding where in the year it sat.
**VERDICT: the generic TS foundation models fit WORSE than the EO ones, not better.** The brief's
intuition ("modality-agnostic may fit better") is refuted by the channel-independence property.
The one honest exception worth naming: **Mantis frozen embeddings are cheap** (8M params, one
forward pass per channel, no fine-tuning) and would cost ~minutes — but see KILL CONDITION below.

---

## THE THREE SURVIVORS — MECHANISM, INTEGRATION, KILL CONDITION, EFFECT SIZE

### S1. PRESTO — refuted as "dead"; revivable at near-zero engineering cost
**MECHANISM.** 402k-param transformer pretrained with masked-modality/masked-timestep
self-supervision on 21.5M Sentinel-1/2 pixel time series. Its pretraining objective —
reconstruct masked channels and masked timesteps — *is our test-time condition*. It tokenizes
(band-group x timestep), so cross-band interactions live inside attention.
**INTEGRATION SKETCH (against 12 bands x 12 months, 4–6 observed).** Already written; see
`src/presto_features.py`, `run_presto.py`, `tools/fetch_presto.py`. Concretely:
  1. `python tools/fetch_presto.py` — vendors `presto/presto.py` (NOT `single_file_presto.py`; N7)
     + `default_model.pt` (3.3 MB) with a shim replacing the dataops imports.
  2. `to_presto(cube)` -> `x (N,12,17)`, `mask (N,12,17)`; our 12 bands map onto Presto slots
     [VV,VH,B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12], NDVI derived at slot 16, ERA5+SRTM slots 12–15
     masked, `dynamic_world = 9`, `latlons = 0`, `month = 0`.
  3. Train cube pushed through `_mask_views` first so train sees 4–6 months like test.
  4. `encoder(..., eval_task=True)` -> (N,128) frozen embeddings.
**WHAT TO CHANGE vs iter17 (the part that was never tried):**
  (a) **FINE-TUNE.** `Presto.construct_finetuning_model` / `PrestoFineTuningModel` +
      `FinetuningHead` exist in the repo. 402k params on 1817 rows x 5 folds x 10 seeds is a few
      minutes, not 25. PROJECT_STATE explicitly flags fine-tuning as untested.
  (b) **Use `eval_task=False`** to get the token sequence and apply OUR pooling
      (`mean_min`, the pooling that won iter38) instead of Presto's built-in mean.
  (c) **Do NOT re-run the frozen mean-pool + 129-param logistic head.** That point is measured.
**EVIDENCE.** Code: `presto/presto.py::Encoder.forward`, `Encoder.mask_tokens`,
`presto/dataops/utils.py::construct_single_presto_input`,
`presto/dataops/pipelines/s1_s2_era5_srtm.py` (BANDS, BANDS_GROUPS_IDX, normalization).
Paper: arXiv:2304.14065 (CropHarvest F1 Kenya .861 / Brazil .888 / Togo .760; 3-timestep result;
DW-removal result; S2-Agri100 constant-location result).
**KILL CONDITION (cheap, offline, 0 submissions).** Do **NOT** reuse adversarial AUC — round-18
retired it and it reads BACKWARDS. Instead use the round-18 replacement, the **PAIRED-DELTA
transfer gate with a RANK-vs-CUT decomposition** that already exists in this project, applied to
fine-tuned-Presto OOF vs champion OOF on window-matched views. Secondary free check: does
fine-tuned Presto's OOF AUC exceed the champion's 0.975 on the SAME `_mask_views` protocol? Frozen
Presto scored 0.967–0.969; if fine-tuning does not close that 0.008, stop.
**EXPECTED EFFECT SIZE, honestly bounded.** The measurable floor in this project is ~0.010–0.013 LB
and only TWO effects have ever cleared it (model-class swap +0.050; per-cell detrend −0.051). A
model-class swap is the right *category*. But: (i) the champion at 0.9108 already encodes
shift-invariance machinery Presto lacks; (ii) frozen Presto's OOF was already 0.008 BELOW champion;
(iii) Presto's own paper shows it competitive-not-dominant off crop-type. **Honest band: −0.05 to
+0.02, centred slightly negative. P(beats 0.910837 by ≥0.006) I would put at 10–20%.**
**EXECUTION RISK. HIGH given 3 days and 2–3 slots.** Fine-tuning is new code (the existing lane is
frozen-only), needs its own seed pool to escape the 0.019 seed-variance floor, and a single-seed
read would be a mirage — this project has been burned by single-seed mirages three times
(`c_repl_vhsq` 0.9133 -> 0.8995). A credible read needs 5 seeds = most of one Colab slot.
**COMPLIANCE.** WEIGHTS legal (rules verified). Presto is **MIT-licensed** and the checkpoint is
openly available to everyone, satisfying the "openly available" clause literally. Pretraining
corpus is external real Sentinel data — **but that is the model's training, not our training data**;
we add no external DATA to our pipeline. If a reviewer disputes this, the dispute applies equally
to *every* pretrained model and the rules already resolved it. `tools/fetch_presto.py` vendoring
(source + 3.3 MB checkpoint in-repo, MIT) is the right reproducibility posture. **LOW RISK.**

### S2. GALILEO — the strongest untried shape fit, but the least de-risked
**MECHANISM.** ICML 2025, nasaharvest. Multi-stream (space-time / space / time / static) transformer
with per-stream masks (0=keep, 1=masked, 2=decode), pretrained on 11 EO datasets, benchmarked on 15
tasks including **pixel time series classification and CropHarvest**. Galileo-Tiny = 5.3M params.
**SHAPE FIT — better than Presto:** identical S1/S2 band-group layout (N14), `patch_size` may be 1,
**and no lat/lon argument exists at all** (N13) — it removes Presto's only hard blocker.
**INTEGRATION SKETCH.** Reuse `src/presto_features.py::to_presto` with slots 12–15 removed: build
`s_t_x (N, 12, 1, 1, 13)` [T=12, H=W=1, bands = our 12 + NDVI] and `s_t_m` with 1 where unobserved;
pass `t_x/sp_x/st_x` as zeros with their masks entirely 1; `months = zeros`; `patch_size=1`.
Encoder -> pooled embedding -> our existing head. Frozen first.
**KILL CONDITION.** Before any submission: (i) confirm the released checkpoint's
`SPACE_TIME_BANDS` order matches our mapping (read `src/data/earthengine/eo.py` — NOT DONE);
(ii) confirm a `patch_size=1`, `H=W=1` forward pass runs without a shape error — a 20-line smoke
test, zero cost; (iii) same paired-delta transfer gate as S1.
**EXPECTED EFFECT SIZE.** Unknown and wider than Presto's, because nothing has been measured.
Galileo beats Presto on CropHarvest in its own paper, so if the Presto lane's true LB is near the
champion, Galileo's is plausibly a little above. **But this is the classic last-48-hours trap:
maximum novelty, zero measurement, and one slot.**
**EXECUTION RISK: VERY HIGH.** New vendoring, unverified band constants, unverified 1x1 path,
unverified checkpoint availability, and a 5.3M-param model needing fine-tuning to shine. With
2–3 slots and 3 days I would not start this.
**COMPLIANCE.** Same as Presto (openly-available weights). Check the licence before shipping —
NOT VERIFIED by me.

### S3. TabPFN v2 — best envelope match, worst mechanism match
**MECHANISM.** In-context learning: a fixed transformer pretrained on synthetic SCM-generated
tables consumes (train table, test row) in one forward pass. 10k samples / 500 features envelope;
ours is 1817 x 144. Missing values native (encoded -2.0 + a concatenated missingness channel).
**INTEGRATION SKETCH.** Flatten the cube to 144 columns exactly as the CSV already is; set
unobserved cells to NaN; **push train rows through `_mask_views` first** so the train table carries
the same 4–6-month missingness pattern as test (without this the missingness channel alone becomes
a perfect train/test discriminator and the model is worthless); `TabPFNClassifier().fit(Xtr,
ytr).predict_proba(Xte)`; then the existing `calibrate_legal` Platt + literal 0.5.
**KILL CONDITION (free).** This project has already measured that **tree/tabular models do not
transfer here**: naive CatBoost 0.6976, catblend −0.0136, shift-robust CatBoost **0.718607**
(LB_LOG iter30/iter40), against a Transformer at ~0.90. The verdict was *"TREES DO NOT TRANSFER ...
conditional shift fatal"*. TabPFN v2 is an in-context tabular learner whose inductive bias is much
closer to a tree ensemble / kernel than to a shift-invariant sequence model, and it has **no
mechanism at all for relative-time reframing** — the single device that makes our champion work.
**Free check before spending anything: run TabPFN on the window-matched OOF split and compare to
the CatBoost number. If it lands in the 0.70–0.80 region like every other tabular model here, it
is dead for the same reason they were.**
**EXPECTED EFFECT SIZE.** **Negative in expectation.** Three independent tabular attempts have
failed by 0.18–0.21. Nothing about TabPFN addresses the reason they failed.
**EXECUTION RISK: LOW** (it is the cheapest thing here to run) but **the return is the problem.**
**COMPLIANCE.** DATA: cleanest of all — synthetic pretraining corpus, no external real data.
AUTOML: live risk, see N16 — defensible only with an explicit written defence.

---

## IF WE COULD ONLY DO ONE THING
Run **fine-tuned Presto** (S1a/b) offline, screened with the round-18 paired-delta transfer gate,
and spend a submission **only** if it clears. The adapter, the vendoring script, the masking
symmetry and the calibration path are all already written and committed — the marginal engineering
is the fine-tuning loop and a 5-seed pool. That is the only candidate in this document where the
cost is small because we already paid it in iter17.
**But the honest recommendation given 3 days, 2–3 slots and a banked 0.910837 finalist is: do NOT
spend a submission slot on this.** Run it offline for the code-review writeup, where a
rigorously-screened negative on a legal pretrained-model lane is worth real rubric points, and
where correcting our own iter17 instruments in public is exactly the kind of methodological
honesty a reviewer rewards.

---

## CAVEATS — everything INFERRED rather than VERIFIED
1. `construct_single_presto_input` / `construct_batch_presto_input`: I read them via a fetched,
   summarised rendering of `presto/dataops/utils.py`, not line-by-line source. The *existence*,
   module location, optional-modality signature and mask semantics are solid (README + `__init__.py`
   import + the fetched body). Exact argument names/defaults are NOT verbatim-verified.
2. Presto CropHarvest per-task **label counts were not verified**. I quote F1 only.
3. Presto's `latlon_embed` / `cartesian`: I have the mechanism from the fetched
   `presto/presto.py` but did not read the full body. The claim "no latlon-masking argument exists"
   is inferred from the absence of such an argument in `Encoder.forward`'s signature.
4. Galileo: band CONSTANTS confirmed at the group level from `src/data/dataset.py`, but the
   underlying `EO_SPACE_TIME_BANDS` list in `src/data/earthengine/eo.py` was **not read**, so the
   exact per-band ORDER is unverified. `patch_size=1` with `H=W=1` is **inferred** from
   `patch_size >= 1` and from Galileo's pixel-timeseries benchmark, **not** from a run. Galileo's
   licence was not checked.
5. TabPFN's missing-value encoding (-2.0, missingness channel) comes from a secondary description,
   not from reading `PriorLabs/TabPFN` source. The 10k/500 envelope is from Prior Labs' own docs.
6. Mantis's fixed input length was **not** verified against source; I only verified MOMENT's 512.
   The channel-independence property is verified for both.
7. Prithvi / Clay / SatMAE / TerraMind / DOFA / AnySat / SITS-Former: disqualified on the
   architectural description in their papers/model cards. I did not read their source. The
   disqualifier (requires spatial extent) is unambiguous enough that I am comfortable, but it is
   documentation-level, not code-level.
8. The claim that our iter17 kill used retired instruments is verified from our OWN files
   (`experiments/LB_LOG.md` iter17 + iter25/26 + round-18 block in `PROJECT_STATE.md`). The
   inference that "therefore Presto might have worked" is a claim about the INSTRUMENTS, not
   positive evidence about Presto. **Presto has never been measured on the leaderboard.** Do not
   let this document be read as saying Presto would score well.
9. `vendor/` does not currently exist in the working tree, so `run_presto.py` cannot run as-is;
   `tools/fetch_presto.py` must be run first (it clones from GitHub at `REV = "main"` — unpinned,
   which is itself a reproducibility flaw worth fixing before any code review).


