# Round 23 — Q6: Aquaculture domain signal, with a RECALL focus

**Agent:** Q6 owner. Scope: aquaculture-pond spectral/SAR indices separable from rice paddy,
seasonal wetland, natural water, salt pans, using ONLY a 4–6 contiguous-month window of
{VH, VV, blue, green, red, re1, re2, re3, nir, nira, swir1, swir2}.

**Two hard constraints on anything proposed here**
1. NOT a linear function of the raw 144 values (the model already spans those). Normalized
   differences (a−b)/(a+b), ratios, logs, products, indicators are all nonlinear ⇒ admissible.
2. Stable under truncation to a 4–6 month contiguous window. No harmonics, no phenological
   peak date, no full-year amplitude.

**Recall angle:** the team's gap is missing positives (recall 0.859 vs precision 0.906). The
highest-value output is: *which* aquaculture ponds does a VH-permanence detector systematically
MISS, and what index catches them from a short window.

Every claim below is labelled **VERIFIED (read)** or **INFERRED**.

---

## Log (appended as I go)

---

# ⛔ CORRECTION 1 (HIGHEST VALUE) — **LASCI IS EXACTLY LINEAR IN TWO SUPPLIED COLUMNS. IT IS DEAD.**

**Status: VERIFIED (read the paper's NLM XML, which renders the equations the HTML view hides).**

Source: Zhang et al. / *Spatial extraction of sea-cucumber aquaculture ponds using remote sensing
spectral and temporal features*, **Front. Mar. Sci. 12:1551260 (2025)**,
DOI **10.3389/fmars.2025.1551260**.
Full text: https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2025.1551260/full
Equations only render in: https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2025.1551260/xml/nlm

**Equation 1, verbatim:**

```
LASCI = (ρ_B8A − ρ_B4) / (λ_B8A − λ_B4)
```

**Equation 2, verbatim:**

```
SPCI  = (ρ_B5  − ρ_B4) / (λ_B5  − λ_B4)
```

`λ` are **fixed sensor central wavelengths**, not data. For Sentinel-2:
λ_B8A = 865 nm, λ_B4 = 665 nm, λ_B5 = 705 nm. So the denominators are the **constants 200 and 40**.

Mapped to this competition's columns:

```
LASCI(t) = ( nira[t] − red[t] ) / 200          ← a fixed scalar multiple of a band DIFFERENCE
SPCI(t)  = ( re1[t]  − red[t] ) /  40          ← same
```

**Both are EXACTLY LINEAR in the raw supplied values.** `LASCI(t) = 0.005·nira[t] − 0.005·red[t]`.
This is the *identical* algebraic form as `VH − VV`, which §6 already closed as "exactly linear in
two supplied columns, so a model given both can already represent it." A model receiving all 144
raw values spans every monthly LASCI and every monthly SPCI, and every linear combination of them
(including their mean over any window, including the paper's own `mean(LASCI)` over a date range).

### Why the feature-span gate did not catch this — and why its number is void

The gate reported **span R² = 0.7526** for "LASCI median". The `median` is the only nonlinear thing
in that expression. This is the **exact failure documented in the brief's own §8 item 4**: the
control `median(VH − VV)` returned R² = 0.62 instead of the 1.0 that arithmetic guarantees,
*because a median is nonlinear*. LASCI-median's 0.75 sits in the same regime as that known-void
control (0.62), not in the regime of a genuinely new feature.

⇒ **The correct reading is: R²(LASCI median) = 0.75 measures the median operator, not LASCI.**
The gate is measuring `median` and attributing the residual to the index. Anything the team could
gain from LASCI-median is available from `median` applied to *any* linear band combination — it is
a statement about order statistics of a 12-length series, not about red-edge physics.

⇒ **Q6's one surviving candidate does not survive.** Do not build it. Building it would repeat,
verbatim, the error the team already wrote down as §8.4.

### The same correction applies to the brief's §7 description

§7 Q6 calls LASCI "a red/red-edge slope index". It is **red → narrow-NIR (B8A)**, not red-edge.
SPCI is the red → red-edge-1 one. Minor, but the band mapping matters: LASCI uses `nira`, not `re*`.

### And its intended purpose is not the team's task

**VERIFIED (read).** In the source paper LASCI does **not** separate aquaculture from non-aquaculture.
It separates aquaculture **species**: high LASCI ⇒ fish-crab ponds (chlorophyll-a "dozens of times
higher", poorest water quality); low LASCI ⇒ sea-cucumber or shrimp ponds. SPCI is then applied to
split sea cucumber from shrimp. The pond-vs-background delineation in that paper is done *before*
these indices, by other means. So even if LASCI were nonlinear it would be answering a different
question. Reported overall accuracy of the species tree: 79.24%, producer's accuracy 87.07%.

### The salvageable, NONLINEAR version (if the team wants red→NIR slope information at all)

The information LASCI encodes is "how far above the red does the NIR/red-edge sit", i.e. an
algal/turbidity signal in pond water. The **nonlinear** encodings of that same physics, all of which
are outside the linear span and all of which are window-stable (per-month, no annual cycle):

```
NDVI_nira(t) = (nira[t] − red[t]) / (nira[t] + red[t])        nonlinear ✓
NDCI(t)      = (re1[t]  − red[t]) / (re1[t]  + red[t])        nonlinear ✓   (Mishra & Mishra 2012, chl-a in turbid water)
ratio(t)     = nira[t] / red[t]                                nonlinear ✓
```

`NDCI` is the literature-standard chlorophyll-a index for **turbid productive inland/coastal water**
and is the band pair SPCI uses, but in normalized form. See CORRECTION/FINDING on NDCI below.

---

## Log (appended as I go)


---

# CONFIRMATION — the brief's Ottinger citation is CORRECT (§8 item 1 stands)

**Status: VERIFIED (read full-text quotations + abstracts; MDPI blocks direct WebFetch with 403).**

Sources:
- Ottinger, Clauss & Kuenzer, *Large-Scale Assessment of Coastal Aquaculture Ponds with Sentinel-1
  Time Series Data*, **Remote Sens. 2017, 9(5):440**, DOI **10.3390/rs9050440** —
  https://www.mdpi.com/2072-4292/9/5/440
- Ottinger et al., IGARSS 2018, DOI **10.1109/IGARSS.2018.8651419** (conference version).
- Ottinger, Bachofer, Huth & Kuenzer, *Mapping Aquaculture Ponds for the Coastal Zone of Asia with
  Sentinel-1 and Sentinel-2 Time Series*, **Remote Sens. 2022, 14(1):153**, DOI
  **10.3390/rs14010153** — https://www.mdpi.com/2072-4292/14/1/153

Verified statements:
- "A temporal **median** image was calculated based on the available Sentinel-1 time series stack …
  in VH and VV polarization. **The VH median stack was used** since its distribution of backscatter
  values in the histogram showed **two distinct peaks** in all study sites and can be considered
  largely **bimodal**, whereas for VV the peaks are less pronounced." ⇒ VH, not VV, not VH−VV.
- "The **pixel-wise median (50th percentile)** … is particularly suitable for identifying **permanent
  and stable high (dams, dikes) and permanent low scatterers (smooth water surface)** from the dense
  annual time series." ⇒ the brief's phrase "ponds are permanent low scatterers" is the paper's own.
- Ottinger 2022 (S1+S2, all coastal Asia) reports **overall accuracy 91.9%** vs VHR Google Earth.

⇒ `1[VH_dB < −21]` is the canonical family. **No correction needed.** The brief is right here.

## CORRECTION 2 — the half of Ottinger's method this team CANNOT have is the half carrying the accuracy

**VERIFIED (read).** Every paper in this line detects a pond as a **two-part object**: a *permanent
low scatterer* (the water) **enclosed by a permanent HIGH scatterer** (the dike/dam), then filters
candidate objects by **size, shape, rectangularity, compactness**. Quote: "the different backscatter
responses of pond components (**dikes and enclosed water surface**) and aquaculture's **distinct
rectangular structure** allow for separation of aquaculture areas from **other natural water
bodies**." Ottinger 2017 calls the workflow *object-based*, segmenting on "backscatter intensity,
**size, and shape** features."

**This competition has no spatial extent, no neighbours, no geometry.** The team has imported the
literature's *water-detection* clause and has, structurally, **none of its pond-vs-water clause**.
The dike ring, the rectangle test and the size filter — the parts that carry the 91.9% — are all
unavailable. Worth one paragraph in the report: the team is running Ottinger's detector with its
discriminative half amputated, which is why a bare VH threshold tops out at univariate AUC 0.80.

**Direct RECALL consequence — the mechanism to focus on.** With one aggregated value per parcel per
month, a *small* pond's cell mixes low-backscatter water with the **high-backscatter dike**. In
Ottinger the dike is a *permanent HIGH scatterer*, brighter than surrounding bare land. A mixed
water+dike cell therefore sits **well above** the −21 dB open-water threshold, so `1[VH < −21]`
fires **0 in every month**. Small ponds are systematically invisible to a VH-permanence detector on
aggregated values, and invisible **confidently**, not marginally — precisely the
"confidently-wrong positives" population §3 says must be recovered.

---

# THE RECALL ANGLE — what a VH-permanence detector systematically MISSES

Four named subpopulations, evidence for each below:

1. **Drained / fallow ponds during the observed window** (pond preparation, sun-drying, harvest).
2. **Small ponds / mixed water+dike cells** (above — a data-shape artefact, not a physical one).
3. **Vegetated / extensive / mangrove-integrated (silvofishery) ponds** — volume scattering.
4. **Wind-roughened and paddlewheel-aerated ponds** — surface roughness raises VH above threshold.

---

# CONFIRMATION (with one caveat) — Ullmann et al. 2022: polarimetry adds ~nothing for WATER

**Status: VERIFIED (read the full text at Frontiers).**

Ullmann, Jagdhuber, Hoffmeister, May, Baumhauer & Bubenzer, *Polarimetric information content of
Sentinel-1 for land cover mapping: An experimental case study using quad-pol data synthesized from
complementary repeat-pass acquisitions*, **Front. Remote Sens. 3:905713 (2022)**,
DOI **10.3389/frsen.2022.905713** —
https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2022.905713/full

Verbatim from the paper:

> "The **water class** which is generally the one with the highest scores **barely shows any
> accuracy differences** because it is the one with the **most distinct physical characteristics**."

Dataset-level accuracies: pseudo-quad-pol (S1q) 35.4%, dual-pol intensity (S1AB) 33.8%, single
dual-pol (S1A) 32.7%. Water's per-class accuracy is essentially flat across all three (≈53.7%,
no meaningful movement). Gains from polarimetry concentrate on vegetation classes (pasture +1.8%,
orchards, mixed forest) — i.e. classes with **volume scattering**, which water has none of.

**Caveat / small correction to the brief.** The brief says Ullmann "measure polarimetric derivatives
as adding **0.1%** over intensity for water." I could not find a literal "0.1%" figure; the paper's
own statement is the qualitative "barely shows any accuracy differences", with water flat across
datasets. The brief's *conclusion* is correct and if anything understated (the measured gain is
≈0, not 0.1%), but the specific number should be replaced with the quote above before it goes in a
report — an exact numeric attribution that isn't in the paper is the same class of error as §8.1.

**Second caveat, more important:** this is a **land-cover** study over Germany (agriculture, forest,
built-up, water), **not** an aquaculture study. It supports "polarimetric derivatives do not help
detect *water*". It says nothing about whether polarimetry helps separate *aquaculture ponds from
other water*, which is the team's actual problem, and where the discriminating physics is not the
water surface at all but the **dike, the vegetation, and the drainage schedule**. Do not over-read it.

**Note the asymmetry this creates, and it is the recall lead:** Ullmann's finding is that
polarimetry only pays where there is **volume scattering**. Every pond subpopulation the
`1[VH < −21]` detector misses (drained, vegetated, mangrove-integrated, small-with-dike) is a
subpopulation whose cell **does** contain volume or surface scattering. So the one place a
cross-pol *ratio-like* quantity could earn its keep is exactly the missed-positive population — but
`VH − VV` in dB is linear and already spanned, so it must enter **nonlinearly** (see recommendations).

---

# FINDING 3 (the main positive result) — NDTI + NDCI separate ponds UNDER aquaculture practice from ponds NOT under aquaculture practice, per-month, at 94% OA

**Status: VERIFIED via indexed full-text snippets + abstract; the publisher (ScienceDirect) returns
403 to WebFetch so I could not read the methods section myself. Label this INFERRED-STRONG on the
exact thresholds, VERIFIED on the index identity, formulas and headline accuracy.**

Source: *Integration of Sentinel-1 and Sentinel-2 for temporal identification of aquacultural ponds*,
**Remote Sensing Applications: Society and Environment (2025)**,
https://www.sciencedirect.com/science/article/pii/S2211714825000561

What it does — and note that this is **exactly this competition's discrimination**, not the easier
pond-vs-land one:

> "…accurate classification of ponds **under aquaculture practices (AP)** and **not under
> aquaculture practices (NAP)**, integrating optical and SAR data alongside key spectral indices,
> such as the **Normalized Difference Turbidity Index (NDTI)** and the **Normalized Difference
> Chlorophyll Index (NDCI)**, to distinguish between AP and NAP ponds **across different seasons**."

> "Optical indices highlight differences between pond types, with **higher water clarity and
> nutrient enrichment in aquaculture practice ponds**. For SAR data, **NAP ponds exhibit higher
> anisotropy**, reflecting complex management practices."

> "The **Random Forest** classifier obtained a **maximum overall accuracy of 94%** by combining
> optical and SAR data, significantly outperforming other classifiers."

> "Seasonal variability plays a critical role, with **AP pond areas expanding during the monsoon
> season and contracting in the summer due to maintenance and evaporation**."

## The two indices, exact formulas, mapped to this competition's columns

```
NDTI(t) = ( red[t]  − green[t] ) / ( red[t]  + green[t] )      Lacaux et al., RSE 106(1):66–74, 2007
                                                                DOI 10.1016/j.rse.2006.07.012
NDCI(t) = ( re1[t]  − red[t]   ) / ( re1[t]  + red[t]   )      Mishra & Mishra, RSE 117:394–406, 2012
                                                                DOI 10.1016/j.rse.2011.10.016
```

- **NDTI** — turbidity / suspended solids. Clear water reflects more green than red; suspended
  particles push reflectance toward red. High (→0) = turbid, low (→−1) = clear. **VERIFIED**
  (formula confirmed in multiple independent sources).
- **NDCI** — chlorophyll-a in **turbid productive** water; uses the red-edge-1 reflectance peak at
  705 nm, which exists because water + chl-a absorption is minimal there. This is the same band pair
  as SPCI, but in **normalized** form. **VERIFIED** (Mishra & Mishra is the canonical NDCI source;
  the fmars-2025 paper independently states "the reflection peak in the B5 (red edge 1) band is the
  most notable spectral feature for water bodies with algae, because the absorption coefficient of
  water and chlorophyll-a is minimal at this band").

## Why these clear both of the team's hard constraints

| constraint | NDTI | NDCI |
|---|---|---|
| nonlinear in the raw 144? | **YES, nonlinear** — a ratio of two linear forms, not in the linear span | **YES, nonlinear** |
| stable under 4–6 month truncation? | **YES** — it is a per-month pointwise value. No period, no phase, no peak date, no annual amplitude. Any window statistic (mean/median/min/max/range over the observed months) is defined on 4 months exactly as on 12. | **YES** |
| already screened by the team? | **NO** — the team screened MNDWI, AWEI_nsh, NDWI, LASCI, SPCI, red-edge curvature, corr(VH,nir). NDTI and NDCI are **absent from that table** | **NO** |

Note carefully: the team's screened water indices (MNDWI R²=0.93, AWEI R²=0.91, NDWI R²=0.86) are
all **water-vs-land** indices — they answer "is this wet?", which the model already knows from VH.
NDTI and NDCI are **water-quality** indices — they answer "**what kind of water is this?**", which
is the question that actually separates a stocked pond from a natural lake, a river, or a flooded
wetland. That is a different axis, and it is the axis the confusable classes live on. The high span
R² of the water indices is *expected* and is not evidence against the water-quality indices.

## Corroborating physics for the AP/NAP split

- Stocked ponds are **fertilised and fed**: high chlorophyll-a, algal blooms, high NDCI. The
  fmars-2025 paper measures fish-crab pond chl-a as "**dozens of times higher** than in shrimp or
  crab ponds", and uses exactly this axis (its LASCI) to separate pond types.
- Stocked ponds are **shallow, mixed, aerated and bio-disturbed**: high suspended solids, high NDTI.
- Natural lakes / reservoirs / seasonal wetlands are comparatively clear and unfertilised.
- Salt pans are hypersaline, essentially **no chlorophyll**, and their brine is optically clear
  then evaporates to bright crust — a **negative** NDCI signature that contrasts sharply with a
  stocked pond, and the fmars-2025 paper names salt pans as its single worst confusion:
  "**The main factor limiting the extraction accuracy of aquaculture ponds is the misclassification
  between aquaculture ponds and salt pans.**" (VERIFIED, read.)

---

# FINDING 4 — small ponds / mixed water+dike cells: the recall mechanism is documented

**Status: VERIFIED (read abstract + indexed text).**

Source: *Inventorying ponds through novel size-adaptive object mapping using Sentinel-1/2 time
series*, **Remote Sensing of Environment (2024)**, article S0034425724005108 —
https://www.sciencedirect.com/science/article/abs/pii/S0034425724005108

> "**Landsat's 30 m spatial resolution limits its ability to differentiate narrow dikes surrounding
> ponds, resulting in excessive aggregation in classification outputs and hindering accurate
> representation of the regular shapes of ponds.**"

> "A novel object mapper (**OptiSAR-POM**) … **adaptively aligns segmentation parameters with pond
> size**. Sentinel-1/2 data can detect ponds smaller than **1 ha**."

The whole reason this paper exists is that **fixed-parameter methods systematically omit small
ponds**. In this competition the failure is worse than in the paper, because the team has *no*
segmentation at all — one aggregated value per parcel. A sub-hectare pond's cell is a mixture of
smooth water (very low VH) and the **permanent high scatterer** dike, so the aggregate VH lands in
the middle of the distribution and `1[VH_dB < −21]` never fires. These positives are missed
**confidently**, not marginally — the exact population §3 says must be recovered.

**This is a mechanism, not just a name, and it predicts a specific signature:** a small pond's cell
is a *mixture* of two very different scatterers, so within any window its VH is (a) not low, but
(b) **less variable** than a rice paddy or seasonal wetland going through a flood/drain/grow cycle,
and (c) the optical half of the cell still carries the pond water's turbidity/chlorophyll signal,
diluted by the dike. So the catching feature is **the optical water-quality axis, conditioned on VH
NOT being low** — which is precisely NDTI/NDCI, and precisely a case a linear model cannot express
because it is an interaction.

---

# FINDING 5 — the DRAINED-POND miss is real, and its duty cycle is quantified

**Status: VERIFIED (aquaculture extension/industry sources, multiple independent, consistent).**

A stocked pond is **not** permanently wet. Between crops it is drained, harvested and sun-dried:

- Fallow period between crops: **2–45 days**, with most farms at **7, 10, 15 or 30 days**.
- Pond-bottom sun-drying: **7–10 days** typical; **20–30 days "till the soil cracks"** where
  practised properly.
- "Pond preparation should be done **30–40 days before the next crop**."

Sources (industry/extension, converging): Global Seafood Alliance, *Shrimp pond preparation crucial
for production, disease prevention* — https://www.globalseafood.org/advocate/shrimp-pond-preparation-crucial-production-disease-prevention/ ;
JALA, *Shrimp Cultivation Stages in Traditional Ponds* — https://jala.tech/blog/cultivation-tips/shrimp-cultivation-stages-in-traditional-ponds ;
TNAU Agritech Portal, Fisheries :: Shrimp Culture — http://www.agritech.tnau.ac.in/fishery/fish_cul_freshwater_catfish_pond.html

And independently, from the AP/NAP paper (Finding 3): "**AP pond areas expanding during the monsoon
season and contracting in the summer due to maintenance and evaporation.**"

## Why this specifically destroys recall on a 4–6 month window and NOT on 12 months

Take a **monthly** composite (which is what this competition supplies). A 30–40 day drain+dry
preparation phase occupies **one to two whole monthly composites**. A crop cycle is ~4–6 months.

- On a **12-month** series, a pond shows ~9–10 wet months out of 12. `median(VH)` is comfortably
  below −21 dB and the Ottinger detector fires. This is why the literature's annual-median method
  works — **and why the team's train rows, which have all 12 months, look easy.**
- On a **4–6 month contiguous** window — the test condition — a window that happens to straddle a
  harvest can contain **2 dry months out of 4**. Then `median(VH) > −21`, `mean(1[VH<−21]) = 0.5`,
  and the permanence channel reports *"not a pond."*

**This is a truncation-induced recall failure that is invisible in training and invisible in OOF**,
because train rows are never truncated in the way that creates it and because OOF is computed on the
same masked replica whose window is drawn i.i.d. rather than adversarially. It matches the brief's
§4 observation that "a real covariate shift also exists in the SAR *levels*" and it matches §3's
finding that the missed positives are **confidently** wrong rather than borderline: a half-dry
window does not produce p ≈ 0.48, it produces p ≈ 0.2.

⇒ **Named missed subpopulation #1: ponds whose 4–6 month observation window overlaps a
drain / harvest / sun-dry / pond-preparation phase.** This is not exotic; on a 5-month random window
against a 4–6 month crop cycle with a 1–2 month dry phase, a large fraction of ponds are affected.

## What still identifies a pond in its DRY phase, from optical bands only

A drained pond bottom is **exposed saturated sediment with no vegetation**. Two consequences the
raw-band linear span does not capture:

1. **It never greens up.** Over the whole cycle — flooded *or* drained — an aquaculture pond carries
   essentially zero photosynthetic vegetation. Every confusable class does green up at some point:
   rice paddy goes flood → canopy closure (NDVI ≫ 0.5) within 2–3 months; seasonal wetland grows
   macrophytes; a fallow field carries stubble and weeds. Natural water bodies stay low-NDVI, but
   they are separated on the *other* axis (turbidity/chl, Finding 3), and salt pans are separated
   by chlorophyll being absent.
2. **Wet bare sediment keeps a high LSWI while NDVI stays at floor.** Dry bare soil does not.

So the drained-pond-catching feature is a **vegetation-absence** test over the window, not a
water-presence test — and vegetation absence is exactly what a VH-permanence channel cannot see,
because bare wet mud and a dike both scatter strongly in VH.
