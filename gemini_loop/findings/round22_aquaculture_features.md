# Round 22 — REFUTING "our feature bank is exhausted"

Target claim under refutation: *"No remaining physically-motivated feature computable from a
12-band x 12-month single-pixel time series would separate aquaculture ponds from confusers."*

Constraints recap (do not violate): bands = `VH, VV, blue, green, nir, nira, re1, re2, re3, red,
swir1, swir2`; 12 months; one location per row; test rows show only **4-6 contiguous months**;
metric 0.6*F1(@0.5) + 0.4*AUC; no external data.

## STATUS: COMPLETE

---

## 1. LITERATURE LOG (raw notes, appended as found)

### L1. Ottinger, Clauss & Kuenzer (2017) "Large-Scale Assessment of Coastal Aquaculture Ponds
with Sentinel-1 Time Series Data", *Remote Sensing* 9(5):440. https://www.mdpi.com/2072-4292/9/5/440
- Search-snippet level (to be verified by fetch): "Dense SAR time series are needed to distinguish
  aquaculture ponds, as relatively stable water bodies, from temporary water bodies. This is a major
  issue for many coastal areas and specifically river deltas, where floodplains or paddy rice fields
  might be confused with aquaculture if the temporal resolution of the time series is inadequate to
  depict hydrological characteristics and seasonality of land cover other than aquaculture."
- **KEY IMPLICATION FOR US**: the discriminant the literature actually names is *water PERSISTENCE /
  temporal stability*, not shape, and not the VH/VV ratio. Ponds = stable water; paddy/floodplain =
  temporary water. That is a purely temporal, per-pixel quantity. It is the single most-repeated
  non-morphological discriminant in this literature.
- Also: "By averaging over time, the temporal median reduces the speckle noise significantly" — i.e.
  the literature's own answer to single-date SAR noise is TEMPORAL AGGREGATION, which is exactly the
  operation available to us and NOT available to a per-month raw feature.

### L1b. Ottinger/Kuenzer methodology thread (search-level, verify)
- "The researchers used the **VH median stack** since its distribution of backscatter values in the
  histogram showed two distinct peaks in all study sites and can be considered largely bimodal, and
  in a later step, they applied a water threshold algorithm which is suited for the separation of
  bimodal distributions." (Otsu)
- "Annual stacks were reduced to new images representing the **annual median and 95th percentile**
  for all four water indices, and OTSU's threshold was then computed and applied to each reduced
  image to delineate water bodies from the land surface."
- **NOTE FOR THE VH-VV NULL**: the canonical SAR pond-mapping pipeline uses **VH ALONE, temporally
  reduced (median / 95th pct)** — NOT the VH/VV ratio. The "most-cited aquaculture SAR feature" is
  actually *low VH backscatter, temporally aggregated*, not the dual-pol ratio. Our closed lane may
  have killed the wrong feature.

### L1c. VERIFIED FULL TEXT — Ottinger, Bachofer, Huth, Kuenzer, "Time Series Sentinel-1 SAR Data
for the Mapping of Aquaculture Ponds in Coastal Asia", IGARSS 2018, pp. 9370-9373,
DOI 10.1109/IGARSS.2018.8651419. PDF: https://elib.dlr.de/126751/1/08651419.pdf
VERBATIM (own extraction from the PDF, p.9372):
> "For the study sites we used scenes in **VH polarization** and only from **an ascending or
> descending orbit to reduce effects of orbit direction and look angle**. Since the availability of
> scenes in ascending and descending orbits varies greatly from area to area we chose only the orbit
> direction which is more available for the specific study site."

> "The **pixel-wise median** was calculated for the pre-processed time series data cube to
> **reduce speckle noise** in the intensity SAR imagery and identify **permanent and stable low
> scatterers**. By averaging over time, the median of the temporal data cube effectively improved
> the recognition and detection of small and narrow surface structures such as dams and levees
> surrounding aquaculture ponds."

> "A connected component segmentation algorithm was applied to the temporally filtered SAR time
> series data to extract pond objects **based on shape and size features** with a mean overall
> accuracy of 0.83."

**THREE HARD IMPLICATIONS FOR US:**
1. The canonical pipeline is **VH only**, never the VH/VV ratio. Our "single most-cited aquaculture
   SAR feature" premise is wrong: the most-cited feature is *temporally-median-filtered VH*.
2. The literature **controls incidence angle by construction** (single orbit direction). Our rows
   almost certainly mix orbits/look angles across the 12 months and across locations. A ratio of two
   co-registered, same-look-angle channels is *less* incidence-sensitive than either alone — so if
   the ratio was null, incidence angle is probably NOT the reason. See section 3 for the real one.
3. Their 0.83 accuracy comes from **shape and size**, which we do not have. This is exactly why the
   published headline features do not transfer, and why the *temporal* residual is where our
   remaining signal must live.

### L2. Search-level note, multi-feature fusion (IJRS 2025, 46(24), Sentinel-1+2 time series)
- "The inclusion of environmental and temporal characteristics improves classification accuracy by
  6% and enhances the ability to identify water bodies with shapes and structures similar to
  aquaculture ponds." — temporal characteristics carry incremental signal ON TOP OF shape, and
  specifically help against shape-confusable water bodies. To verify.


### L3. VERIFIED FULL TEXT — Ottinger, Liu, Ullmann, Huth, Kuenzer, Bachofer (2026, in press),
"Pond aquaculture dynamics in Asia: Satellite time series for analyzing the spatio-temporal
development of coastal aquaculture", **Aquaculture 610, 742940**.
PDF: https://elib.dlr.de/215700/1/Ottinger%20et%20al.%20(2025)_Pond%20aquaculture%20dynamics%20in%20Asia.pdf
This is the most directly useful paper found. It runs an explicit **ROC benchmark of four water
indices x two temporal reducers** for detecting *pond water*, which is very close to our task.

VERBATIM (Section 3.2.1, own extraction):
> "In a next step, all images were aggregated at the pixel-level using a reducer. To account for
> potential outliers during the ROC test, **the median operator was selected as the reducer method.
> To enhance the detection of ponds intermittently covered by water throughout the year, the 95th
> percentile was also employed as a secondary reducer** for the test."

VERBATIM (Section 3.2.1 results of the ROC test):
> "From the ROC graph, it is evident that **AWEI and WIFI were more effective water indices than
> NDWI or MNDWI**. Regarding the reducer, we see that **the median performed better than the 95th
> percentile**. The relationship between the thresholds associated with 'median AWEI' and 'median
> WIFI' and the distance from their points on the ROC curve to the upper-left boundary indicates
> that **the WIFI median has the best combination of water index and reducer**."

VERBATIM (their Table 3, index equations, Landsat band names — note these are EXACTLY our band
names blue/green/red/nir/swir1/swir2):
> NDWI (McFeeters, 1996):  (Bgreen - Bnir) / (Bgreen + Bnir)
> MNDWI (Xu, 2006):        (Bgreen - Bswir1) / (Bgreen + Bswir1)
> AWEI (Feyisa et al., 2014): Bblue + 2.5 x Bgreen - 1.5 x (Bnir + Bswir1) - 0.25 x Bswir2
> WIFI (Fisher et al., 2016): 1.7204 + 171 x Bgreen + 3 x Bred - 70 x Bnir - 45 x Bswir1 - 71 x Bswir2

(The AWEI form above is AWEI_nsh, the no-shadow variant of Feyisa et al., RSE 140:23-35, 2014.
WIFI = "WI2015" of Fisher, Flood & Danaher, RSE 175:167-182, 2015/2016; coefficients assume surface
reflectance scaled to 0-1 and the water threshold is 0.)

VERBATIM (pond activity definition — this is a *temporal* labelling rule, not a shape rule):
> "This method allows us to assess aquaculture pond presence over time for the reference pond
> dataset by checking how these water masks align with the reference dataset, **assuming that active
> ponds are covered by water during the Landsat observation period**."

**WHY THIS REFUTES "FEATURE BANK EXHAUSTED":**
- We are told our closed lanes include SAR ratios and tree models, but there is no record of us
  having tried **AWEI_nsh or WIFI/WI2015**. The one paper that actually *benchmarks* water indices
  for POND water ranks them **AWEI > MNDWI/NDWI** and **WIFI best of all**. AWEI and WIFI are
  multi-band linear combinations using blue+green+red+nir+swir1+swir2 — all six are in our column
  set. They are NOT monotone functions of NDWI or MNDWI, so they are not trivially recoverable.
- Crucially the winner is a **temporally reduced** index (median over the year), i.e. exactly the
  kind of order-statistic a Transformer with masking does *not* automatically compute.

### L4. **THE STRONGEST SINGLE REFUTATION** — Frontiers in Marine Science 12:1551260 (2025),
"Spatial extraction of sea-cucumber aquaculture ponds using remote sensing spectral and temporal
features". DOI 10.3389/fmars.2025.1551260
https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2025.1551260/full

This paper classifies **pond TYPE** (sea-cucumber vs shrimp vs fish-crab) using **only per-pixel
red / red-edge spectral slopes and their monthly temporal behaviour** — no shape, no texture, no
neighbourhood. That is *precisely* our data shape, and it is a harder discrimination than
pond-vs-not-pond.

Two indices, both built from bands we HAVE:
- **LASCI** ("Land-based Aquaculture Species Classification Index"): the spectral **slope between
  B4 (red, 665 nm) and B8A (narrow NIR, 865 nm)**. In our columns: `red` and `nira`.
- **SPCI** ("Sea Cucumber and Prawn Classification Index"): the spectral **slope between B4 (red,
  665 nm) and B5 (red-edge 1, 705 nm)**. In our columns: `red` and `re1`.

Decision rules are **temporal statistics of those slopes**:
- T1 = 0.11 on the **mean LASCI over Mar 1-31 and Jul 9-Nov 26** — separates fish-crab ponds
  (>0.11) from sea-cucumber/shrimp ponds (<0.11).
- T2 = 0.06-0.08 on the **SPCI slope during Mar 1 - Apr 10** (a declining phase).
- T3 = -0.01 on the **SPCI slope during Apr 10 - Jul 29** (a rapidly increasing phase).
- Sea-cucumber ponds show "**slow decline followed by rapid increase**" in SPCI.
Accuracy: OA 79.24 %, PA 87.07 %, UA 74.81 %, **F1 = 0.81** — on pond *type*, from per-pixel
red-edge time series alone.

Mechanism given: sea-cucumber ponds "have better water quality with **lower chlorophyll-a
concentrations**, reflected in negative LASCI values throughout March-November, unlike fish-crab
ponds."

**WHY THIS KILLS THE "EXHAUSTED" CLAIM.** We hold three red-edge bands plus a narrow NIR and,
per the brief, are not exploiting them. The published mechanism is water-column **chlorophyll-a and
turbidity driven by feed/fertiliser loading** — an aquaculture-SPECIFIC biogeochemical signal that
natural lakes, reservoirs, salt pans and clean seasonal wetlands do not have, and that rice paddies
express with completely different timing (canopy, not water column). This is a physically-motivated
feature family, computable per pixel per month, that our closed-lane list does not cover.

### L4b. EXACT EQUATIONS from L4 (fetched from the NLM XML of the same paper)
> "LASCI = (rho_B8A - rho_B4) / (lambda_B8A - lambda_B4)"
> "SPCI  = (rho_B5  - rho_B4) / (lambda_B5  - lambda_B4)"
where rho = surface reflectance and lambda = central wavelength (nm).
Central wavelengths (Sentinel-2A): B4 = 664.6 nm, B5 = 704.1 nm, B6 = 740.5 nm, B7 = 782.8 nm,
B8A = 864.7 nm. So the denominators are constants: (864.7-664.6)=200.1 nm and (704.1-664.6)=39.5 nm.
**In our column names:**
  `LASCI = (nira - red) / 200.1`
  `SPCI  = (re1  - red) / 39.5`
(The constant denominator means these are, up to scale, simple band DIFFERENCES `nira - red` and
`re1 - red` — see the honest redundancy discussion in section 8; the *value added* is the temporal
statistics taken over them, and the fact that the divisor differs between them so that a *ratio*
LASCI/SPCI is a genuine spectral-curvature feature.)

Physical mechanism, verbatim from the same paper:
> "Sea cucumbers require minimal artificial feed" and benefit from "better water quality," whereas
> "fish and shrimp farming ... significantly increase ... organic matter ... creating potential
> risks of eutrophication."
> "fish-crab aquaculture has higher LASCI values ... corresponding to the poorest water quality."
> "From June to September, fish have high feeding rates ... levels of nutrients ... increase."
Analysis window: "March 1 to November 30" when "ponds generally remain in shallow water and do not
freeze."

### L5. Pond dikes are stable through drying — Ocean & Coastal Management (2026),
"Mapping coastal aquaculture ponds in China without dependence on water levels: New insights from
pond dikes derived from Sentinel-1/2 imagery", https://doi.org/10.1016/j.ocecoaman.2026.107... 
(ScienceDirect S0195925526000806). Search-level quotes, to verify if used:
> "Pond dikes are artificially constructed in long strips, which remain stable across various water
> levels, even during a drying phase."
> "Misclassification is mainly due to **the drainage of aquaculture ponds during non-farming
> seasons** and the coarse resolution of satellite images. Additionally, aquaculture ponds at
> **low water levels** or sheltered ponds are often omitted in detection approaches."
**IMPLICATION**: this is a *named failure mode of the water-persistence feature*. Ponds are NOT
always wet — they are drained between crops. So "always wet" is the wrong feature; "wet, then
abruptly dry, then abruptly wet again, on a management (not climatic) schedule" is the right one.
It also warns us that a plain "fraction of months wet" feature will systematically miss the drained
positives, which may be exactly the population our F1 is losing at the 0.5 cut.

### L6. **SECOND STRONG REFUTATION** — Peng, Sengupta, Duan, Chen & Tian (2022), "Accurate mapping
of Chinese coastal aquaculture ponds using biophysical parameters based on Sentinel-2 time series
images", **Marine Pollution Bulletin 181, 113901**.
https://www.sciencedirect.com/science/article/abs/pii/S0025326X22005835
- The paper's whole premise is our problem: "misclassification due to **similar geometric
  characteristics of various water bodies**" — i.e. shape FAILS to separate ponds from other water.
- Their fix: integrate spatial characteristics with **three biophysical parameters —
  Chlorophyll-a, Trophic State Index (TSI), and Floating Algae Index (FAI)** — derived from
  bio-optical models on Sentinel-2 time series. Overall accuracy 91 %, Kappa 0.83, national scale.
- **This is the same lane as L4 and it is independently validated at national scale.** The
  discriminant is the *water column's trophic state*, which is a direct consequence of aquaculture
  management (feed, fertiliser, aeration, stocking density). Natural lakes, reservoirs, salt pans
  and clean wetlands sit elsewhere in Chl-a/TSI space; rice paddies have a canopy, not a
  phytoplankton bloom.
- FAI (Hu 2009, RSE 113:2118-2129) is band-computable from what we hold:
  `FAI = nir - [ red + (swir1 - red) * (lambda_nir - lambda_red)/(lambda_swir1 - lambda_red) ]`
  With Sentinel-2 centres red=664.6, nir=832.8, swir1=1613.7 nm the interpolation weight is
  (832.8-664.6)/(1613.7-664.6) = 0.1772, so
  `FAI = nir - red - 0.1772 * (swir1 - red)`.
  **All three bands are in our column set.** FAI is explicitly designed to be robust to aerosol and
  thin cloud, which matters for a monthly-composited product.

### L7. Pond management calendar — timescales (this is the DISAPPOINTING but important finding)
Sources: FAO, "Shrimp culture: pond design, operation and management"
https://www.fao.org/4/ac210e/AC210E06.htm ; Global Seafood Alliance / Responsible Seafood Advocate,
"Shrimp pond preparation crucial for production, disease prevention"
https://www.globalseafood.org/advocate/shrimp-pond-preparation-crucial-production-disease-prevention/
- Grow-out cycle: roughly **3-6 months for penaeid shrimp**, up to **8-11 months** in some
  freshwater systems; ponds "are typically drained after 8 to 11 months to harvest".
- **Drying: "1-2 weeks, depending on weather conditions"**; "thorough dry-out for a week or longer".
- **Fallow between crops: 2-45 days; "the majority of farms having fallow periods of either 7 days,
  10 days, 15 days or 30 days."**
**HARD CONSEQUENCE FOR OUR DATA.** Our observations are **monthly composites**. A 7-15 day dry-down
inside a 30-day compositing window is, at best, a partial-strength dip in one month, and at worst
completely averaged away — and if the composite is a cloud-free *median* or *maximum-NDVI* pick, the
dry days may be dropped entirely. **Therefore: "count of dry-down events per year" and "sharpness of
the wet-to-dry transition" — the two features one would design first from the management-cycle story
— are largely destroyed by monthly compositing.** I state this against interest: it is the single
best physical argument that part of the temporal feature bank really IS closed to us, and it
probably explains additional nulls. What survives is the *slow* component: the multi-month grow-out
phase, the seasonal fill/drain of extensive systems, and the water-quality trajectory (L4/L6),
whose timescale is months, not days.

### L8. NDCI — Mishra & Mishra (2012), "Normalized difference chlorophyll index: A novel model for
remote estimation of chlorophyll-a concentration in turbid productive waters", **Remote Sensing of
Environment 117:394-406**, DOI 10.1016/j.rse.2011.10.016
- "the NDCI model utilizes MSI spectral bands located at **665 nm and 705 nm**, which are maximally
  sensitive to chl-a absorption, and backscattering induced reflectance, respectively."
- `NDCI = (rededge - red) / (rededge + red)` = **in our columns `(re1 - red)/(re1 + red)`**.
- Designed explicitly for **turbid productive (Case-2) waters** — which is exactly what an
  aquaculture pond is and what a clean natural lake or a salt pan is not.
- Note this is the *normalised* sibling of the SPCI of L4. Having both the normalised (NDCI) and
  the slope (SPCI) forms is not redundant in the presence of varying illumination/atmosphere.

### L9. Salt pans and other water bodies — the literature says SEASONALITY is the discriminant
From the coastal-aquaculture mapping literature (Jiangsu 1985-2025 study, *Remote Sensing* 18(11):
1782, https://doi.org/10.3390/rs18111782 ; and the STF-RF framework literature). Search-level
verbatim, flagged as unverified-by-fetch:
> "Since the **sizes, shapes, and spectra of coastal aquaculture ponds and other surface water
> bodies such as salt pans are highly similar**, employing simple object features extracted from a
> single image without considering **their different characteristics resulting from seasonal
> changes** is insufficient for separating them and may result in misclassification."
> "**Temporal filtering removes transient water bodies** through intra-annual time series
> construction, **temporal aggregation and frequency analysis**, while discrimination of aquaculture
> ponds and other water bodies involves morphological features like compactness, rectangularity and
> LSI; textural features like GLCM contrast and local variance..."
**READ THIS CAREFULLY — it is the cleanest statement of our situation in the literature.** The
published pipeline has TWO stages: (a) *temporal aggregation + frequency analysis* to reject
transient water, then (b) *morphology + texture* to reject the remaining shape-confusable water.
**We have stage (a) and not stage (b).** So the claim "the pond literature is heavily morphological
and that is unavailable to us" is true only of stage (b). Stage (a) — water frequency, temporal
aggregation, transient-water rejection — is fully available to us and is where we should be
spending. It is also *exactly* the family (order statistics / counts) that section 3.6 identifies as
NOT already in the Transformer's span.
Also useful: STF-RF (Spectral-Temporal Filtering + Random Forest) on Landsat time series reaches
">90 % classification accuracy" distinguishing aquaculture ponds from spectrally similar land cover.

### L10. Also worth naming: Zhang et al., "Interannual changes of coastal aquaculture ponds in China
at 10-m spatial resolution during 2016-2021", **Remote Sensing of Environment 283 (2022) 113301**
(ScienceDirect S0034425722004539); and Duan et al., "A Large-Scale Deep-Learning Approach for
Multi-Temporal Aqua and Salt-Culture Mapping", *Remote Sensing* 13(8):1415,
https://doi.org/10.3390/rs13081415 — the latter is explicitly a **multi-temporal** aqua-vs-salt
discrimination, confirming salt culture is treated as a distinct temporal class, not a spectral one.

### L11. Rice phenology — Xiao et al. (2005, 2006), the LSWI/EVI crossing rule
Xiao, Boles, Liu, Zhuang, Frolking, Li, Salas & Moore (2005), "Mapping paddy rice agriculture in
southern China using multi-temporal MODIS images", **Remote Sensing of Environment 95(4):480-492**.
Xiao et al. (2006), RSE 100(1):95-113 (SE Asia).
> "LSWI (land surface water index) centered at 1,640 nm was used for identifying water properties
> in the flooding and transplanting stages of paddy rice with the condition of **LSWI + T >= NDVI or
> LSWI + T >= EVI** (Xiao et al., 2005; 2006)."
> "When paddy rice fields are flooded and transplanted, there is a **temporary inversion** of the
> vegetation indices in which **LSWI values either approach or exceed the NDVI or EVI values**; this
> can be characterized as the flooding/transplanting signal in paddy rice fields."
> "For 500-m spatial resolution MODIS images, a confidence interval of 5 % was used ...
> **LSWI + 0.05 >= EVI or LSWI + 0.05 >= NDVI**."
Formulas: `LSWI = (nir - swir1)/(nir + swir1)`; `NDVI = (nir - red)/(nir + red)`;
`EVI = 2.5*(nir - red)/(nir + 6*red - 7.5*blue + 1)`. **All bands present in our columns.**
**WHY THIS MATTERS TO US.** Rice paddies are our nastiest confuser and Xiao's rule is the
canonical, purely per-pixel, purely temporal rice detector. The rice signature is a **transient
1-2 month inversion followed by rapid greening** — a *crossing event with a specific temporal
shape*. An aquaculture pond stays in the inverted (water-dominant) state and **never greens up**.
So the discriminating quantity is not "is LSWI>EVI now" but "**how many months is LSWI>EVI, and does
a high-EVI canopy phase follow within 1-3 months**". That is a run-length / ordering feature — again
in the class section 3.6 says the Transformer does not get for free.

### L12. **HONEST COUNTER-EVIDENCE — this cuts against the temporal-percentile proposal**
Zhang, Roy et al. (2024), "Classifying raw irregular time series (CRIT) for large area land cover
mapping by adapting transformer model", **Science of Remote Sensing 9:100123**,
https://www.sciencedirect.com/science/article/pii/S2666017224000075
> "CRIT ... directly classifies irregular good-quality surface reflectance time series **without any
> composite or temporal percentile derivation** by adapting Transformer."
> "Results showed that the CRIT trained with three years of samples had 1.4-1.5 % higher overall
> accuracies with less computation time than classifying 16-day composites and **2.3-2.4 % higher
> than classifying temporal percentiles**."
> "The CNN was not as good as CRIT in classifying the raw irregular time series as CNN simply
> filling temporal positions with no observations as zeros while the **CRIT used a masking mechanism
> to rule out their contribution**."
**READ**: at large n, a masked Transformer on the raw irregular series BEATS temporal percentiles by
2.3-2.4 pp. Our architecture is already the winner of that comparison, and our +0.013 from relative-
time encoding is the same finding (CRIT feeds day-of-year as input). So: **do NOT propose replacing
the raw sequence with percentile summaries.** Propose them only as *auxiliary* channels, and expect
a small effect. This is the strongest single argument FOR your "exhausted" claim and I will not
pretend otherwise. The counter-argument is regime, not principle: CRIT had millions of Landsat
training pixels and no domain shift; we have n=1817 and adversarial AUC 0.99, where low-variance
hand-built summaries are exactly the kind of inductive bias that pays.

### L13. Also relevant, same direction:
- "Linearly interpolating missing values in time series helps little for land cover classification
  using recurrent or attention networks", **ISPRS Journal of Photogrammetry and Remote Sensing**
  (2024), https://www.sciencedirect.com/science/article/pii/S0924271624001813 — do not gap-fill our
  missing months; masking is correct and we already do it.
- "Paving the way toward foundation models for irregular and unaligned Satellite Image Time Series",
  arXiv:2407.08448 — relevant because pretrained weights are legal in this competition.

### L14. **THIRD INDEPENDENT CONFIRMATION of the water-quality lane** — Hou, Xu et al. (2022),
"Improving Satellite Retrieval of Coastal Aquaculture Pond by Adding Water Quality Parameters",
**Remote Sensing 14(14):3306**, https://doi.org/10.3390/rs14143306
> "There are other water bodies with **similar morphology (e.g., saltworks, rice fields, and small
> reservoirs) that are difficult to distinguish from aquaculture ponds**, causing a lot of
> omission/commissioning errors in areas with complex land-use types."
> "six transfer characteristics **including water quality characteristics improved the accuracy of
> distinguishing aquaculture ponds from salt pans, rice fields, and wetland parks**, which typically
> had **F1 scores > 85 %**."
**Note the confuser list is exactly ours** (saltworks, rice fields, small reservoirs, wetland parks)
and the paper states plainly that **morphology does not separate them** — water quality does. Three
independent groups (L4 Frontiers 2025, L6 MarPolBull 2022, L14 RemSens 2022) converge on the same
conclusion. This is the single most robust finding of this round.

## 2. WHAT SEPARATES PONDS FROM CONFUSERS — TEMPORAL MECHANISMS

Synthesis of L1-L14. Five mechanisms, in descending order of how much survives our constraints.

**M1 — Water-column trophic state (STRONGEST, and we are not using it).** Aquaculture ponds are
fed, fertilised and stocked. Chlorophyll-a, suspended organic solids and turbidity are elevated far
above natural lakes, irrigation reservoirs, and salt pans, and the elevation follows the *feeding
calendar* (L4: "From June to September, fish have high feeding rates ... levels of nutrients ...
increase"). Salt pans are hypersaline and essentially unproductive; irrigation reservoirs are
oligo-/mesotrophic; seasonal wetlands are vegetated, not phytoplankton-dominated. Sensed via
red-edge: chlorophyll absorbs at ~665 nm and the pigment/backscatter peak sits at ~705 nm, so the
red-to-red-edge slope is the discriminant. **We hold red, re1, re2, re3, nira — five bands across
that exact feature — and per the brief we are not exploiting them.** This is the refutation.

**M2 — Managed (aperiodic, step-like) hydrology vs climatic (smooth, annual) hydrology.** A pond is
filled and drained on a *management* schedule that is not locked to the rainfall year; a seasonal
wetland or floodplain follows a smooth annual climatic curve; a permanent lake or reservoir barely
moves. So the discriminant is not "how wet" but **the SHAPE of the wetness trajectory: step-like and
phase-unlocked vs sinusoidal and phase-locked**. L9's two-stage pipeline names this exactly:
"temporal filtering removes transient water bodies through intra-annual time series construction,
temporal aggregation and frequency analysis". *Caveat from L7: the fastest steps (7-15 day dry-outs)
are largely destroyed by monthly compositing; only multi-month management structure survives.*

**M3 — Water persistence, but as a two-sided feature.** L1: ponds are "relatively stable water
bodies" vs "temporary water bodies" (paddy, floodplain). But L5 warns the converse failure:
"misclassification is mainly due to the drainage of aquaculture ponds during non-farming seasons".
So the useful quantity is neither "always wet" nor "sometimes wet" — it is **wet for a long
CONTIGUOUS run, then a clean dry run**, i.e. persistence measured as run length rather than as a
count of wet months.

**M4 — Absence of a canopy phase (the rice killer).** Rice is the one confuser that is water THEN
vegetation. Xiao's rule (L11) captures the flooding inversion; what separates rice from a pond is
that rice **exits** the inverted state into a high-EVI canopy within 1-3 months and the pond does
not. So the feature is a *transition*, not a state.

**M5 — Radar: low, temporally STABLE, cross-pol backscatter.** L1c: temporal median of VH isolates
"permanent and stable low scatterers". Open pond water is a specular surface → very low VH;
wind and emergent vegetation add variance. A wetland with emergent macrophytes shows
volume/double-bounce scattering and much higher VH variance. So **median(VH) low AND IQR(VH) low**
is the pond signature, whereas the *ratio* VH/VV is not (section 3).

## 3. THE VH-VV NULL — BEST EXPLANATION

**Verdict: your VH-VV null is the expected result, not an anomaly. The literature predicts it. Do
not spend another cycle on it, and do not treat it as evidence the feature bank is exhausted —
it is evidence that ONE specific feature was mis-cited as "the most-cited pond feature".**

Four converging reasons, each with a source.

### 3.1 The canonical pond feature was never the ratio — it is temporally-reduced VH alone
Ottinger et al. (IGARSS 2018, verbatim in L1c above) use **"scenes in VH polarization"** and a
**pixel-wise temporal median**. The 2026 Aquaculture review (L3) likewise reduces to annual
median. The dual-pol ratio does not appear in the canonical pond pipeline at all. So "the single
most-cited aquaculture SAR feature did nothing for us" rests on a false premise: the most-cited
feature is `median_t(VH)`, and that is a *temporal order statistic*, not a *band ratio*.

### 3.2 For WATER specifically, polarimetric derivatives are measured to add ~0.1 % accuracy
Ullmann et al. (2022), "Polarimetric information content of Sentinel-1 for land cover mapping: An
experimental case study using quad-pol data synthesized from complementary repeat-pass
acquisitions", *Frontiers in Remote Sensing* 3:905713.
https://www.frontiersin.org/articles/10.3389/frsen.2022.905713/full
VERBATIM:
> "different mathematical derivates were calculated, such as the co-pol ratio (HH/VV) and
> cross-pol-ratio (HH/HV and VV/VH)"
> "the actual improvement in classification accuracy only ranges between **0.1 % (water)** and
> 1.8 % (pasture)."
> "the information content of dual-polarimetric systems is only a fraction of the one which is
> achievable by quad-pol configurations, especially because it only contains diagonal matrix
> elements."
**This is the closest thing in the literature to a direct measurement of our null, and its answer
for the water class is 0.1 %.** Ratios are explicitly framed as "mathematical derivates", i.e.
derived quantities, not new measurements.

### 3.3 The ratio actively imports the noisier channel's noise
Established C-band physics, corroborated across the water-mapping literature:
- **VH is bimodal over water/land; VV is not.** "The VH backscatter histogram shows a bimodal
  distribution with low values over water and high values over land, while the VV histogram has
  multiple peaks and less obvious separation between water and land." (water-detection literature;
  matches Ottinger's own choice of the VH stack for Otsu because it "showed two distinct peaks ...
  and can be considered largely bimodal").
- **VV is the wind-sensitive channel; VH is not.** Bragg scattering from wind-roughened water is
  strongly co-polarised, so a breeze at overpass lifts VV by many dB while VH moves far less. The
  cross-pol channel "has backscattering coefficients relatively independent of water surface
  roughness conditions caused by high wind speed."
Consequence: in dB, `VH - VV` = (clean water/land discriminant) minus (a wind-state nuisance
variable). **The ratio is the difference between your best channel and your worst.** It has *lower*
SNR for the pond/not-pond question than VH alone. This alone predicts a null-to-negative result.

### 3.4 At single-pixel scale, speckle dominates and the ratio doubles the variance
Sentinel-1 GRD is nominally ~4.4 equivalent looks. A single pixel's intensity is therefore Gamma-
distributed with a coefficient of variation near 1/sqrt(L) ~ 0.48 — several dB of noise per
observation. In dB the ratio's variance is the SUM of the two channels' speckle variances (they are
only partially correlated), so `VH_dB - VV_dB` is **noisier than either input**. The literature's
universal remedy is exactly what we are told not to have: spatial multilooking over a pond object,
or **temporal aggregation** (Ottinger's pixel-wise median, "to reduce speckle noise ... and identify
permanent and stable low scatterers"). We have no spatial neighbourhood, so **temporal aggregation
is our only speckle remedy** — which is an argument FOR temporal order statistics on VH and AGAINST
per-month ratios.

### 3.5 Incidence angle — probably NOT the cause, contrary to the hypothesis offered
Worth stating plainly because it changes what to do next. Ottinger et al. restrict to a single orbit
direction explicitly "to reduce effects of orbit direction and look angle" (L1c), so incidence angle
is a real confounder in principle. But VH and VV are acquired **simultaneously through the same
antenna at the same look angle**, and their incidence-angle responses over water are similar in
sign. Taking the ratio therefore **cancels** much of the incidence-angle term. If incidence angle
were the dominant nuisance, the ratio should have *outperformed* the raw channels. It did not.
=> Incidence angle is not the explanation. 3.2/3.3/3.4 are.

### 3.6 Redundancy: the Transformer already has it
`VH_dB - VV_dB` is an exactly-representable linear function of two inputs the model is already fed
at every timestep. A Transformer with a linear input projection computes arbitrary linear
combinations of the 12 bands in its very first layer, at zero cost. **A ratio of two supplied bands
carries literally no new information** — its only possible benefit is optimisation/inductive bias,
which is second-order at 71k params and n=1817. This is the general lesson and it is the correct
filter to apply to every proposal below:
> **A feature is only worth adding if it is NOT a smooth pointwise function of the bands at a single
> timestep.** Pointwise band algebra (NDWI, MNDWI, VH/VV, any 2-band index) is already in the model's
> span. What is NOT in its span, and what it must learn from 1817 examples with a 0.99 adversarial
> shift, is: **order statistics across time, counts of threshold crossings, run lengths, and
> rank/quantile summaries** — non-smooth, permutation-sensitive, sequence-level reductions.
This is the sharpest reframing available and it explains several of your nulls at once (VH-VV,
ROCKET, tree models on raw features). It also says exactly where the remaining signal must be.

## 4. WATER INDICES AND THEIR TEMPORAL STATISTICS

The one paper that actually ran a ROC benchmark of water indices **for pond water** (L3) ranks:
**WIFI-median > AWEI-median > MNDWI > NDWI**, and **median > 95th percentile** as reducer.
We should be computing AWEI and WIFI, not just NDWI/MNDWI. Exact formulas in our column names
(reflectance scaled 0-1):

| index | formula in our band names | source |
|---|---|---|
| NDWI | `(green - nir)/(green + nir)` | McFeeters 1996, IJRS 17(7):1425-1432 |
| MNDWI | `(green - swir1)/(green + swir1)` | Xu 2006, IJRS 27(14):3025-3033 |
| **AWEI_nsh** | `blue + 2.5*green - 1.5*(nir + swir1) - 0.25*swir2` | Feyisa et al. 2014, **RSE 140:23-35** |
| **WIFI (WI2015)** | `1.7204 + 171*green + 3*red - 70*nir - 45*swir1 - 71*swir2` | Fisher, Flood & Danaher, **RSE 175:167-182** |
| LSWI | `(nir - swir1)/(nir + swir1)` | Xiao et al. 2005, **RSE 95(4):480-492** |
| EVI | `2.5*(nir - red)/(nir + 6*red - 7.5*blue + 1)` | Huete et al. 2002, RSE 83:195-213 |
| NDVI | `(nir - red)/(nir + red)` | — |

Temporal statistics worth computing OVER THE OBSERVED MONTHS ONLY (see §6 for the gate):
median, IQR, min, max, the Otsu two-state separability of the within-row value set, the proportion
of observed months above the index's water threshold (AWEI>0, WIFI>0, MNDWI>0), the longest
contiguous wet run divided by the number of observed months, and max|Δ| / Σ|Δ| (step-vs-wander).

## 5. RED-EDGE / WATER-QUALITY LANE — THE UNDER-EXPLOITED OPPORTUNITY

This is where I most strongly disagree with "exhausted". Three independent published groups
(L4, L6, L14) all find that **water-quality/biophysical parameters are what separates aquaculture
ponds from salt pans, rice fields and reservoirs**, and all three of them are per-pixel spectral
quantities requiring no morphology.

Candidate indices, all computable per month from our columns:

| name | formula (our band names) | mechanism | source |
|---|---|---|---|
| **NDCI** | `(re1 - red)/(re1 + red)` | chl-a in turbid productive water | Mishra & Mishra 2012, **RSE 117:394-406** |
| **SPCI** | `(re1 - red)/39.5` | red-edge slope, pond-type discriminant | fmars 2025.1551260 |
| **LASCI** | `(nira - red)/200.1` | red-to-NIR slope, trophic proxy | fmars 2025.1551260 |
| **FAI** | `nir - red - 0.1772*(swir1 - red)` | floating algae / bloom, aerosol-robust | Hu 2009, RSE 113:2118-2129; used in L6 |
| **NDTI** | `(red - green)/(red + green)` | turbidity / suspended sediment | Lacaux et al. 2007, RSE 106:66-74 |
| **red-edge curvature** | `re2 - 0.5*(re1 + re3)` | position of the red-edge inflection; a 3-point curvature that NO 2-band index can express | (we hold re1,re2,re3 — S2 B5/B6/B7) |
| **NDCI-re2** | `(re2 - red)/(re2 + red)` | deeper chl-a / cyanobacteria sensitivity | variant of Mishra & Mishra |
| **narrow-vs-broad NIR** | `nira - nir` | separates water-column backscatter from canopy NIR | uses B8A vs B8, both held |

The **red-edge curvature** and **`nira - nir`** deserve special mention: they are the only two
features here that *cannot* be formed from the classic 6-band Landsat set, which is why the
Landsat-era pond literature (L3, L9) never used them, and why they are the least likely to be
already-discovered ground. Whether they help is an empirical question — but the claim that no
physically-motivated feature remains is false while these are untested.

Temporal statistics to take over them: median, IQR, and **the sign and magnitude of a Theil-Sen
slope across the observed months** (robust, defined for n>=3, and directly encodes the L4 rule
"slow decline followed by rapid increase").

## 6. WINDOW-TRUNCATION ROBUSTNESS — THE HARD GATE

**First, the meta-point, which I think is worth more than any single feature.**
Train rows carry 12 months; test rows carry 4-6 contiguous months. Any statistic computed on the
train rows is therefore computed on **different support** from the same statistic on test rows.
That is a guaranteed covariate shift injected by *us*, on top of the shift already in the data, and
it plausibly contributes to the adversarial AUC of 0.99. It also explains a class of nulls: a
feature can be genuinely informative and still score zero because its train and test distributions
do not overlap.

**Recommendation #0 (an enabler, not a feature): random contiguous-window cropping of the training
rows.** For each training row, sample a contiguous window of length uniform on {4,5,6} at a uniform
start, mask everything outside it, and compute all features (and train the Transformer) on that.
Use K seeded crops per row as augmentation. This (a) matches the support exactly, (b) multiplies
effective n, (c) is seeded/reproducible, (d) uses no external data, (e) is orthogonal to every
closed lane. It converts window-fragile features into window-honest ones *by construction* and
gives you a valid offline harness for testing everything below. If this is not already in the
pipeline it is the highest-expected-value action available before the deadline.
It is also the correct diagnosis of the **duration-normalization null (-0.0064)**: dividing a
feature by the number of observed months is a *post-hoc patch* for a support mismatch. Statistics
whose expectation is already n-invariant (medians, quantiles, proportions, correlations) need no
such patch, and dividing them by n actively injects an n-dependence that was not there — which is a
clean mechanism for why that lane went negative.

**The gate itself.** A feature passes only if its expectation over a random contiguous 5-month
window is close to its value over 12 months. Classified:

**PASSES (n-invariant expectation, bounded, defined for n>=3-4):**
- median / IQR / min / max of any per-month index (median and IQR pass cleanly; min and max drift
  upward/downward with n — marginal, use with cropping).
- **proportions**: fraction of observed months with AWEI>0, with MNDWI>0, with LSWI+0.05>=EVI.
- **Spearman / Pearson correlation between two index series over observed months** — bounded,
  count-free, and the single most window-robust family here.
- **Theil-Sen slope** of an index over observed months (per month, so units are n-free).
- **Otsu two-state separability** eta of the within-row index values (bounded [0,1]).
- **longest wet run / n_obs** (a proportion; degrades gracefully).
- **max|Δ| / Σ|Δ|** over consecutive observed months (bounded; lower bound depends weakly on n).

**FAILS THE GATE — do not build these (and this is why several of the "obvious" management-cycle
features were doomed):**
- **Annual count of dry-down events.** A 5-month window cannot observe an annual count. Fatal.
- **Number of crop cycles per year.** Same. Fatal.
- **Amplitude and phase of an annual harmonic / any Fourier or seasonal decomposition.** A 12-month
  period is not identifiable from a 5-month contiguous window; the fit is rank-deficient. Fatal.
  (This is probably also why ROCKET at -0.009 did nothing: random convolutional kernels over a
  length-12 series are exactly seasonal-shape detectors.)
- **Timing/month-of-minimum or month-of-maximum.** The true extremum frequently lies outside the
  window. Fatal.
- **Seasonal amplitude (annual max - annual min).** Requires the window to contain both phases; a
  4-6 month contiguous window often contains only one. Fatal.
- **Total sums / counts of anything.** Scale directly with n. Fatal (and this is the family that
  "duration normalization" was trying and failing to rescue).

**MARGINAL, only viable WITH recommendation #0:** anything using min or max; anything using a
threshold crossing count (as opposed to a proportion); run lengths in absolute months.

## 7. FEATURE SELECTION AT n=1817 WITH 144 RAW FEATURES

The literature does not measure our exact quantity (n=1817, 144 features, adversarial AUC 0.99,
truncated test windows) and I will not pretend otherwise. What it does say:
- **Raw sequence + masked Transformer beats hand-built temporal percentiles** at large n (L12, CRIT,
  2.3-2.4 pp). Your architecture is already on the right side of that comparison.
- **Do not interpolate the gaps** (L13, ISPRS J. 2024).
So the honest position is: *feature selection is unlikely to beat the raw matrix as a REPLACEMENT.*
The remaining value is in **auxiliary channels that are not in the model's span** and in a
**principled pre-submission filter**.

**The filter I would actually build — a redundancy R-squared test.** For any candidate feature f:
1. Compute f on the (cropped, per #0) training rows.
2. Fit a ridge regression from the 144 raw values (with a missingness indicator per cell) to f,
   train-only, cross-validated.
3. If out-of-fold R-squared(f | raw) > ~0.9, the feature is a near-deterministic function of what the
   model already sees. Deprioritise it.
**This test would have predicted the VH-VV null for free, before spending a submission**: R-squared
of `VH - VV` on the raw matrix is exactly 1.0. It also correctly predicts that NDWI/MNDWI/NDCI as
*per-month* channels are near-redundant (smooth 2-band functions), while `median_over_observed(AWEI)`
and `spearman(VH, MNDWI)` are not (non-smooth, permutation-sensitive reductions over a masked
variable-length sequence — a Transformer can approximate them but must learn them from 1817 rows).
This is a cheap, offline, train-only, submission-free triage and I would run it on every candidate
before anything else.

**A second, cheaper triage — window-stability:** compute Spearman rho between f(12 months) and
f(random 5-month crop) across training rows. Require rho > 0.7. Kills the §6 "FAILS" list
automatically and quantitatively, and will surface fragility you have not anticipated.

**On selection method:** with adversarial AUC 0.99, any selection criterion computed on OOF is
selecting partly on the shift. Prefer (a) the two train-only triages above, then (b) add the
survivors as a *small fixed block* (5-15 scalars) concatenated to the Transformer's pooled
embedding, rather than doing data-driven selection. Fewer decisions fit to noise. Given your
seed variance of 0.019, no single-seed comparison can resolve a real +0.005 effect — use paired
10-seed runs with the same seed list for baseline and candidate and compare the paired difference.

## 8. RANKED PROPOSALS

Ranked by expected value, **window-robustness applied as a hard filter first**. Each entry:
MECHANISM / FORMULA / WINDOW GATE / EVIDENCE / KILL CONDITION / EXPECTED EFFECT / REDUNDANCY.

---
**#0. Random contiguous 4-6 month cropping of training rows (enabler).**
- MECHANISM: matches train support to test support; removes a self-inflicted covariate shift.
- FORMULA: for row i, seed s: start ~ U{0..12-L}, L ~ U{4,5,6}; mask all months outside [start,
  start+L). K crops per row.
- WINDOW GATE: this IS the gate. Passes trivially.
- EVIDENCE: not a literature feature — it follows from the data description plus L12's finding that
  masking (not filling) is the correct way to handle absent timesteps.
- KILL CONDITION: train the existing Transformer with cropping, compare OOF and adversarial AUC. If
  adversarial AUC does not fall meaningfully below 0.99, the shift is not driven by window length
  and the value of #0 is only as an augmentation. Cheap: no submission needed to check the
  adversarial AUC.
- EXPECTED EFFECT: the largest single item here. Also a prerequisite for honestly testing #1-#7.
- REDUNDANCY: none. Orthogonal to relative-time encoding (that tells the model *where*; this fixes
  *how many*).

---
**#1. Red-edge water-quality block (chlorophyll / trophic state), temporally reduced.**
- MECHANISM (M1): feed and fertiliser make pond water eutrophic and turbid; salt pans, reservoirs
  and clean wetlands are not. Sensed by the red / red-edge slope.
- FORMULA: per observed month t compute
  `NDCI_t = (re1_t - red_t)/(re1_t + red_t)`,
  `LASCI_t = (nira_t - red_t)/200.1`,
  `FAI_t = nir_t - red_t - 0.1772*(swir1_t - red_t)`,
  `REC_t = re2_t - 0.5*(re1_t + re3_t)`   (red-edge curvature),
  `NDTI_t = (red_t - green_t)/(red_t + green_t)`.
  Then reduce over observed months only: `median`, `IQR`, `TheilSen slope`. 15 scalars.
  Optionally mask the reduction to months where the pixel is wet (AWEI_t > 0) — a water-only
  trophic summary, which is what the source papers actually do.
- WINDOW GATE: **PASSES.** Medians, IQRs and per-month slopes are n-invariant in expectation.
- EVIDENCE: L4 (fmars 2025.1551260; F1=0.81 on pond TYPE from red-edge slopes alone), L6 (Peng et
  al., MarPolBull 181:113901, 91 % OA with Chl-a/TSI/FAI), L14 (Hou et al., RemSens 14:3306, "water
  quality characteristics improved the accuracy of distinguishing aquaculture ponds from salt pans,
  rice fields, and wetland parks ... F1 scores > 85 %").
- KILL CONDITION: single-feature AUC of each scalar on cropped training rows; and redundancy
  R-squared vs the raw matrix. If every scalar has |AUC-0.5| < 0.03 AND R-squared > 0.9, kill.
- EXPECTED EFFECT: the best-supported lane. Three independent groups; a mechanism our confuser list
  matches exactly; and bands (re1/re2/re3/nira) the Landsat-era literature could not use.
- REDUNDANCY: the per-month indices ARE in the Transformer's span (§3.6) — **only the temporal
  reductions are new**. Feed the reductions, not the per-month channels. Be disciplined about this;
  it is the VH-VV lesson.

---
**#2. AWEI and WIFI temporal medians (the ROC-benchmarked winners for POND water).**
- MECHANISM (M3/M5): better water/land separation for pond water specifically than NDWI/MNDWI.
- FORMULA: `AWEI_t = blue_t + 2.5*green_t - 1.5*(nir_t + swir1_t) - 0.25*swir2_t`;
  `WIFI_t = 1.7204 + 171*green_t + 3*red_t - 70*nir_t - 45*swir1_t - 71*swir2_t` (reflectance 0-1);
  features: `median_obs(AWEI)`, `IQR_obs(AWEI)`, `median_obs(WIFI)`, `IQR_obs(WIFI)`,
  `frac_obs(AWEI > 0)`, `frac_obs(WIFI > 0)`.
- WINDOW GATE: **PASSES** (medians, IQRs, proportions).
- EVIDENCE: L3 verbatim — "AWEI and WIFI were more effective water indices than NDWI or MNDWI ...
  the median performed better than the 95th percentile ... the WIFI median has the best combination
  of water index and reducer."
- KILL CONDITION: as #1. Additionally check the two proportions are not degenerate (all 0 or all 1).
- EXPECTED EFFECT: moderate. This is the literature's own optimum for this exact sub-task.
- REDUNDANCY: the per-month index is in span; the median/IQR/proportion are not. Note WIFI's
  coefficients are far from anything a Transformer would find quickly from 1817 rows — the strong
  negative weights on swir2 and nir are an unusual direction. Genuinely worth trying.

---
**#3. Cross-band temporal correlations (the most window-robust family, and the least redundant).**
- MECHANISM (M2/M4/M5): a filling/draining pond drives VH down and MNDWI up together; rice drives
  VH up with EVI as the canopy grows; a permanent lake drives nothing. The *coupling* between two
  channels over time is the signature, and it is invariant to each channel's level and scale — so
  it is immune to the incidence-angle and atmospheric offsets that plague absolute values.
- FORMULA: over the observed months only,
  `r1 = spearman(VH_t, MNDWI_t)`, `r2 = spearman(VH_t, EVI_t)`, `r3 = spearman(VV_t, MNDWI_t)`,
  `r4 = spearman(NDCI_t, AWEI_t)`, `r5 = spearman(LSWI_t, EVI_t)`. Set to 0 when n_obs < 4.
- WINDOW GATE: **PASSES BEST OF ALL.** Bounded [-1,1], count-free by construction, defined at n=4.
- EVIDENCE: mechanism-level rather than a single citation — it operationalises L11's LSWI-vs-EVI
  inversion (r5, r2) and L1's stable-vs-transient water (r1). I have not found a paper that computes
  exactly these correlations for pond mapping; **say so plainly, this one is an extrapolation.**
- KILL CONDITION: single-feature AUC on cropped train; plus check n_obs>=4 coverage in test.
- EXPECTED EFFECT: uncertain but the redundancy argument is the strongest of any item — a rank
  correlation over a masked variable-length sequence is a normalised second-order cross-moment,
  which is precisely what a small attention model with 71k params and 1817 examples will struggle
  to learn and will happily use if handed.
- REDUNDANCY: **lowest of everything here.** Not a smooth pointwise function of the bands.

---
**#4. Xiao rice-rejection block.**
- MECHANISM (M4): rice = flooding inversion FOLLOWED BY a canopy; ponds never green up.
- FORMULA: `LSWI_t = (nir_t - swir1_t)/(nir_t + swir1_t)`,
  `EVI_t = 2.5*(nir_t - red_t)/(nir_t + 6*red_t - 7.5*blue_t + 1)`;
  `f1 = frac_obs(LSWI_t + 0.05 >= EVI_t)`, `f2 = max_obs(EVI_t)`, `f3 = median_obs(EVI_t)`,
  `f4 = f1 * (1 - clip(f2,0,1))` (flooded AND never green).
- WINDOW GATE: **f1, f3 PASS** (proportion, median). **f2 MARGINAL** (max drifts with n; only use
  with #0). f4 inherits f2's marginality.
- EVIDENCE: L11, Xiao et al. RSE 95(4):480-492 (2005) and RSE 100(1):95-113 (2006) — verbatim rule
  "LSWI + 0.05 >= EVI or LSWI + 0.05 >= NDVI".
- KILL CONDITION: this one has an unusually clean check. Split training positives/negatives and plot
  the joint density of (f1, f2). If rice-like negatives do not form a distinct high-f1/high-f2 lobe,
  rice is not actually a major confuser in this dataset and the whole block can be dropped for free.
- EXPECTED EFFECT: moderate, contingent on rice actually being in the negative class.
- REDUNDANCY: f1 and f4 are threshold-crossing proportions — NOT in span. f3 largely is.

---
**#5. Two-state / step-vs-smooth statistics on the water index.**
- MECHANISM (M2): managed fill/drain is step-like and bimodal; climatic seasonality is smooth and
  unimodal; a permanent reservoir is flat.
- FORMULA: let `x_t = AWEI_t` over observed months.
  `eta = Otsu between-class variance of {x_t} / total variance of {x_t}` (bounded [0,1]);
  `step = max_t |x_{t+1} - x_t| / sum_t |x_{t+1} - x_t|` over consecutive OBSERVED months;
  `runfrac = longest contiguous run with x_t > 0, divided by n_obs`.
- WINDOW GATE: **eta and runfrac PASS; step PASSES with a weak n-dependent lower bound.**
- EVIDENCE: L9 ("temporal filtering removes transient water bodies through ... temporal aggregation
  and frequency analysis"); L1's bimodality argument, and L3's Otsu machinery applied here at the
  row level instead of the image level. The row-level application is our own extension — flag it.
- KILL CONDITION: window-stability rho between eta(12mo) and eta(5mo crop). I expect this to be the
  weakest of the passing set; if rho < 0.7, kill it before anything else.
- EXPECTED EFFECT: small-to-moderate. Physically the most appealing, statistically the shakiest at
  n_obs=4.
- REDUNDANCY: low. Otsu-eta is emphatically not a smooth function of the inputs.

---
**#6. The actual canonical SAR feature: temporally-reduced VH (NOT the ratio).**
- MECHANISM (M5): pond water is specular → very low VH, and *stably* low.
- FORMULA: `median_obs(VH)`, `IQR_obs(VH)`, `frac_obs(VH < median_obs_over_dataset(VH))`.
  (Do not use the dataset-wide threshold if it constitutes tuning; use the row's own quantiles.)
- WINDOW GATE: **PASSES.**
- EVIDENCE: L1c verbatim — "we used scenes in VH polarization"; "the pixel-wise median ... to reduce
  speckle noise ... and identify permanent and stable low scatterers".
- KILL CONDITION: redundancy R-squared. Median over a masked sequence is not in span, but a
  Transformer that already has 12 months of VH may approximate it well; expect high R-squared.
- EXPECTED EFFECT: small. Listed mainly to correct the record: this, not VH/VV, is the canonical
  feature, and it has not been tested.
- REDUNDANCY: moderate-to-high. Test with the R-squared filter first.

---
**#7. `nira - nir` and the red-edge curvature as standalone channels.**
- MECHANISM: B8A (narrow NIR, 865 nm) excludes the 940 nm water-vapour shoulder that B8 includes;
  over water the two diverge with atmospheric column and with water-column scattering. The
  three-point red-edge curvature `re2 - 0.5*(re1+re3)` measures the *shape* of the red edge, which
  no two-band index can express, and shifts with chlorophyll concentration.
- FORMULA: `median_obs(nira - nir)`, `median_obs(re2 - 0.5*(re1+re3))`, plus their IQRs.
- WINDOW GATE: **PASSES.**
- EVIDENCE: weakest of the set — mechanism-level plus the observation that these bands are absent
  from Landsat, hence absent from L3/L9/L11. **No paper found that computes these for pond mapping.
  State this as untested extrapolation, not as a literature result.**
- KILL CONDITION: single-feature AUC; if |AUC-0.5| < 0.02 on cropped train, drop.
- EXPECTED EFFECT: low individually, but near-zero cost once #1's machinery exists.
- REDUNDANCY: per-month, fully in span; the medians/IQRs are not.

---
**EXPLICITLY NOT PROPOSED** (fails §6 gate, listed so the lane is closed with a reason, not a
hunch): annual dry-down count; crops-per-year; harmonic/Fourier amplitude & phase; month-of-min /
month-of-max; seasonal amplitude (annual max minus annual min); any sum or raw count; any
morphological, textural, or neighbourhood feature; anything using an external raster.

## 9. CAVEATS

1. **The refutation is partial, and honestly so.** The claim "no remaining physically-motivated
   feature exists" is false — the red-edge/trophic-state family (§5) is published, triple-confirmed,
   per-pixel, and untested by us. But the *stronger* claim behind it — "the easy temporal wins are
   gone" — is largely TRUE, and L7 and L12 are the evidence: monthly compositing destroys the
   7-15 day drain/dry pulse that the management-cycle story depends on, and a masked Transformer on
   the raw series already beats temporal percentiles by 2.3-2.4 pp at large n.
2. **L12 is real counter-evidence and I have not explained it away.** If percentile summaries lose
   to raw sequences at large n, our only argument for them is the small-n + large-shift regime. That
   argument is plausible and untested. Treat every §8 item as an *auxiliary* block appended to the
   pooled embedding, never as a replacement for the raw matrix.
3. **Reflectance scaling is unverified.** AWEI and especially WIFI have coefficients that assume
   surface reflectance in 0-1. If the competition columns are scaled 0-10000, or are TOA rather than
   BOA, or are already normalised, WIFI in particular will be numerically meaningless. **Check the
   column value ranges before computing anything in §4.** Similarly, if VH/VV are supplied in linear
   power rather than dB, medians and IQRs behave very differently — check.
4. **The red-edge lane assumes the optical bands are BOA surface reflectance.** NDCI and FAI are
   water-quality algorithms validated on atmospherically-corrected data. Monthly compositing of
   10 m water pixels also risks adjacency effects from the surrounding dikes at a 10x10 m patch,
   which is a known problem for small-water-body optical retrieval and which I did not find
   quantified for 10 m pixels — an unquantified risk.
5. **Sources I could not verify by full-text fetch** (403 / paywall) and which are therefore quoted
   at search-snippet level only, flagged inline: the IJRS 2025 multi-feature-fusion "+6 % from
   temporal characteristics"; the Ocean & Coastal Management 2026 pond-dike paper (L5); the Jiangsu
   salt-pan paper and STF-RF quotes (L9); the GIScience & Remote Sensing 2025 review; the Peng et
   al. 2022 (L6) and Hou et al. 2022 (L14) full texts (abstracts verified, methods not). **Verify
   before quoting these in any write-up.** L1c, L3, L4 and L4b are verified from full text or the
   publisher's XML and can be quoted as-is.
6. **The FAI interpolation constant 0.1772** was computed by me from Sentinel-2A central
   wavelengths (664.6 / 832.8 / 1613.7 nm), not taken from a paper. Hu (2009) defines FAI for MODIS
   bands; the S2 adaptation is standard but the constant should be recomputed if the competition
   used Sentinel-2B (slightly different centres).
7. **No number in this document is an estimate of our leaderboard delta.** The accuracies quoted
   (0.83, 0.81 F1, 91 % OA, >85 % F1, 2.3-2.4 pp, 0.1 %) are the source papers' own, measured on
   their data with their (mostly morphological) pipelines, and do not transfer to a per-pixel
   4-6-month problem scored by 0.6*F1 + 0.4*AUC.
8. **Seed variance 0.019 exceeds every effect size proposed here.** Nothing in §8 can be validated
   by a single run. Use paired 10-seed comparisons with a fixed seed list, and prefer the two
   train-only offline triages in §7 to spending submissions.
