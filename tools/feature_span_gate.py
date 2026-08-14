"""Feature-span gate — is a candidate feature ALREADY inside the model's reachable span?

WHY THIS EXISTS
---------------
This project has killed several physically-motivated features on the leaderboard, most notably
VH−VV in three independent forms, all null. Round-22 research supplied both the correct reading of
the literature and the mechanical explanation, and the explanation is cheap enough to have been a
gate all along:

  * The canonical aquaculture SAR feature is **VH alone, pixel-wise temporal median** (Ottinger et
    al., IGARSS 2018, DOI 10.1109/IGARSS.2018.8651419) — *"we used scenes in VH polarization"*,
    *"the pixel-wise median was calculated … to identify permanent and stable low scatterers"*. The
    dual-pol RATIO is not in the canonical pipeline at all. Ullmann et al. (Front. Remote Sens.
    3:905713, 2022) measured what polarimetric derivatives add over intensity for water: **0.1 %**.
    So our VH−VV null was the *predicted* result and we had mis-cited our own motivation.
  * Mechanically, `VH − VV` is an EXACTLY LINEAR function of two supplied columns. A model that
    already receives both columns can represent it at zero cost, so handing it over as a new input
    adds no information — only width, and added width has lost every time in this project.

THE GATE. For a candidate feature f, regress f on the 144 raw values with a cross-fitted ridge and
report R². R² → 1 means f lies inside the linear span of what the model already sees, and it cannot
be expected to help. This is a NECESSARY-condition screen, not a sufficient one: a low R² says the
feature is *reachable* only nonlinearly, not that it is useful — that is what `univ_auc` is for.

THE SECOND GATE, which is the one specific to this competition. Test rows show only 4–6 contiguous
months. Any feature computable over 12 months but unstable under window truncation is useless no
matter how good its physics. We therefore report Spearman ρ between the feature computed on the
full 12 months and on the masked test-like window. **Round-22 flagged that this kills the whole
Fourier/harmonic family — a 12-month period is unidentifiable from a 5-month window — which is a
candidate explanation for the ROCKET null (−0.009).**

DIAGNOSIS ONLY. Nothing here touches the operating point; no feature is added by this tool.

USAGE
    python tools/feature_span_gate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from src.data import BAND_ORDER, load_bundle  # noqa: E402
from src.seq_model import _mask_views  # noqa: E402
from src.utils import get_logger, load_config, roc_auc  # noqa: E402

log = get_logger()

B = {b: i for i, b in enumerate(BAND_ORDER)}
_DN = 1e4          # S2 columns are raw DN; the water indices below assume 0–1 reflectance.


def _nm(c, band):
    """Reflectance-scaled band over months, NaN where unobserved. SAR stays in dB."""
    x = c[:, :, B[band]]
    return x if band in ("VH", "VV") else x / _DN


def _nanmed(x):
    with np.errstate(invalid="ignore"):
        return np.nanmedian(x, axis=1)


def candidates(c: np.ndarray) -> dict:
    """Literature-motivated candidates, each computable from ONE location's 12x12 table."""
    vh, vv = _nm(c, "VH"), _nm(c, "VV")
    g, r, nir, nira = _nm(c, "green"), _nm(c, "red"), _nm(c, "nir"), _nm(c, "nira")
    re1, sw1, sw2 = _nm(c, "re1"), _nm(c, "swir1"), _nm(c, "swir2")

    with np.errstate(invalid="ignore", divide="ignore"):
        mndwi = (g - sw1) / (g + sw1)
        ndwi = (g - nir) / (g + nir)
        # Feyisa et al., RSE 140:23-35 (2014), no-shadow form.
        awei = 4.0 * (g - sw1) - (0.25 * nir + 2.75 * sw2)
        # fmars 2025.1551260: pond-type separation from red/red-edge slopes alone.
        lasci = (nira - r) / 200.1
        spci = (re1 - r) / 39.5

    # THE CONTROL MUST BE EXACTLY LINEAR IN THE 144 RAW VALUES, or it does not validate the gate.
    # A first version of this tool used median_over_months(VH − VV) as the control and it scored
    # R² = 0.6206, not ~1.0 — because a MEDIAN is a nonlinear function of the raw values, so that
    # row was testing the median, not the difference. The honest control is the difference at ONE
    # fixed month, which is literally `col[m,VH] − col[m,VV]`: two of the 144 columns, coefficients
    # +1 and −1. If THAT does not return ~1.0 the ridge basis is wrong and every other row is void.
    m_mid = c.shape[1] // 2
    out = {
        "CONTROL VH-VV @1 month (exactly linear in 2 cols)":
            c[:, m_mid, B["VH"]] - c[:, m_mid, B["VV"]],
        "VH_minus_VV_median (NONlinear: a median, not a difference)": _nanmed(vh - vv),
        "VH_median        (Ottinger canonical)": _nanmed(vh),
        "VH_p10           (permanence tail)": np.nanpercentile(vh, 10, axis=1),
        "MNDWI_median": _nanmed(mndwi),
        "NDWI_median": _nanmed(ndwi),
        "AWEI_nsh_median  (Feyisa 2014)": _nanmed(awei),
        "LASCI_median     (fmars 2025)": _nanmed(lasci),
        "SPCI_median      (fmars 2025)": _nanmed(spci),
        "red_edge_curv    (nira-nir, extrapolated)": _nanmed(nira - nir),
    }
    # Cross-band temporal correlation: nonlinear in the raw columns by construction, so it is the
    # family most likely to escape the span. Flagged in round-22 as an extrapolation with no paper.
    n = c.shape[0]
    rho = np.full(n, np.nan)
    for i in range(n):
        a, b = vh[i], nir[i]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() >= 4 and np.ptp(a[m]) > 0 and np.ptp(b[m]) > 0:
            rho[i] = spearmanr(a[m], b[m]).statistic
    out["corr_VH_nir     (cross-band, NONLINEAR)"] = rho
    return out


def span_r2(f: np.ndarray, X: np.ndarray, seed: int = 42) -> float:
    """Cross-fitted R² of a ridge regression of f on the 144 raw values."""
    ok = np.isfinite(f)
    f, X = f[ok], X[ok]
    if len(f) < 100 or np.nanstd(f) == 0:
        return float("nan")
    pred = np.zeros_like(f)
    for tr, va in KFold(5, shuffle=True, random_state=seed).split(X):
        m = RidgeCV(alphas=np.logspace(-3, 4, 20)).fit(X[tr], f[tr])
        pred[va] = m.predict(X[va])
    ss_res = float(((f - pred) ** 2).sum())
    ss_tot = float(((f - f.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main() -> None:
    cfg = load_config()
    b = load_bundle(cfg)
    y, full = b.y, b.train_cube
    n = len(y)

    # The masked replica: the SAME sampler training uses, so the window gate is measured against
    # the real test window distribution rather than a uniform guess over 4-6.
    masked, owner = _mask_views(full, np.arange(n), b.schema, b.window_dist, cfg, 1, 42, oof=True)
    assert np.array_equal(owner, np.arange(n)), "K=1 should give one view per row in row order"

    obs = np.isfinite(masked[:, :, 0]).sum(1)
    log.info("masked replica: %d rows, observed months min %d max %d mean %.2f",
             n, obs.min(), obs.max(), obs.mean())

    # The model's reachable span = the 144 raw values, NaN-filled the way the net sees them.
    X = np.nan_to_num(full.reshape(n, -1), nan=0.0)
    log.info("span basis: %d raw values per row", X.shape[1])

    c_full, c_mask = candidates(full), candidates(masked)

    log.info("")
    log.info("%-52s %8s %9s %9s", "candidate", "span R2", "window rho", "univ AUC")
    log.info("%s", "-" * 82)
    for k in c_full:
        ff, fm = c_full[k], c_mask[k]
        r2 = span_r2(fm, X)
        ok = np.isfinite(ff) & np.isfinite(fm)
        rho = spearmanr(ff[ok], fm[ok]).statistic if ok.sum() > 50 else float("nan")
        good = np.isfinite(fm)
        auc = roc_auc(y[good], np.nan_to_num(fm[good])) if good.sum() > 50 else float("nan")
        auc = max(auc, 1 - auc)                    # direction-free discriminability
        flag = ""
        if np.isfinite(r2) and r2 > 0.90:
            flag += "  <- IN SPAN, expect null"
        if np.isfinite(rho) and rho < 0.70:
            flag += "  <- WINDOW-UNSTABLE"
        log.info("%-52s %8.4f %9.4f %9.4f%s", k, r2, rho, auc, flag)

    log.info("")
    log.info("READING. `span R2` > 0.90 means the feature is already inside the linear span of the")
    log.info("144 values the model receives, so adding it buys width and no information -- this is")
    log.info("the mechanical form of our VH-VV null, and the CONTROL row should sit at ~1.0.")
    log.info("`window rho` < 0.70 means the feature does not survive truncation to a test-like 4-6")
    log.info("month window, which disqualifies it regardless of physics. `univ AUC` is single-")
    log.info("feature discriminability on TRAIN and is descriptive only -- train-only AUC has never")
    log.info("predicted transfer in this project (LB_LOG iter46: OOF is blind, adv-AUC retired).")
    log.info("")
    log.info("A LOW span R2 IS NOT A GO SIGNAL. It says the feature is unreachable LINEARLY, not")
    log.info("that it helps. This gate can only ever VETO; nothing here funds a submission.")


if __name__ == "__main__":
    main()
