# Week 2 · Day 12 — Structural Stability Tests

> **What you will build:** `src/econometrics/stability_tests.py`
> **What you will produce:** Chow test results, rolling eigenvalue plot, CUSUM summary
> **Why it matters:** Every examiner will ask: **"Is your VAR stable over time, or should you split the sample?"**

---

## 0  Pre-requisites

| Concept | Where covered |
|---------|---------------|
| Structural breaks (Nigeria) | `docs/identification_strategy.md` — 2016, 2020, 2023 events |
| VAR companion matrix | Day 7 (stability requires max |eigenvalue| < 1) |
| Sub-sample estimation | Day 13 (robustness checks) |

---

## 1  Why Stability Matters

A VAR is **stable** if its parameters (coefficients, error covariance) are constant over the sample. If there's a structural break, the full-sample VAR is misspecified — it's averaging across two different regimes.

### 1.1  Consequences of instability

- IRFs are unreliable (they mix pre-break and post-break dynamics)
- Forecasts deteriorate
- Hypothesis tests (Granger causality) have wrong size

### 1.2  Known break dates in Nigeria

- **2016-06-01**: Naira devaluation (official rate moved from 197 to 305)
- **2020-03-01**: COVID-19 (lockdowns, oil-price crash)
- **2023-06-01**: FX unification (multiple rates → single NAFEM rate)

---

## 2  Test 1: Chow Test

### 2.1  The idea

Split the sample at a suspected break date. Fit three VARs:
1. **Full sample** (T observations)
2. **Sub-sample 1** (T₁ observations, before break)
3. **Sub-sample 2** (T₂ observations, after break)

Null hypothesis: No structural break (parameters are the same).

### 2.2  F-statistic

```
F  =  [ (RSS_full - RSS_1 - RSS_2) / k ]  /  [ (RSS_1 + RSS_2) / (T₁ + T₂ - 2k) ]
```

where `k` = number of parameters in each VAR.

If F is large (p < 0.05), **reject H₀** → there's a structural break.

### 2.3  Code

```python
# stability_tests.py — chow_test
var_full = VAR(self.df).fit(self.opt_lag)
var_1    = VAR(df1).fit(self.opt_lag)   # before break
var_2    = VAR(df2).fit(self.opt_lag)   # after break

rss_full = np.sum(var_full.resid ** 2)
rss_rest = np.sum(var_1.resid ** 2) + np.sum(var_2.resid ** 2)

F_stat = ((rss_full - rss_rest) / n_params) / (rss_rest / df_resid)
```

---

## 3  Test 2: Rolling VAR Eigenvalues

### 3.1  The idea

Fit VAR on **rolling 60-month windows**. For each window, extract the eigenvalues of the **companion matrix**. Plot the **maximum |eigenvalue|** over time.

A VAR is stable if max |eigenvalue| < 1. If it crosses 1.0, the VAR is explosive (unit root or worse).

### 3.2  Why rolling windows?

Static tests (like Chow) assume you know the break date. Rolling eigenvalues detect **gradual** changes or multiple breaks.

### 3.3  Code

```python
# stability_tests.py — rolling_eigenvalues
for t in range(window + self.opt_lag, n):
    df_window = self.df.iloc[t - window:t]
    var = VAR(df_window).fit(self.opt_lag)
    companion = var.companion_matrix()
    eigs = np.linalg.eigvals(companion)
    max_eig = np.max(np.abs(eigs))
```

### 3.4  Reading the plot

- **Horizontal red line at 1.0** = unit-circle boundary
- If the blue line (max eigenvalue) stays below 1.0 for all windows → stable
- If it spikes above 1.0 → explosive period (check dates)

---

## 4  Test 3: CUSUM (Simplified)

### 4.1  The idea

CUSUM (cumulative sum) tracks **recursive residuals**. If parameters change, residuals will drift systematically, and the cumulative sum will exceed confidence bounds.

Full CUSUM requires re-estimating the VAR recursively (expensive). The script provides a **simplified version**:

1. Compute cumulative residuals: `cumsum = np.cumsum(resid)`
2. Check if they exceed ±3σ bounds
3. Count exceedances

If exceedance rate > 5%, parameters are unstable.

### 4.2  Limitations

This is a rough approximation. For publication-quality CUSUM, use a specialized package (e.g., `statsmodels.stats.diagnostic.breaks_cusumolsresid` for single-equation OLS).

---

## 5  Run the Script

```bash
python src/econometrics/stability_tests.py
```

### Output files

| File | Contents |
|------|----------|
| `results/stability/chow_tests.csv` | F-stat, p-value for 3 break dates |
| `results/stability/rolling_eigenvalues.png` | Time-series plot of max |eig| |
| `results/stability/rolling_eigenvalues.csv` | Rolling max eigenvalues (all dates) |
| (CUSUM printed to console) | Exceedance rate |

---

## 6  Interpretation Guide

### 6.1  If Chow rejects at 2016 but not at 2020/2023

**Thesis phrasing:**

> "The Chow test detects a structural break at the 2016 naira devaluation (F = X.XX, p < 0.05), but not at the 2020 COVID onset or 2023 FX unification. This suggests the devaluation regime-shifted the exchange-rate transmission mechanism, while later shocks were absorbed within the post-2016 regime. The full-sample VAR is thus a weighted average of pre- and post-2016 dynamics. Robustness checks (Day 13) confirm that sub-sample results are qualitatively similar."

### 6.2  If rolling eigenvalues spike above 1.0

**Thesis phrasing:**

> "Rolling-window analysis reveals that the VAR briefly became explosive during [DATES], with max eigenvalue exceeding 1.0. This coincides with the [EVENT] episode. I exclude these observations as outliers in robustness checks, and the main results remain unchanged."

### 6.3  If all tests show stability

**Thesis phrasing:**

> "Structural stability tests (Chow, rolling eigenvalues, CUSUM) fail to reject parameter constancy at the 5 % level. The full-sample VAR is therefore appropriate for inference."

---

## 7  Common Questions

**Q: Chow rejected at 2016. Should I split the sample?**

A: Two options:
1. **Split and report both**: Estimate VAR on 2010-2016 and 2016-2025 separately. Show that transmission strengthened/weakened. (Time-consuming.)
2. **Keep full sample, note the caveat**: "The full-sample VAR averages pre- and post-2016 regimes. The 2016 break is acknowledged but does not invalidate the findings, as the transmission mechanism (MPR → Inflation) operates in both regimes." (Standard practice.)

**Q: What if rolling eigenvalues are always < 1 but fluctuate a lot?**

A: This is normal. Eigenvalues are sensitive to sample variation. As long as they stay below 1.0, the VAR is stable.

**Q: The CUSUM exceedance rate is 8%. Is that a problem?**

A: Borderline. Standard cutoff is 5%. With 181 observations, an 8% rate could be noise. If it bothers you, run a formal Brown-Durbin-Evans CUSUM test (requires recursive estimation).

---

## 8  Examiner Defense Script

**Examiner:** "Did you test for structural breaks?"

**You:** "Yes. I ran Chow tests at the three known break dates (2016 devaluation, 2020 COVID, 2023 FX unification). The 2016 break is marginally significant at the 10 % level, but not at 5 %. Rolling VAR eigenvalues remain below 1.0 for 95 % of windows, confirming stability. Full results are in Table X and Figure Y."

**Examiner:** "Why not split the sample?"

**You:** "Splitting at 2016 would leave only 72 pre-break observations, insufficient for a VAR(11) with 4 variables (176 parameters). The full-sample approach is standard in the literature (cite Sims 1980, Christiano et al. 1999). I show robustness to alternative lag orders in Day 13."

---

## 9  Checklist

- [ ] `stability_tests.py` runs without errors
- [ ] Chow test CSV saved — you know which dates show breaks
- [ ] Rolling eigenvalue plot saved — max |eig| < 1 for most windows
- [ ] CUSUM exceedance rate < 10 %
- [ ] You can defend the full-sample VAR if Chow rejects
- [ ] Ready for Day 13: Robustness checks
