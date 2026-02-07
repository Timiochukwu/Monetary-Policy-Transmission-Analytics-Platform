# Week 2 · Day 11 — Historical Shock Decomposition

> **What you will build:** `src/econometrics/historical_decomposition.py`
> **What you will produce:** 4 stacked-area plots, event attribution table
> **Why it matters:** Explains **what drove past events** — "Was the 2023 inflation spike due to MPR policy, FX shocks, or supply shocks?"

---

## 0  Pre-requisites

| Concept | Where covered |
|---------|---------------|
| VAR MA representation | Day 7 (VAR coefficients → IRF → MA) |
| Structural shocks ε | Day 8 (orthogonalized via Cholesky) |
| Historical events | `docs/identification_strategy.md` |

---

## 1  Theory Refresher

### 1.1  What is historical decomposition?

The VAR can be written in **moving-average (MA) form**:

```
y_t  =  μ + Σ_{s=0}^{t-1} Φ_s ε_{t-s}
```

where:
- `μ` = baseline (unconditional mean, driven by intercepts)
- `Φ_s` = MA coefficient matrices (from IRF)
- `ε_t` = orthogonalized structural shocks

Historical decomposition **partitions** the observed `y_t` into:
```
y_t  =  baseline  +  Σ_j contrib_j(t)
```

where `contrib_j(t)` = cumulative contribution from shock source `j`.

### 1.2  Economic interpretation

If Inflation at 2023-06 was 22.8%, the decomposition tells you:
- **5.0%** came from MPR shocks (tight policy)
- **2.1%** came from ExchangeRate shocks (naira depreciation)
- **−3.1%** came from M2 shocks (liquidity contraction)
- **6.8%** came from Inflation's own shocks (supply-side)
- **Baseline**: ~12%

This lets you attribute events to specific channels.

---

## 2  Code Walkthrough

### 2.1  Extract structural shocks

```python
# historical_decomposition.py — compute
cov_u = self.res.sigma_u
P     = np.linalg.cholesky(cov_u)
P_inv = np.linalg.inv(P)

resid = self.res.resid.values        # reduced-form residuals
eps   = (P_inv @ resid.T).T          # structural shocks: ε = P^{-1} u
```

### 2.2  Compute contributions

```python
# For each time t, variable i, shock j:
for t in range(T_resid):
    for i in range(K):
        for j in range(K):
            # Sum over all past shocks
            for s in range(t + 1):
                contrib[t, i, j] += ma_coef[t - s, i, j] * eps[s, j]
```

This is the **convolution** of MA coefficients and structural shocks.

---

## 3  Reading the Plots

### 3.1  Stacked-area chart

Each plot shows one variable (e.g., Inflation) over time. The Y-axis is the **level** (not the change). Colored bands represent cumulative contributions from each shock.

**Key visual cues:**
- **Black line** = actual data (overlay)
- **Grey band** = baseline (intercept effect)
- **Colored bands** = shock contributions (stacked)

If the colored bands match the black line, the decomposition is accurate.

### 3.2  Event markers

Three vertical red lines mark structural breaks:
- **2016-06**: Naira devaluation
- **2020-03**: COVID-19 onset
- **2023-06**: FX market unification

Look at the Inflation plot near 2023-06: which shock band expanded the most?

---

## 4  Event Attribution Table

The script prints a table like this:

```
  FX Unification (2023-06-01)
  ─────────────────────────────────────────────────────────────
    Inflation          (actual:   22.80)
      MPR              :    1.15  (  5.0 %)
      ExchangeRate     :    0.47  (  2.1 %)
      M2               :   -0.71  ( -3.1 %)
      Inflation        :    1.54  (  6.8 %)
```

**Interpretation:**
- Inflation's own shocks (supply-side: food, fuel) contributed 6.8%
- MPR shocks contributed 5.0% (tight policy pulled inflation down slightly)
- ExchangeRate shocks contributed 2.1% (naira depreciation added inflation)
- M2 shocks contributed −3.1% (liquidity contraction reduced inflation)

### 4.1  Thesis use

> "The June 2023 inflation spike (22.8%) is decomposed as follows: 43% from the baseline trend, 23% from exchange-rate pass-through, 22% from supply-side shocks (captured in Inflation's own innovations), and 21% from MPR policy. The moderate contribution of MPR reflects the CBN's gradualist approach to tightening during the FX unification episode."

---

## 5  Run the Script

```bash
python src/econometrics/historical_decomposition.py
```

### Output files

| File | Contents |
|------|----------|
| `results/historical_decomposition/hist_decomp_Inflation.png` | Stacked-area plot |
| `results/historical_decomposition/hist_decomp_Inflation.csv` | Shock contributions (all dates) |
| (+ 3 more for MPR, ExchangeRate, M2) | |

---

## 6  Common Questions

**Q: Why doesn't the stacked area exactly match the black line?**

A: Two reasons:
1. The baseline is an unconditional mean — it doesn't capture time variation in the intercept.
2. Small numerical errors accumulate over 181 months.

If the gap is large (>5%), check for coding errors.

**Q: Can I use this to forecast?**

A: No. Historical decomposition is **backward-looking** (it uses all past shocks, including future ones relative to time t). For forecasting, use Day 14's out-of-sample forecast module.

**Q: The 2020 COVID spike — which shock dominated?**

A: Look at the Inflation plot around 2020-03. You'll likely see the **ExchangeRate band** expand (naira depreciated sharply) and the **Inflation own-shock band** expand (supply disruptions).

**Q: What's the difference between this and FEVD (Day 9)?**

A: FEVD tells you **on average, over all time**, which shocks explain variance. Historical decomposition tells you **at specific dates**, which shocks drove the observed movement.

---

## 7  Checklist

- [ ] `historical_decomposition.py` runs without errors
- [ ] 4 plots saved in `results/historical_decomposition/`
- [ ] You've identified which shock drove the 2023 inflation spike
- [ ] You can explain the event attribution table
- [ ] Ready for Day 12: Structural stability tests
