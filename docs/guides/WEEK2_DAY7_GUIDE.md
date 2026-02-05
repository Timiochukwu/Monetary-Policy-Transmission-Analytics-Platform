# Week 2 · Day 7 — VAR Model + Granger Causality

> **What you will build:** `src/econometrics/var_model.py`
> **What you will produce:** VAR coefficient tables, lag-selection
> table, Granger causality matrix
> **Why it matters:** The VAR is the engine that powers Days 8 and 9
> (IRF and FEVD).  Getting it right — especially the lag order and
> the Cholesky ordering — is critical for the policy-shock simulation
> in Day 10.

---

## 0  Pre-requisites

| Concept | Where covered |
|---------|---------------|
| Cholesky identification / ordering | `docs/identification_strategy.md` |
| Cointegration evidence (rank 1) | Day 4 guide + `interpretation.md` |
| ARDL bounds test result | Day 6 guide |
| Why VAR on levels | Day 4 guide §3.7 |

---

## 1  Theory Refresher

### 1.1  What is a VAR?

A VAR(p) is a system of equations where *each* variable is regressed
on its own past and the past of *every other* variable:

```
y_t  =  A₀  +  A₁ y_{t-1}  +  A₂ y_{t-2}  +  …  +  Aₚ y_{t-p}  +  u_t
```

Where **y** is a 4×1 vector `[MPR, ExchangeRate, M2, Inflation]ᵀ` and
each **Aᵢ** is a 4×4 coefficient matrix.

### 1.2  Levels vs differences

The Week 1 cointegration evidence (Johansen rank = 1, EG pair
Inflation ~ M2) justifies running the VAR **on levels**.  If we
differenced, we would lose the long-run equilibrium information that
drives the impulse responses.

### 1.3  Why the ordering matters

A VAR has *more unknowns than equations* (the covariance matrix of
**u** is not diagonal).  **Cholesky identification** resolves this by
imposing a recursive (triangular) structure on *contemporaneous*
shocks.

Our ordering:  **MPR → ExchangeRate → M2 → Inflation**

This means:
- MPR responds to nothing contemporaneously (it is set by the CBN)
- ExchangeRate responds to MPR *this month*, but not to M2 or Inflation
- M2 responds to MPR and ExchangeRate this month
- Inflation responds to all three this month

This ordering is justified in `docs/identification_strategy.md`.  **Do
not change it without a theoretical reason.**

---

## 2  Lag Selection

### 2.1  Why it matters

Too few lags → model misspecification, biased IRFs.
Too many lags → overfitting, noisy IRFs, fewer effective observations.

### 2.2  Information criteria

The code fits VAR(1) through VAR(12) and records four criteria:

| Criterion | Tendency | Notes |
|-----------|----------|-------|
| **AIC** | Selects more lags | Best for prediction / IRF |
| **BIC** | Penalises more lags | Best for inference |
| **HQIC** | Between AIC and BIC | Compromise |
| **FPE** | Final Prediction Error | Similar to AIC |

The script uses **AIC** as the primary selector.  Compare with BIC for
a sanity check.  If they agree, you have confidence.  If they disagree
by more than 2–3 lags, report both and note the trade-off.

### 2.3  Code

```python
# var_model.py — select_lags
self.model = VAR(self.df)
for lag in range(1, max_lags + 1):
    r = self.model.fit(lag)
    records.append({'lag': lag, 'AIC': r.aic, 'BIC': r.bic, …})
# pick the lag with minimum AIC
self.opt_lag = lag_table.loc[lag_table['AIC'].idxmin(), 'lag']
```

The table is printed with a `◄` marker on the AIC-optimal row.

---

## 3  Coefficient Tables

Each equation in the VAR has its own set of coefficients.  The code
prints them with significance stars:

```
  Equation: Inflation
  Variable                 Coeff    Std Err  t-stat   p-val  Sig
  L1.MPR                   0.0325    0.0662   0.490   0.6239
  L1.ExchangeRate          0.0028    0.0036   0.765   0.4443
  …
```

### 3.1  What to look for

- **L1.MPR in the Inflation equation** — the one-month pass-through
  from the policy rate to prices.  This is the *headline number* for
  your thesis.
- **Significance of own-lags** (e.g. `L1.Inflation` in the Inflation
  equation) — tells you about short-run inflation persistence.
- **Cross-variable lags** — tells you about transmission channels.

### 3.2  p-value computation

Statsmodels' `VARResults` may not expose p-values directly in all
versions.  The code falls back to computing them from t-statistics
using the t-distribution with df = T − pk − 1:

```python
try:
    pvalues = self.res.pvalues          # use if available
except AttributeError:
    from scipy.stats import t as t_dist
    df_resid = n - p * k - 1
    pvalues  = 2 * (1 - t_dist.cdf(np.abs(tvalues), df_resid))
```

---

## 4  Granger Causality

### 4.1  The idea

"X Granger-causes Y" means: **past values of X contain information
about future Y, beyond what Y's own past already tells you.**

It does *not* mean X *causes* Y in the structural sense — it is a
predictive statement.  But in the context of monetary policy
transmission, Granger causality from MPR to Inflation is strong
circumstantial evidence of the transmission channel.

### 4.2  The F-test

For each ordered pair (causing → caused):

| | |
|---|---|
| **H₀** | Past values of ``causing`` have *no* predictive power for ``caused`` |
| **H₁** | They do |

The test compares: unrestricted VAR (all lags of all variables) vs.
restricted VAR (lags of ``causing`` dropped from ``caused``'s equation).

### 4.3  Code + API note

```python
# var_model.py — granger_causality
gc = self.res.test_causality(
    caused, causing=causing, kind='f', signif=alpha)

# Result attributes:
#   gc.test_statistic   – F-statistic
#   gc.pvalue           – p-value
#   gc.conclusion       – dict with test details
```

> **API pitfall:** The keyword is `signif` (not `alpha`), the
> statistic is `test_statistic` (not `f_statistic`), and the
> p-value is `pvalue` (not `p_value`).  These differ from other
> statsmodels test results.

### 4.4  Reading the output

The table shows all 12 ordered pairs.  Focus on the **causal links
that match the transmission mechanism**:

| Link | Economic meaning |
|------|-----------------|
| MPR → Inflation | Direct monetary policy transmission |
| MPR → ExchangeRate | Interest-rate / UIP channel |
| ExchangeRate → Inflation | Exchange-rate pass-through |
| MPR → M2 | Credit channel |

If MPR → Inflation is *not* significant in Granger, it does not mean
transmission does not exist — it may operate with a lag longer than
the VAR captures, or be nonlinear.  The IRFs (Day 8) will give a
richer picture.

---

## 5  Run the Script

```bash
cd /path/to/Monetary-Policy-Transmission-Analytics-Platform
python src/econometrics/var_model.py
```

### Output files

| File | Contents |
|------|----------|
| `results/var/var_lag_selection.csv` | AIC/BIC/HQIC/FPE for lags 1–12 |
| `results/var/var_coefficients.csv` | Full coefficient matrix |
| `results/var/var_granger_causality.csv` | 12-pair Granger table |
| `results/var/var_summary.csv` | Metadata (obs, optimal lag, ordering) |

---

## 6  Common Questions

**Q: Why does AIC select 11 lags when BIC selects 3?**
A: AIC penalises model complexity less than BIC.  With 181
observations, 11 lags consume a large fraction of the sample.  For
IRFs and policy simulations, AIC is preferred (captures dynamics
better).  If you are writing a methods section, note both and explain
why AIC was chosen.

**Q: The Granger test says MPR does NOT cause Inflation. Is that a
problem?**
A: Not necessarily.  Granger causality is a linear, in-sample,
lag-specific test.  The transmission may be:
- Nonlinear (captured better by regime-switching models)
- Delayed beyond the VAR horizon
- Mediated through other variables (MPR → ER → Inflation)

The IRFs (Day 8) will reveal the dynamic path even if the Granger
test is insignificant.

**Q: Should I run the VAR on differences instead?**
A: No — the Johansen evidence (rank 1) and the ARDL bounds-test
result support running on levels.  Differencing would remove the
equilibrium information.  If you are unsure, run both and compare
IRFs.

**Q: What does `var_coefficients.csv` look like?**
A: Rows are lagged variable names (e.g. `L1.MPR`, `L2.MPR`, …,
`L11.MPR`), columns are equations (`MPR`, `ExchangeRate`, `M2`,
`Inflation`).  Each cell is the coefficient of that lagged variable
in that equation.

---

## 7  Checklist

- [ ] `var_model.py` runs without errors
- [ ] Lag-selection table is printed — note AIC vs BIC disagreement
- [ ] Granger causality table is printed — identify the significant links
- [ ] You know which link is the *headline* transmission channel
- [ ] All four output CSVs are saved in `results/var/`
- [ ] You can explain why the VAR runs on levels, not differences
- [ ] Ready for Day 8: Impulse Response Functions
