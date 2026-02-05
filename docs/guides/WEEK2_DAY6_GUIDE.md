# Week 2 · Day 6 — ARDL Bounds Testing

> **What you will build:** `src/econometrics/ardl.py`
> **What you will produce:** Bounds-test decision, long-run equation,
> ECM (short-run), residual diagnostics
> **Why it matters:** Day 3 gave you mixed I(0)/I(1) orders. ARDL is
> the *only* standard framework that handles this without differencing
> or dropping variables.

---

## 0  Pre-requisites

| Concept | Where covered |
|---------|---------------|
| Integration orders (mixed) | Day 3 guide |
| Cointegration (why it matters) | Day 4 guide |
| ECM / error-correction | Day 4 guide |
| `docs/interpretation.md` | Read fully before starting |

Run this to confirm Day 4 outputs exist:
```bash
ls results/stationarity/integration_orders.csv
```

---

## 1  Theory — Why ARDL?

### 1.1  The problem
Day 3 produced: MPR = I(1), ExchangeRate = I(1), M2 = I(0), Inflation = I(0).

- **Johansen / VECM** requires *all* variables to be I(1). Unusable here.
- **Standard OLS on levels** is spurious if any variable is I(1).
- **Differencing everything** throws away the long-run information that
  cointegration tests confirmed exists.

### 1.2  The solution: ARDL bounds-testing
Pesaran, Shin & Smith (2001) showed that an Autoregressive Distributed
Lag model can test for a long-run relationship *regardless* of whether
the variables are I(0), I(1), or a mix — **without pre-testing for
unit roots**.

The two outputs you care about:

| Output | Meaning |
|--------|---------|
| **Bounds test** | Does a long-run equilibrium exist? (yes/no/inconclusive) |
| **ECM equation** | If yes, what IS that equilibrium, and how fast does the system return to it after a shock? |

---

## 2  The ARDL Model — Mechanics

### 2.1  Notation

The ARDL(p, q₁, q₂, q₃) for our system:

```
y_t = const + α₁ y_{t-1} + … + αₚ y_{t-p}         ← AR lags
    + β₁₀ x₁_t + β₁₁ x₁_{t-1} + … + β₁q₁ x₁_{t-q₁}   ← DL lags for MPR
    + β₂₀ x₂_t + … + β₂q₂ x₂_{t-q₂}                    ← DL lags for ExchangeRate
    + β₃₀ x₃_t + … + β₃q₃ x₃_{t-q₃}                    ← DL lags for M2
    + ε_t
```

Where: y = Inflation, x₁ = MPR, x₂ = ExchangeRate, x₃ = M2.

### 2.2  Lag selection

The code runs a **grid search**: every combination of
p ∈ {1,…,4} and q_i ∈ {1,…,4}, fitting an ARDL for each and recording
AIC.  256 models total.  The spec with the lowest AIC wins.

```python
# ardl.py — _select_lags
for p in range(1, max_p + 1):
    for qs in itertools.product(*q_ranges):     # q_ranges = [1..4] × 3
        order = dict(zip(self.regressors, qs))
        model = ARDL(self.endog, lags=p,
                     exog=self.exog, order=order, trend='c')
        res   = model.fit()
        if res.aic < best_aic:  ...             # update winner
```

> **Why q starts at 1, not 0:** The UECM reparameterisation (used for
> the bounds test) requires at least one distributed-lag term per
> regressor.

### 2.3  UECM — the reparameterisation

The ARDL is algebraically equivalent to an **Unrestricted Error
Correction Model** (UECM):

```
Δy_t = const
     + π  y_{t-1}                    ← speed of adjustment
     + θ₁ x₁_{t-1} + θ₂ x₂_{t-1} + θ₃ x₃_{t-1}   ← level regressors
     + Σ φⱼ  Δy_{t-j}               ← short-run AR
     + Σ ψᵢⱼ Δxᵢ_{t-j}             ← short-run DL
     + ε_t
```

Statsmodels has a dedicated `UECM` class that does this reparameterisation
automatically.  The code fits the AIC-selected ARDL as a UECM:

```python
from statsmodels.tsa.ardl import UECM

model = UECM(endog, lags=ar_lags, exog=exog, order=dl_order, trend='c')
res   = model.fit()
```

---

## 3  The Bounds Test — Decision Rule

### 3.1  What it tests

| | |
|---|---|
| **H₀** | No long-run relationship (coefficients on levels = 0) |
| **H₁** | Cointegration exists |

The F-statistic tests whether π = θ₁ = θ₂ = θ₃ = 0 jointly.

### 3.2  Critical-value bands

Pesaran (2001) derived *two* sets of critical values:

| Bound | Assumption | Interpretation |
|-------|-----------|----------------|
| **Lower** (I(0)) | All regressors are stationary | If F < lower → definitely no cointegration |
| **Upper** (I(1)) | All regressors are unit-root | If F > upper → definitely cointegrated |
| **Between** | Mixed | Inconclusive — need additional information |

### 3.3  The `case` parameter

`bounds_test(case=N)` selects which deterministic terms to include in
the critical-value calculation:

| Case | Deterministic terms | When to use |
|------|-------------------|-------------|
| 1 | None | Rarely |
| 2 | Restricted constant | Rarely |
| **3** | **Unrestricted constant** | **Default — use this** |
| 4 | Unrestricted constant + restricted trend | If data has a trend |
| 5 | Unrestricted constant + trend | If strong deterministic trend |

The code uses **case 3**.

### 3.4  Reading the result

```python
bt = res.bounds_test(case=3)
bt.stat            # F-statistic
bt.crit_vals       # DataFrame: lower/upper at 10/5/1/0.1 %
bt.p_values        # Series: lower_p, upper_p
```

Decision:
- `upper_p < 0.05` → **cointegration confirmed**
- `lower_p < 0.05 < upper_p` → **inconclusive**
- `lower_p > 0.05` → **no cointegration**

---

## 4  Extracting the Equations

### 4.1  Long-run multipliers

From the UECM parameters:

```
Long-run effect of x_i  =  θᵢ / (−π)
```

Where:
- `π` = coefficient on `y_{t-1}` (speed of adjustment, should be negative)
- `θᵢ` = coefficient on `x_i_{t-1}`

```python
# ardl.py — _extract_coefficients
pi    = params["Inflation.L1"]          # speed of adjustment
theta = params["MPR.L1"]               # level coefficient on MPR
long_run_MPR = theta / (-pi)           # long-run multiplier
```

### 4.2  Speed of adjustment

`π` measures how quickly the system corrects deviations from the
long-run equilibrium.  If π = −0.08, the system closes **8 % of the
gap per month** — so about 12 months to close half the gap.

- **Negative π** → stable, mean-reverting (good)
- **Positive π** → explosive (problem — re-check the model)

### 4.3  Short-run coefficients

Everything prefixed `D.` in the UECM params is a short-run effect:

| Parameter | Meaning |
|-----------|---------|
| `D.MPR.L0` | Effect of a *change* in MPR this month on Δ Inflation |
| `D.MPR.L1` | Effect of last month's MPR *change* |
| `D.Inflation.L1` | Short-run AR persistence in inflation |

---

## 5  Run the Script

```bash
cd /path/to/Monetary-Policy-Transmission-Analytics-Platform
MPLBACKEND=Agg python src/econometrics/ardl.py
```

### Output files

| File | Contents |
|------|----------|
| `results/ardl/ardl_bounds_test.csv` | F-stat + critical-value bands |
| `results/ardl/ardl_long_run.csv` | Long-run multipliers + speed of adjustment |
| `results/ardl/ardl_uecm_coefficients.csv` | Full UECM parameter table |
| `results/ardl/ardl_diagnostics.csv` | DW, Jarque-Bera |
| `results/ardl/ardl_residual_diagnostics.png` | 3-panel residual chart |

---

## 6  Interpreting Your Results

The bounds test will fall into one of three buckets.  Here is how to
write each in a thesis:

**If confirmed (F > upper at 5 %):**
> "The ARDL bounds F-statistic of X.XX exceeds the upper critical
> value of Y.YY at the 5 % significance level (Pesaran et al., 2001,
> Case 3). We therefore reject the null of no cointegration: a
> long-run equilibrium relationship exists among the four
> macroeconomic variables."

**If inconclusive (between bounds):**
> "The ARDL bounds F-statistic of X.XX falls between the lower
> (Y.YY) and upper (Z.ZZ) critical values at 5 %, yielding an
> inconclusive result. However, at the 10 % level the statistic
> exceeds the upper bound, providing tentative evidence of a
> long-run relationship. We proceed with the ECM specification
> subject to this caveat."

**If rejected (F < lower at 5 %):**
> "The ARDL bounds test fails to reject the null of no
> cointegration at the 5 % level (F = X.XX < lower bound Y.YY).
> The VAR in first differences is therefore the appropriate
> framework for Week 2 dynamics."

---

## 7  Common Questions

**Q: Why does AIC select only 4 AR lags when the VAR uses 11?**
A: ARDL and VAR optimise different objectives.  The VAR's 11 lags
capture dynamics across *all four equations*; the ARDL only needs
enough AR lags to whiten the residuals of the *Inflation* equation.

**Q: What does a DW close to 0 mean?**
A: Durbin-Watson near 0 indicates strong positive serial correlation
in the residuals.  In an ECM with slow adjustment (π ≈ −0.08), this
is expected.  It does not invalidate the bounds test or the long-run
coefficients, but it warns that the short-run dynamics may need more
lags.  The bounds test is robust to moderate serial correlation.

**Q: Why is the F-statistic computed on levels, not differences?**
A: The bounds test is specifically designed to work on the level
specification.  Differencing would remove the long-run information
that the test is looking for.

---

## 8  Checklist

- [ ] `ardl.py` runs without errors
- [ ] Bounds-test F-statistic and decision are printed
- [ ] Long-run equation (speed of adjustment + multipliers) is saved
- [ ] You can state the bounds-test decision in thesis language
- [ ] You understand what π = −0.08 means economically
- [ ] Residual diagnostics plot is saved
- [ ] Ready for Day 7: VAR model
