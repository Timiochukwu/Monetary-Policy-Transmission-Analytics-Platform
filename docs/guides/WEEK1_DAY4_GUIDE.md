# Week 1 · Day 4 — Cointegration Testing

> **What you will build:** `src/econometrics/cointegration_tests.py`
> **What you will produce:** EG pairwise table, Johansen rank tables, residual plot
> **Why it matters:** After Day 3 classified variables as I(0) or I(1),
> Day 4 asks the next question — *do any of them move together in the
> long run?*  If yes, an error-correction model is needed; if not,
> differences are sufficient.

---

## 0  Pre-requisites

| Concept | Where it was introduced |
|---------|------------------------|
| I(0) vs I(1) | Day 3 guide |
| ADF / KPSS consensus | Day 3 guide |
| Nigerian macro variables | Day 1 guide + `docs/identification_strategy.md` |
| Statsmodels environment | Day 3 (already installed) |

Run this first to make sure Day 3 outputs exist:

```bash
ls results/stationarity/
# You should see: integration_orders.csv, unit_root_tests_levels.csv, …
```

---

## 1  Theory Refresher — What Is Cointegration?

Two (or more) I(1) series are **cointegrated** if a *linear combination*
of them is I(0).  Economically: both variables wander randomly, but a
specific relationship between them is always pulled back to equilibrium.

**Example:** MPR and Inflation might both trend upward (unit root), but
the *gap* between them (after adjusting for a long-run multiplier) stays
bounded — a **long-run equilibrium**.

### Why we need two tests

| Test | Scope | Strengths | Weaknesses |
|------|-------|-----------|------------|
| Engle-Granger (EG) | Pairwise | Simple, intuitive | Ignores cross-equation info; biased in small samples |
| Johansen | Multivariate | Efficient ML estimator; detects rank directly | Sensitive to I(0) contamination; assumes all I(1) |

We run **both** and cross-check.

---

## 2  Engle-Granger — Step by Step

### 2.1  The two steps

```
Step 1:  OLS    y_t  =  α  +  β·x_t  +  e_t          (spurious if not coint.)
Step 2:  ADF    on  e_t                                 (if stationary → coint.)
```

If the residuals `e_t` are stationary, the two series share a
long-run equilibrium.  The residual is the **equilibrium error** —
it measures how far the system is from its long-run relationship at
each point in time.

### 2.2  Null hypothesis

> H₀ : no cointegration  (residuals are I(1))
> H₁ : cointegration exists  (residuals are I(0))

Reject H₀ when **p < 0.05**.

### 2.3  Direction matters

EG is *asymmetric*: the choice of dependent variable affects the
result.  That is why the code tests **every ordered pair** (12 pairs
for 4 variables).  A pair (A → B) may be cointegrated while (B → A)
is not; the cointegrated direction tells you which variable adjusts
to restore equilibrium.

### 2.4  Code walkthrough

```python
# src/econometrics/cointegration_tests.py  —  _run_eg method

from statsmodels.tsa.stattools import coint          # (1)

t_stat, p_val, crit = coint(y.values, x.values)     # (2)

return {
    …
    "cointegrated": bool(p_val < 0.05),              # (3)
}
```

| Line | What happens |
|------|-------------|
| (1) | `coint()` performs the full two-step EG internally: OLS → ADF on residuals. |
| (2) | Returns the ADF t-statistic on residuals, a MacKinnon p-value, and critical values at 1/5/10 %. |
| (3) | Wrap in `bool()` because `coint()` returns numpy scalars; `numpy.bool_` fails Python `is True` identity checks. |

### 2.5  Reading the results

```
Dependent        Independent        t-stat  p-value  Result
MPR              ExchangeRate      -2.6269   0.2269  Not coint.   ✗
…
Inflation        M2                -3.4217   0.0400  Cointegrated ✓
```

Only **Inflation ~ M2** passes at 5 %.  See `docs/interpretation.md`
for the economic story (quantity-theory channel).

---

## 3  Johansen — Step by Step

### 3.1  The idea

Johansen estimates a VAR and extracts **eigenvalues** of a key matrix.
Each eigenvalue corresponds to a potential cointegrating vector.
A *zero* eigenvalue means no equilibrium; a *positive* eigenvalue
means one exists.  In practice we test whether each eigenvalue is
statistically different from zero.

### 3.2  Two test statistics

| Statistic | Formula | Intuition |
|-----------|---------|-----------|
| Trace | λ_trace(r) = −T Σ ln(1 − λᵢ) for i > r | Tests H₀: rank ≤ r against H₁: rank > r.  Uses *all* remaining eigenvalues. |
| Max-eigenvalue | λ_max(r) = −T ln(1 − λᵣ₊₁) | Tests H₀: rank = r against H₁: rank = r + 1.  Uses only the *next* eigenvalue. |

### 3.3  Determining the rank

Sequential procedure (start at r = 0):

1. If test stat > CV at 5 % → reject H₀: rank = r, increment r.
2. Stop at the first r where we *fail* to reject.
3. That r is the estimated cointegration rank.

### 3.4  `coint_johansen` API — the key details

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen   # NOT .johansen

result = coint_johansen(data, det_order, nlags)

# result attributes:
#   .eig   – eigenvalues (descending)          shape (p,)
#   .lr1   – trace statistics (pre-computed)   shape (p,)
#   .lr2   – max-eig statistics                shape (p,)
#   .cvt   – trace CVs                         shape (p, 3)  cols: 10%, 5%, 1%
#   .max_eig_stat_crit_vals – max-eig CVs      shape (p, 3)  cols: 10%, 5%, 1%
```

> **Common pitfall:**  The old `statsmodels.tsa.vector_ar.johansen`
> module was removed in 0.14.x.  Always import from `.vecm`.
>
> **Another pitfall:**  Max-eigenvalue critical values are in
> `max_eig_stat_crit_vals`, *not* `cvx`.  The trace CVs are in `cvt`.

### 3.5  `det_order` — what deterministic terms to include

| Value | Deterministic terms | When to use |
|-------|-------------------|-------------|
| −1 | None | Rarely — only if you are sure there is no intercept |
| 0 | Constant | Default; most macro applications |
| 1 | Constant + trend | If the data has a clear deterministic trend *after* differencing |

The code uses `det_order = 0` (constant only).

### 3.6  Reading the Johansen output

```
  r      Trace stat     CV 5 %  Decision
  0          62.879     47.855  Reject  H₀: rank=0    ← eigenvalue 1 is significant
  1          30.947     29.796  Reject  H₀: rank=1    ← marginal
  2          15.768     15.494  Reject  H₀: rank=2    ← very marginal
  3           4.606      3.841  Reject  H₀: rank=3
  → Trace test rank: 4

  r     MaxEig stat     CV 5 %  Decision
  0          31.933     27.586  Reject  H₀: rank=0
  1          15.179     21.131  Fail to reject        ← stop here
  → Max-eig rank: 1
```

The trace test says rank 4; the max-eigenvalue test says rank 1.
When they disagree, **prefer max-eigenvalue** (Lutkepohl, 2005) —
it has better small-sample properties.  The rank-1 result also matches
the single EG cointegrating pair.

### 3.7  I(1)-only robustness

The code runs a second Johansen on MPR and ExchangeRate only (the two
I(1) variables).  Johansen is only valid for I(1) series, so this is
the *theoretically correct* application.  Result: **rank 0** — MPR and
the exchange rate do not share a long-run equilibrium on their own.

---

## 4  Residual Plots — Visual Confirmation

For each cointegrated EG pair, the code plots the OLS residuals with a
± 1σ band.  **Stationary residuals** (mean-reverting, no trend) confirm
the long-run equilibrium visually.

```python
# The residual is simply:
X   = sm.add_constant(x.values)
ols = sm.OLS(y.values, X).fit()
resid = ols.resid                        # the equilibrium error
```

Look for:
- Mean close to zero
- No upward or downward trend
- Occasional large deviations that *come back* (mean-reversion)

---

## 5  Run the Script

```bash
cd /path/to/Monetary-Policy-Transmission-Analytics-Platform

# Make sure Day 3 outputs exist first
python src/econometrics/stationarity_tests.py

# Then run Day 4
MPLBACKEND=Agg python src/econometrics/cointegration_tests.py
```

### Output files produced

| File | Contents |
|------|----------|
| `results/stationarity/engle_granger_pairwise.csv` | 12-row table: all ordered pairs |
| `results/stationarity/johansen_all.csv` | Full-system Johansen rank table |
| `results/stationarity/johansen_i1_only.csv` | MPR + ExchangeRate only |
| `results/stationarity/eg_residuals.png` | Equilibrium-error plot(s) |

---

## 6  Common Questions

**Q: Why does Johansen give a different rank than EG?**
A: Different tests, different power.  EG is pairwise and biased
downward in small samples.  Johansen is multivariate but sensitive to
I(0) contamination.  The intersection of evidence (rank 1, one EG pair)
is the reliable signal.

**Q: My Johansen trace and max-eigenvalue disagree. Which do I trust?**
A: Max-eigenvalue.  See §3.6 above and Lutkepohl (2005, §8.1).

**Q: Should I worry about the I(0) variables in the Johansen test?**
A: Yes — that is exactly why we also run the I(1)-only robustness
check and why ARDL is the primary Week 2 framework.

**Q: What does `det_order` actually change?**
A: The constant shifts the critical values.  Using `det_order = 0`
(constant) is the standard choice for macro data in levels.

---

## 7  Checklist

- [ ] `integration_orders.csv` exists from Day 3
- [ ] `cointegration_tests.py` runs without errors
- [ ] `engle_granger_pairwise.csv` created — 12 rows
- [ ] `johansen_all.csv` and `johansen_i1_only.csv` created
- [ ] `eg_residuals.png` saved (only if ≥ 1 cointegrated pair)
- [ ] Read `docs/interpretation.md` for the economic narrative
- [ ] You can explain *why* ARDL is preferred over VECM here
