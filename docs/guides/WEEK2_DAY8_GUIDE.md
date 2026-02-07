# Week 2 · Day 8 — Impulse Response Functions

> **What you will build:** `src/econometrics/irf.py`
> **What you will produce:** 4 IRF plots (one per shock), 4 CSV tables
> **Why it matters:** IRFs are the **dynamic story** of monetary policy
> transmission.  They answer: "If the CBN raises MPR by 1 percentage
> point, how does Inflation respond over the next 24 months?"  This is
> the headline result for your thesis.

---

## 0  Pre-requisites

| Concept | Where covered |
|---------|---------------|
| Cholesky identification / ordering | Day 7 guide + `identification_strategy.md` |
| VAR(11) estimation | Day 7 (the IRF engine) |
| Orthogonalised vs generalised IRFs | This guide, §1.3 |

---

## 1  Theory Refresher

### 1.1  What is an IRF?

An **impulse response function** traces the effect of a one-time shock
to one variable on all variables in the system, over time.

In the VAR framework:

```
y_t  =  A₁ y_{t-1}  +  A₂ y_{t-2}  +  …  +  Aₚ y_{t-p}  +  u_t
```

where **u** is a vector of reduced-form errors.  An IRF asks: "What
happens to **y** if we inject a shock into one equation's error term?"

### 1.2  The identification problem

The reduced-form errors **u** are correlated (the covariance matrix Σ
is not diagonal).  We cannot interpret them as independent structural
shocks.  **Cholesky decomposition** solves this by imposing a recursive
structure:

```
u_t  =  P ε_t
```

where **P** is lower-triangular (from the Cholesky decomposition of Σ)
and **ε** are orthogonal structural shocks.  The ordering of variables
in **y** determines the structure of **P**.

Our ordering:  **MPR → ExchangeRate → M2 → Inflation**

This means:
- A shock to MPR is exogenous (not contaminated by same-period shocks
  to other variables).
- A shock to ExchangeRate is exogenous *conditional on MPR*.
- And so on down the chain.

### 1.3  Orthogonalised vs generalised IRFs

| Type | Identification | Ordering-dependent? |
|------|---------------|---------------------|
| **Orthogonalised** (Cholesky) | Recursive (lower-triangular) | **Yes** — results change if you reorder |
| **Generalised** (Pesaran & Shin, 1998) | Uses historical shock correlations | No — invariant to ordering |

We use **orthogonalised IRFs** because monetary policy has a clear
institutional hierarchy: the CBN sets MPR first, then markets react.
Generalised IRFs would dilute this structure.

---

## 2  Reading the IRF Plots

Each plot shows the response of **all four variables** to a shock in
**one** variable.  Focus on `irf_MPR_shock.png`:

### 2.1  The MPR → Inflation path

The bottom-right subplot shows **Inflation's response to an MPR shock**.

Key features to note:
- **Impact effect** (month 0): Often close to zero — prices are sticky.
- **Peak response** (month 3–12): The maximum effect on Inflation.
- **Long-run effect** (month 24): May decay toward zero (if no
  cointegration) or stabilise at a non-zero level (if cointegration
  exists).

In the output, the peak Inflation response to a +1 SD MPR shock is
~0.52 pp at month 12.  This means a 1 percentage point MPR hike
reduces Inflation by about 0.5 pp after one year (if the shock SD ≈ 1).

### 2.2  The puzzle sign (if Inflation rises)

If Inflation *increases* after an MPR shock, this is the **price
puzzle** (Sims, 1992).  It arises when:
- The VAR omits a forward-looking variable (commodity prices,
  expectations).
- The lag order is too short.
- The identification is wrong (MPR responds to Inflation
  contemporaneously, violating the ordering).

If you see a price puzzle, discuss it in your thesis and propose
robustness checks (e.g., adding oil prices as an exogenous variable).

---

## 3  IRF Tables

The script prints IRF values at selected horizons (0, 1, 3, 6, 12, 24).
Example:

```
  Shock: MPR  (responses at selected horizons)
  Horizon     MPR  ExchangeRate   M2    Inflation
      0      0.40     -0.00      0.00      0.12
      1      0.40      1.46      0.00      0.23
      6      0.32      5.18      0.10      0.48
     12      0.24      9.52      0.48      0.52
     24      0.22      1.84      2.31     -0.27
```

Interpretation:
- **MPR (column 1)**: After an MPR shock, MPR itself remains elevated
  for ~12 months, then decays.  This is the **persistence** of the
  policy rate.
- **ExchangeRate (column 2)**: The naira depreciates sharply in the
  first 6–12 months (5–10 units) after an MPR hike.  This is the UIP
  channel.
- **Inflation (column 4)**: Peaks at month 12 (+0.52 pp), then declines.

---

## 4  Code Walkthrough

### 4.1  Compute IRFs

```python
# irf.py — compute method
self.irf_obj = self.res.irf(periods)
self.irfs    = self.irf_obj.orth_irfs   # shape: (T+1, K, K)
```

`orth_irfs` is a 3D array:
- `irfs[t, i, j]` = response of variable `i` at time `t` to shock `j`

### 4.2  Plot one shock

```python
shock_idx = self.ordering.index('MPR')
for i, response_var in enumerate(self.ordering):
    y = self.irfs[:, i, shock_idx]     # extract time series
    ax.plot(x, y, color='#2E86AB', linewidth=2.5)
```

This plots the response of each variable (rows of `irfs`) to the MPR
shock (column `shock_idx`).

---

## 5  Run the Script

```bash
cd /path/to/Monetary-Policy-Transmission-Analytics-Platform
MPLBACKEND=Agg python src/econometrics/irf.py
```

### Output files

| File | Contents |
|------|----------|
| `results/irf/irf_MPR_shock.png` | 4-panel: all responses to MPR shock |
| `results/irf/irf_ExchangeRate_shock.png` | All responses to ER shock |
| `results/irf/irf_M2_shock.png` | All responses to M2 shock |
| `results/irf/irf_Inflation_shock.png` | All responses to Inflation shock |
| `results/irf/irf_*.csv` | 4 CSV tables (one per shock) |

---

## 6  Common Questions

**Q: The Inflation response to MPR is positive for the first 6 months.
Is this the price puzzle?**

A: Yes.  This is common in Nigerian VARs.  Possible causes:
- Commodity-price channel omitted (oil, food).
- The CBN may raise MPR *in response* to rising Inflation, and the VAR
  picks up the correlation.
- Sample includes the 2016 devaluation and 2020 COVID shock, which
  created simultaneous MPR hikes and Inflation spikes.

Solutions: add oil prices as exogenous, use sign restrictions instead
of Cholesky, or split the sample pre/post-2020.

**Q: Why does the ExchangeRate response to MPR increase for 12 months
then collapse?**

A: The initial depreciation is the UIP channel (higher MPR → capital
inflows → naira appreciation is the *theory*, but Nigeria has capital
controls, so the data shows the opposite).  The collapse at month 24
may be a sample-end artifact or a structural break.

**Q: Should I use 24 months or 36 months?**

A: 24 is standard for monthly macro data.  With 181 observations and a
VAR(11), effective sample is 170.  A 36-month horizon would be
extrapolating too far.  Stick to 24.

**Q: How do I convert the IRF to a policy scenario ("+100 bps MPR
hike")?**

A: Day 10.  The IRF is scaled by the shock SD (from the Cholesky
decomposition).  For a +100 bps hike, you scale the IRF by
`100 / (MPR shock SD * 100)`.  This will be automated in Day 10.

---

## 7  Thesis Phrasing

Use this language in your Results chapter:

> "A one-standard-deviation shock to MPR (approximately 0.4 percentage
> points) leads to a peak Inflation response of 0.52 percentage points
> after 12 months, consistent with the quantity-theory and
> exchange-rate pass-through channels identified in the ARDL and FEVD
> analyses.  The delayed peak reflects price stickiness in Nigeria's
> consumer markets (Mordi et al., 2007).  The subsequent decay toward
> month 24 suggests that the effect is transitory, as predicted by the
> New Keynesian framework."

---

## 8  Checklist

- [ ] `irf.py` runs without errors
- [ ] All four IRF plots are saved in `results/irf/`
- [ ] You have identified the **peak Inflation response** to MPR
- [ ] You can explain the Cholesky ordering in one sentence
- [ ] If a price puzzle exists, you have noted it for the Discussion
- [ ] Ready for Day 9: FEVD
