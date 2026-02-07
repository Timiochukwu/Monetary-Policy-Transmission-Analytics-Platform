# Week 2 · Day 9 — Forecast Error Variance Decomposition

> **What you will build:** `src/econometrics/fevd.py`
> **What you will produce:** 4 stacked-area plots, 4 CSV tables, key
> result table (MPR → Inflation)
> **Why it matters:** FEVD answers: **"What percentage of Inflation
> variance is explained by MPR shocks?"**  This is the *strength* of
> the transmission channel.  IRFs show the *shape*; FEVD shows the
> *magnitude*.

---

## 0  Pre-requisites

| Concept | Where covered |
|---------|---------------|
| IRF (Day 8) | Variance decomposition is derived from IRFs |
| Cholesky ordering | Day 7 + Day 8 guides |
| VAR(11) | Day 7 |

---

## 1  Theory Refresher

### 1.1  What is FEVD?

The **forecast error variance decomposition** partitions the variance
of the h-step-ahead forecast error into the shares attributable to each
structural shock.

For variable `i` at horizon `h`:

```
Var(forecast error of y_i at h)  =  Σ_{j=1}^K  contribution from shock j
```

Each contribution is expressed as a **fraction** (0 to 1), and they sum
to 1.0.

### 1.2  Interpretation

If FEVD shows that MPR shocks explain 18 % of Inflation variance at
horizon 24, this means:
- **18 % of the unpredictable variation** in Inflation (after
  conditioning on the VAR's information set) is due to MPR policy
  shocks.
- The remaining 82 % comes from ExchangeRate shocks (35 %), Inflation's
  own shocks (44 %), and M2 shocks (3 %).

### 1.3  Ordering sensitivity

Like IRFs, FEVD is **ordering-dependent** when using Cholesky
identification.  Variables ordered first mechanically explain more
variance (because they absorb the contemporaneous correlations).

This is why MPR is first in the ordering — it reflects the institutional
reality that the CBN moves first, and markets react.

---

## 2  Reading the FEVD Plots

Each stacked-area plot shows the variance decomposition of **one**
variable over time.  Focus on `fevd_Inflation.png`:

### 2.1  The Inflation variance decomposition

The Y-axis ranges from 0 to 1 (100 %).  Each coloured band represents
one shock source.  At horizon 24:

| Shock source | Share of Inflation variance |
|--------------|------------------------------|
| **MPR** | 18.4 % |
| **ExchangeRate** | 34.6 % |
| **M2** | 3.3 % |
| **Inflation (own)** | 43.7 % |

Interpretation:
- **ExchangeRate dominates** (34.6 %) — the pass-through channel is the
  strongest.
- **MPR is second** (18.4 %) — a meaningful policy lever.
- **Inflation's own shocks** (43.7 %) capture supply shocks, fiscal
  shocks, and other unmodeled drivers.

### 2.2  Evolution over time

At short horizons (1–3 months), Inflation's own shocks explain ~97 %
of variance (prices are sticky).  By month 12, MPR's share rises to
16 %, and ExchangeRate's share is 34 %.  The shares stabilise after
month 18.

---

## 3  FEVD Tables

The script prints variance shares at selected horizons:

```
  Response variable: Inflation
  Horizon     MPR  ExchangeRate   M2    Inflation    Sum
      1     0.0151    0.0149    0.0002    0.9698  1.0000
      3     0.0466    0.1086    0.0043    0.8404  1.0000
      6     0.0859    0.2230    0.0171    0.6740  1.0000
     12     0.1610    0.3429    0.0291    0.4670  1.0000
     24     0.1842    0.3457    0.0334    0.4367  1.0000
```

The `Sum` column verifies that shares add to 1.0 (rounding errors may
produce ~1.0000).

---

## 4  Key Result Table

The script prints a dedicated table for the **MPR → Inflation**
transmission:

```
  KEY RESULT — MPR → Inflation transmission

     Horizon     MPR → Inflation  (% of variance)
          1               1.51%
          3               4.66%
          6               8.59%
         12              16.10%
         24              18.42%
```

This is the **headline number** for your thesis abstract:

> "MPR policy shocks explain 18.4 % of Inflation variance at the
> 24-month horizon, confirming a statistically and economically
> significant transmission channel."

---

## 5  Code Walkthrough

### 5.1  Compute FEVD

```python
# fevd.py — compute method
self.fevd_obj = self.res.fevd(periods)
self.decomp   = self.fevd_obj.decomp   # shape: (K, T, K)
```

`decomp` is a 3D array:
- `decomp[i, h, j]` = fraction of variance of variable `i` at horizon
  `h+1` explained by shock `j`

Note: horizon is 1-indexed in the table (horizon 1 = `decomp[:, 0, :]`).

### 5.2  Plot one variable

```python
response_idx = self.ordering.index('Inflation')
data = self.decomp[response_idx, :, :]   # shape: (T, K)

ax.stackplot(x, data.T, labels=self.ordering, colors=color_list)
```

This creates a stacked-area chart where each shock is a coloured band.

---

## 6  Run the Script

```bash
cd /path/to/Monetary-Policy-Transmission-Analytics-Platform
MPLBACKEND=Agg python src/econometrics/fevd.py
```

### Output files

| File | Contents |
|------|----------|
| `results/fevd/fevd_Inflation.png` | Stacked-area: Inflation variance decomposition |
| `results/fevd/fevd_MPR.png` | MPR variance decomposition |
| `results/fevd/fevd_ExchangeRate.png` | ExchangeRate variance decomposition |
| `results/fevd/fevd_M2.png` | M2 variance decomposition |
| `results/fevd/fevd_*.csv` | 4 CSV tables (one per variable) |

---

## 7  Common Questions

**Q: Why does ExchangeRate explain more Inflation variance than MPR?**

A: Nigeria is a highly import-dependent economy.  The exchange-rate
pass-through to consumer prices (food, fuel, durable goods) is
immediate and large.  MPR affects Inflation indirectly (via credit
conditions and expectations), which takes longer and is weaker.

**Q: Is 18.4 % a "good" transmission strength?**

A: Compare to the literature:
- Advanced economies (US, UK): 20–30 %
- Emerging markets (Brazil, South Africa): 10–25 %
- Nigeria (Adebiyi & Mordi, 2012): ~15 %

18.4 % is in line with peer emerging markets.  It confirms that the
transmission channel exists but is not the dominant driver of Inflation.

**Q: Why does Inflation's own-shock share remain so high (44 %)?**

A: The VAR omits key supply-side drivers:
- Oil prices (Nigeria is oil-dependent)
- Food supply shocks (droughts, insecurity)
- Fiscal deficits (quasi-fiscal CBN lending)

These appear in the residuals as "Inflation shocks."  A richer model
would add oil prices and fiscal balance as exogenous variables.

**Q: Should I report the 12-month or 24-month FEVD?**

A: Report both.  The 12-month number (16.1 %) captures the medium-run
transmission that policymakers care about.  The 24-month number
(18.4 %) is the long-run steady-state share.

---

## 8  Thesis Phrasing

Use this language in your Results chapter:

> "The forecast error variance decomposition reveals that MPR policy
> shocks account for 18.4 % of Inflation variance at the 24-month
> horizon, while exchange-rate shocks contribute 34.6 %.  The dominance
> of the exchange-rate channel reflects Nigeria's dependence on
> imported consumer goods, consistent with Chit & Okafor (2021).  The
> non-trivial contribution of MPR confirms that monetary policy remains
> an effective inflation-management tool, despite the presence of
> structural supply-side constraints."

---

## 9  Comparison with IRF (Day 8)

| Tool | Question answered | Output |
|------|-------------------|--------|
| **IRF** | How does Inflation *respond* to an MPR shock over time? | Time-series plot (pp change) |
| **FEVD** | How much of Inflation *variance* is explained by MPR? | Percentage (18.4 %) |

Both are complementary:
- IRF shows the *shape* and *timing* of the response (peak at month 12).
- FEVD shows the *strength* of the channel (18 % of variance).

Together, they tell the full story.

---

## 10  Checklist

- [ ] `fevd.py` runs without errors
- [ ] All four FEVD plots are saved in `results/fevd/`
- [ ] You have noted the **MPR → Inflation share** at 12 and 24 months
- [ ] You can explain why ExchangeRate dominates in the Nigerian context
- [ ] You understand the difference between IRF and FEVD
- [ ] Ready for Day 10: +100 bps policy shock simulation
