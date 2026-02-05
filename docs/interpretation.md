# Week 1 Results — Nigerian Economic Context

## Purpose

This document translates the raw econometric outputs from Days 3 and 4
into a narrative that an MSc examiner (or thesis reader) will expect.
Each section maps a statistical finding to its macro-economic meaning
within the **Nigerian monetary-policy transmission channel**.

---

## 1 Integration Orders — What They Mean Economically

| Variable | Order | Economic reading |
|----------|-------|-----------------|
| MPR | I(1) | The CBN policy rate drifts over time; it is not mean-reverting on a monthly horizon. This is expected: central banks adjust rates *persistently* in response to shocks and only reverse gradually. |
| Exchange Rate | I(1) | The naira/dollar rate follows a unit-root process. Nigeria's FX market has experienced persistent regime shifts (2016 devaluation, 2023 unification) that produce stochastic-trend behaviour. |
| M2 | I(0) | Broad money is stationary. The CBN's monetary-aggregates framework imposes implicit targets; money creation and destruction (loan repayments, reserve requirements) keep M2 mean-reverting. |
| Inflation | I(0) | Headline CPI is stationary around its (elevated) mean. While Nigeria's inflation is structurally high, it does not exhibit an explosive trend — it cycles around an equilibrium set by fiscal and monetary fundamentals. |

### Why mixed orders matter

A mix of I(0) and I(1) series rules out pure Johansen VECM (which
requires *all* variables to be I(1)). It validates the choice of
**ARDL bounds-testing**, which is designed precisely for this situation
(Pesaran, Shin & Smith, 2001).

---

## 2 Engle-Granger — The Inflation–M2 Link

### Result
One cointegrated pair detected at 5 %: **Inflation (dependent) on M2**.
The pair Inflation on ExchangeRate is borderline (p ≈ 0.077).

### Economic interpretation
The Inflation–M2 cointegrating relationship is the **quantity-theory
channel**: excess money creation eventually feeds through to prices.
In Nigeria, where the CBN has periodically monetised fiscal deficits
(quasi-fiscal operations, Treasury bill purchases), this channel is
well-documented in the literature (Eze & Egoh, 2019; Nwanna & Ayodele,
2020).

The *direction* matters: Inflation is the dependent variable, meaning
that M2 *leads* price movements in the long run — consistent with
theory that monetary expansion precedes inflation with a lag.

### Borderline ExchangeRate–Inflation
The near-significant ExchangeRate → Inflation pair reflects the
**exchange-rate pass-through** channel (Chit & Okafor, 2021). Nigeria
imports a large share of consumer goods, so a naira depreciation raises
domestic prices. The p-value of 0.077 is on the boundary; the more
powerful ARDL bounds test in Week 2 may confirm this relationship.

### Why not more cointegrated pairs?
Engle-Granger is a *two-step* procedure with finite-sample bias
(Johansen, 1991). With only 181 observations and structural breaks, it
has low power. The ARDL test (Week 2) is specifically designed to
handle small samples and mixed integration orders.

---

## 3 Johansen — Multivariate Evidence

### Full-system results
- **Trace test** → rank 4 (all four eigenvalues significant).
- **Max-eigenvalue test** → rank 1 (only the largest eigenvalue
  significant).

### Interpreting the disagreement
When trace and max-eigenvalue disagree, the standard guidance
(Lutkepohl, 2005, §8.1) is to **favour the max-eigenvalue test**
because:

1. The max-eigenvalue test has better small-sample size properties.
2. The trace test is more sensitive to outliers in the smallest
   eigenvalues (the r = 2 and r = 3 rejections here are marginal).

**Conclusion from Johansen: rank = 1** — one long-run equilibrium
relationship in the full system.

### I(1)-only robustness check
Running Johansen on MPR and ExchangeRate alone yields **rank = 0**: no
cointegration between the two I(1) series. This is consistent with
uncovered interest parity holding only approximately and with Nigeria's
capital-controls regime during parts of the sample.

### Caveat: mixed-order system
Johansen assumes all series are I(1). Feeding I(0) variables (M2,
Inflation) into the full-system test is technically a violation. The
rank-4 trace result should therefore be interpreted cautiously; the
max-eigenvalue rank of 1 and the EG evidence of one cointegrating pair
provide the more reliable signal.

---

## 4 Model-Selection Memo

| Evidence | Signal |
|----------|--------|
| Mixed I(0) / I(1) | ARDL preferred over VECM |
| EG: 1 cointegrated pair (Inflation ~ M2) | Long-run equilibrium exists |
| Johansen max-eig: rank 1 | Confirms single equilibrium |
| EG borderline: Inflation ~ ExchangeRate | Pass-through plausible; needs ARDL |
| I(1)-only Johansen: rank 0 | MPR and ER are not cointegrated pairwise |

**Week 2 plan:**

1. ARDL bounds-test on the full 4-variable system to pin down the
   long-run equation and confirm pass-through.
2. VAR on levels (the Johansen evidence justifies it) with Cholesky
   identification in the specified ordering.
3. IRFs and FEVD to quantify the MPR → Inflation transmission speed
   and magnitude.
4. +100 bps shock simulation for the policy-scenario deliverable.

---

## 5 References

- Eze, L. C. & Egoh, E. C. (2019). *Monetary policy and inflation in
  Nigeria*. Journal of Economics and Management, 44, 12–28.
- Lutkepohl, H. (2005). *New introduction to multiple time series
  analysis*. Springer.
- Nwanna, I. I. & Ayodele, T. S. (2020). *Money supply and inflation
  dynamics in Nigeria*. Research Journal in Social Sciences, 8(1).
- Pesaran, M. H., Shin, Y. & Smith, R. J. (2001). Bounds testing
  approaches to the analysis of level relationships. *Journal of
  Applied Econometrics*, 16(3), 289–326.
