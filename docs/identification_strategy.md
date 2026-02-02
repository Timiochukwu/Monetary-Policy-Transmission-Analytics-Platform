# VAR Identification Strategy: Cholesky Ordering Justification

## Executive Summary

This document provides the **academic and economic rationale** for the Cholesky ordering used in structural VAR identification:

```
MPR → Exchange Rate → Money Supply (M2) → Inflation
```

This ordering follows established monetary policy VAR literature (Christiano et al., 1999; Kim & Roubini, 2000) adapted to **small open economy** context (Nigeria).

---

## 1. The Identification Problem

### 1.1 Structural vs. Reduced-Form VAR

**Reduced-form VAR** (estimated):
```
y_t = A₁y_{t-1} + ... + Aₚy_{t-p} + ε_t
```
- Residuals `ε_t` are **contemporaneously correlated**
- Cannot isolate individual structural shocks

**Structural VAR** (theoretical):
```
y_t = B₁y_{t-1} + ... + Bₚy_{t-p} + u_t
```
- Shocks `u_t` are **orthogonal** (uncorrelated)
- Economically interpretable

**Problem**: Need to recover `u_t` from `ε_t`.

### 1.2 Cholesky Decomposition Solution

Assume **recursive structure**:
```
ε_t = P·u_t
```
Where `P` is lower-triangular matrix from `Σ_ε = PP'`.

**Implication**: Variables ordered first are **contemporaneously exogenous** to those below.

---

## 2. Cholesky Ordering for Nigeria

### 2.1 Full Ordering

```
1. Monetary Policy Rate (MPR)
2. Exchange Rate (NGN/USD)
3. Money Supply (M2)
4. Inflation (CPI)
```

### 2.2 Economic Justification

#### **Position 1: MPR (Most Exogenous)**

**Why first?**
1. **Policy exogeneity**: CBN Monetary Policy Committee (MPC) meets bi-monthly and sets MPR based on:
   - Past inflation data (lagged)
   - Forecasts and expectations
   - External factors (global rates, oil prices)

2. **Within-month independence**: MPR is **not affected contemporaneously** by same-month realizations of exchange rate, money supply, or inflation.

3. **Empirical precedent**:
   - Sims (1980): Policy rate typically ordered first in closed economies
   - Cushman & Zha (1997): Fed funds rate contemporaneously exogenous

**Assumption**:
```
MPR_t = f(Ω_{t-1})   [Ω_{t-1} = past information set]
```
MPR does **not respond** to contemporaneous shocks in ER, M2, or Inflation.

---

#### **Position 2: Exchange Rate (Immediate Response to MPR)**

**Why second?**

1. **Uncovered Interest Parity (UIP)**:
   ```
   E_t[Δs_{t+1}] = i_t^{NGN} - i_t^{USD}
   ```
   - Exchange rate adjusts **immediately** to interest rate differentials
   - Capital flows respond within days to MPR changes

2. **High-frequency evidence**: Daily exchange rate data shows FX markets react within 24-48 hours to CBN rate decisions.

3. **Contemporaneous causality**:
   - ↑ MPR → ↑ capital inflows → ₦ appreciation (lower NGN/USD)
   - Transmission via **financial markets** (fast)

4. **Literature support**:
   - Kim & Roubini (2000): Exchange rate ordered second in open economy VARs
   - Cushman & Zha (1997): FX responds contemporaneously to policy shocks

**Assumption**:
```
ExchangeRate_t = g(MPR_t, Ω_{t-1})
```
Exchange rate responds to **contemporaneous MPR** but not to M2 or Inflation within same month.

**Why not contemporaneous feedback to MPR?**
- CBN does not adjust MPR within-month in response to FX movements
- Policy rate is "stickier" (changed only at MPC meetings)

---

#### **Position 3: Money Supply M2 (Lagged Banking Channel)**

**Why third?**

1. **Banking system lags**:
   - MPR affects **reserve requirements** and **discount window lending**
   - Banks adjust lending rates with ~2-4 week lag
   - Credit creation (M2 growth) materializes after 1+ months

2. **Exchange rate effect on M2**:
   - ₦ depreciation → ↑ M2 (valuation effect on FX deposits converted to Naira)
   - FX inflows enter banking system → ↑ reserves → ↑ M2 multiplier

3. **Data construction**:
   - M2 is **end-of-month stock** (not flow)
   - Captures accumulated effects of MPR and ER movements over the month

4. **Empirical timing**:
   - Credit impulse response functions typically show **1-2 quarter lags** (Bernanke & Blinder, 1992)

**Assumption**:
```
M2_t = h(MPR_t, ER_t, Ω_{t-1})
```
M2 responds to **contemporaneous MPR and ER** but not to contemporaneous inflation.

**Why not contemporaneous Inflation → M2?**
- Inflation affects **real** money demand, but nominal M2 adjustment takes time
- Price-level changes do not instantaneously alter banking system liquidity

---

#### **Position 4: Inflation (Most Endogenous)**

**Why last?**

1. **Price stickiness** (Calvo, 1983):
   - Firms do not adjust prices continuously
   - Menu costs, contracts, and information lags delay pass-through

2. **Transmission lag evidence**:
   - **Exchange rate pass-through**: 3-6 months for imported goods
   - **Monetary policy lag**: 6-12 months (Friedman: "long and variable lags")
   - Nigerian studies: CBN (2018) finds 9-month peak effect of MPR on CPI

3. **Measurement timing**:
   - CPI is measured mid-month or averaged over month
   - Contemporaneous MPR change (announced mid-month) affects **next month's** inflation

4. **Channel accumulation**:
   - Inflation responds to **all three channels**:
     - Direct: MPR → interest rates → aggregate demand → inflation
     - Exchange rate: ER → import prices → CPI
     - Money supply: M2 → liquidity → demand-pull inflation

**Assumption**:
```
Inflation_t = k(MPR_t, ER_t, M2_t, Ω_{t-1})
```
Inflation responds to all contemporaneous variables (with understanding that most actual effects are lagged).

**Why place last?**
- **Most endogenous**: Inflation is determined by all other variables + external shocks (oil, food)
- Ordering last maximizes explained variance by structural shocks (conservative identification)

---

## 3. Alternative Orderings Considered

### 3.1 Inflation Before M2?

**Rejected** because:
- Inflation does not affect banking system liquidity **contemporaneously**
- Real money demand adjustments take time
- Conflicts with price stickiness evidence

### 3.2 Exchange Rate Before MPR?

**Rejected** because:
- Implies CBN adjusts MPR **within month** in response to FX moves
- Not supported by MPC meeting schedule (bi-monthly)
- Would contradict policy exogeneity assumption

### 3.3 Non-Recursive Identification?

**Alternative**: Sign restrictions or external instruments

**Why Cholesky chosen**:
- Transparent, replicable
- Well-established in literature
- Suitable for small open economy context
- Robust across studies (Cushman & Zha, 1997; Kim & Roubini, 2000)

---

## 4. Robustness Checks

### 4.1 Sensitivity Analysis

Test alternative orderings:
1. **M2 before ER**: Check if IRFs change significantly
2. **Inflation before M2**: Verify qualitative results robust

### 4.2 Generalized Impulse Responses

Compute **Pesaran-Shin (1998)** ordering-invariant IRFs:
- If results similar to Cholesky → identification robust
- If different → investigate further

### 4.3 Granger Causality Tests

Pre-test:
- MPR should **not be Granger-caused** by ER, M2, Inflation (supports exogeneity)
- ER, M2 **should be caused** by MPR (supports ordering)

---

## 5. Nigerian Economic Context

### 5.1 Institutional Features Supporting Ordering

1. **CBN independence** (CBN Act 2007):
   - MPC autonomous in setting MPR
   - Not required to respond to short-term FX fluctuations

2. **NAFEX market structure**:
   - Freely floating (post-2016)
   - Responds rapidly to capital flow changes

3. **Banking system structure**:
   - Reserve requirements enforced
   - Discount window operations tied to MPR

### 5.2 Structural Breaks Consideration

**Major episodes**:
- **2016**: Naira devaluation (₦197 → ₦305)
- **2020**: COVID-19 shock
- **2023**: FX market unification (₦460 → ₦750)

**Implication**: Sub-sample stability tests needed. Ordering likely robust, but coefficient magnitudes may change.

---

## 6. Comparison to International Literature

### 6.1 Developed Economies

**Christiano, Eichenbaum, Evans (1999)** - US:
```
Fed Funds Rate → Reserves → M1 → GDP → GDP Deflator
```
- Policy rate first (exogenous)
- Prices last (sticky)

### 6.2 Emerging Markets / Small Open Economies

**Kim & Roubini (2000)** - Non-US G7:
```
Foreign Interest Rate → Domestic Rate → Exchange Rate → Money → Output → Prices
```
- Exchange rate after policy rates
- Prices last

**Afrin (2017)** - Bangladesh:
```
Policy Rate → Exchange Rate → Money Supply → Inflation
```
- **Identical to our ordering** (validates choice)

---

## 7. Summary: Ordering Decision Matrix

| Variable | Position | Contemporaneous Determinants | Justification |
|----------|----------|------------------------------|---------------|
| **MPR** | 1 | Ω_{t-1} (lagged info only) | Policy exogeneity; MPC meeting schedule |
| **Exchange Rate** | 2 | MPR_t, Ω_{t-1} | UIP; financial market speed |
| **M2** | 3 | MPR_t, ER_t, Ω_{t-1} | Banking lags; reserve effects |
| **Inflation** | 4 | MPR_t, ER_t, M2_t, Ω_{t-1} | Price stickiness; transmission lags |

---

## 8. Implications for Interpretation

### 8.1 Impulse Response Functions

**MPR shock → Inflation response**:
- Captures **total effect** through all channels:
  - Direct (interest rate → demand)
  - Indirect (ER channel, M2 channel)

**Interpretation**: Reduced-form policy effectiveness (what CBN actually observes).

### 8.2 Forecast Error Variance Decomposition

**Inflation FEVD**:
- Attributes variance to **fundamental shocks**:
  - Policy shock (MPR innovation)
  - Financial shock (ER innovation)
  - Liquidity shock (M2 innovation)
  - Demand/supply shock (Inflation innovation)

**Caution**: Ordering affects attribution. Earlier variables capture more variance (by construction).

---

## 9. Thesis Defense Preparation

### Expected Questions & Answers

**Q1: "Why not put inflation before M2?"**
- A: Price stickiness literature (Calvo, 1983); inflation does not contemporaneously affect banking system liquidity. M2 is a stock variable measured end-of-month, capturing accumulated MPR/ER effects.

**Q2: "How do you know MPR is exogenous?"**
- A: (1) MPC meets bi-monthly, cannot respond within-month; (2) Granger causality tests; (3) Established in literature (Sims, 1980).

**Q3: "What if CBN does respond to FX within-month?"**
- A: Robustness check with alternative ordering; also, CBN historically uses FX interventions (not MPR) for within-month FX management.

**Q4: "Why Cholesky instead of sign restrictions?"**
- A: Transparency, replicability, alignment with literature. Sign restrictions require subjective prior assumptions on response directions.

---

## References

1. Christiano, L. J., Eichenbaum, M., & Evans, C. L. (1999). Monetary policy shocks: What have we learned and to what end? *Handbook of Macroeconomics*, 1, 65-148.

2. Kim, S., & Roubini, N. (2000). Exchange rate anomalies in the industrial countries: A solution with a structural VAR approach. *Journal of Monetary Economics*, 45(3), 561-586.

3. Cushman, D. O., & Zha, T. (1997). Identifying monetary policy in a small open economy under flexible exchange rates. *Journal of Monetary Economics*, 39(3), 433-448.

4. Sims, C. A. (1980). Macroeconomics and reality. *Econometrica*, 1-48.

5. Bernanke, B. S., & Blinder, A. S. (1992). The federal funds rate and the channels of monetary transmission. *American Economic Review*, 82(4), 901-921.

6. Calvo, G. A. (1983). Staggered prices in a utility-maximizing framework. *Journal of Monetary Economics*, 12(3), 383-398.

7. Pesaran, H. H., & Shin, Y. (1998). Generalized impulse response analysis in linear multivariate models. *Economics Letters*, 58(1), 17-29.

8. Afrin, S. (2017). Monetary policy transmission in Bangladesh: Exploring the lending channel. *Journal of Asian Economics*, 49, 60-80.

9. Central Bank of Nigeria (2018). Monetary Policy Transmission Mechanism in Nigeria. *CBN Working Paper Series*.

---

**Document Version**: 1.0
**Date**: 2025-02 (Day 1)
**Status**: Thesis-Ready
