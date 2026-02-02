# Econometric Methodology

## Overview

This document provides the theoretical foundation for the econometric methods employed in analyzing monetary policy transmission in Nigeria.

---

## 1. Unit Root and Stationarity Testing

### 1.1 Why Test for Stationarity?

Non-stationary variables can lead to **spurious regressions** (Granger & Newbold, 1974). Time series econometrics requires:
- Stationary variables for standard inference
- Knowledge of integration order for cointegration testing
- Proper model specification (levels vs. differences)

### 1.2 Tests Employed

#### Augmented Dickey-Fuller (ADF) Test

**Null Hypothesis**: Variable has a unit root (non-stationary)

Test equation:
```
Δy_t = α + βt + γy_{t-1} + Σδ_i Δy_{t-i} + ε_t
```

- **Test statistic**: t-statistic on γ
- **Rejection**: If t-stat < critical value → stationary
- **Lag selection**: Schwarz Information Criterion (SIC)

#### Phillips-Perron (PP) Test

Non-parametric correction for serial correlation and heteroskedasticity.
- More robust to heteroskedasticity than ADF
- Useful cross-validation with ADF

#### KPSS Test

**Null Hypothesis**: Variable is stationary (reverse of ADF/PP)

- Provides confirmation when ADF/PP reject unit root
- **LM statistic**: Tests stationarity around deterministic trend

**Decision Rule**:
- If ADF rejects unit root AND KPSS fails to reject stationarity → **I(0)**
- If ADF fails to reject AND KPSS rejects stationarity → **I(1)**
- Conflicting results → investigate structural breaks

---

## 2. Autoregressive Distributed Lag (ARDL) Model

### 2.1 Specification

The ARDL(p, q₁, q₂, ..., qₖ) model for inflation:

```
Inflation_t = α₀ + Σ(i=1 to p) α_i Inflation_{t-i}
            + Σ(j=0 to q₁) β₁ⱼ MPR_{t-j}
            + Σ(j=0 to q₂) β₂ⱼ ExchangeRate_{t-j}
            + Σ(j=0 to q₃) β₃ⱼ M2_{t-j}
            + ε_t
```

Where:
- `p` = lags of dependent variable (inflation)
- `qᵢ` = lags of independent variables

### 2.2 Bounds Testing Approach (Pesaran et al., 2001)

**Advantages**:
- Works with mixed I(0)/I(1) variables
- No pre-testing for unit roots required
- Estimates long-run and short-run relationships simultaneously

**Test Procedure**:

1. **Estimate unrestricted ECM**:
   ```
   ΔInflation_t = α₀ + Σφ_i ΔInflation_{t-i} + Σθ_i ΔMPR_{t-i}
                + λ₁ Inflation_{t-1} + λ₂ MPR_{t-1} + ... + ε_t
   ```

2. **F-statistic test**:
   - H₀: λ₁ = λ₂ = ... = 0 (no cointegration)
   - Compare F-stat to critical bounds [I(0), I(1)]
   - If F > upper bound → **cointegration exists**

3. **Estimate long-run coefficients**:
   ```
   Long-run elasticity = -λ_j / λ₁
   ```

4. **Error Correction Model (ECM)**:
   ```
   ΔInflation_t = short-run dynamics + γ·ECM_{t-1} + ε_t
   ```
   - **ECM coefficient (γ)**: Speed of adjustment to long-run equilibrium
   - Expected: -1 < γ < 0 (convergence)

### 2.3 Lag Selection

- **Information Criteria**: Akaike (AIC), Schwarz (SIC)
- **Maximum lags**: min(12, T/5) for monthly data
- **Sequential testing**: Start with max lags, test down

---

## 3. Vector Autoregression (VAR)

### 3.1 VAR Model Structure

A VAR(p) system for our 4 variables:

```
[MPR_t        ]   [c₁]   [A₁₁ A₁₂ A₁₃ A₁₄] [MPR_{t-1}    ]       [ε₁t]
[ER_t         ] = [c₂] + [A₂₁ A₂₂ A₂₃ A₂₄] [ER_{t-1}     ] + ... [ε₂t]
[M2_t         ]   [c₃]   [A₃₁ A₃₂ A₃₃ A₃₄] [M2_{t-1}     ]       [ε₃t]
[Inflation_t  ]   [c₄]   [A₄₁ A₄₂ A₄₃ A₄₄] [Inflation_{t-1}]     [ε₄t]
```

Where:
- `Aᵢⱼ` = coefficient matrices at each lag
- `εᵢt` = reduced-form residuals
- `p` = optimal lag length (selected via AIC/BIC)

### 3.2 Identification: Cholesky Decomposition

**Problem**: Reduced-form VAR residuals are correlated. Need to recover **structural shocks**.

**Solution**: Recursive identification via Cholesky ordering.

**Ordering** (contemporaneous effects):
```
MPR → Exchange Rate → M2 → Inflation
```

**Economic Rationale**:

1. **MPR (Block exogenous)**:
   - CBN sets MPR independently based on past data
   - Not affected contemporaneously by other variables within month

2. **Exchange Rate**:
   - Responds immediately to MPR via uncovered interest parity
   - Capital flows react to interest rate differentials

3. **Money Supply (M2)**:
   - Adjusts with ~1 month lag via banking system credit
   - Depends on MPR (reserve requirements, lending) and ER (FX transactions)

4. **Inflation (Block endogenous)**:
   - Slowest to respond (price stickiness)
   - Pass-through from ER and M2 takes 3-12 months
   - Affected by all shocks with lag

**Technical Implementation**:
- Decompose covariance matrix: Σ = PP'
- Structural shocks: u_t = P⁻¹ε_t
- Lower-triangular P ensures recursive structure

### 3.3 Lag Length Selection

**Criteria**:
- **Akaike Information Criterion (AIC)**: -2log(L) + 2k
- **Schwarz Information Criterion (SIC)**: -2log(L) + k·log(T)
- **Hannan-Quinn (HQ)**: -2log(L) + 2k·log(log(T))

**Selection Strategy**:
- Test lags 1-12 (monthly data)
- Choose lag where most criteria agree
- Verify residual diagnostics (no autocorrelation)

### 3.4 Diagnostics

1. **Stability**:
   - Check eigenvalues of companion matrix
   - All roots < 1 in modulus (inside unit circle)

2. **Residual Tests**:
   - **Portmanteau test**: No autocorrelation up to lag h
   - **Normality**: Jarque-Bera test
   - **Heteroskedasticity**: White test

---

## 4. Impulse Response Functions (IRF)

### 4.1 Definition

IRF traces the effect of a **one-standard-deviation shock** to variable j on variable i over time:

```
IRF(i,j,h) = ∂y_{i,t+h} / ∂ε_{j,t}
```

Where:
- h = horizon (0, 1, 2, ..., 24 months)
- ε_{j,t} = structural shock to variable j at time t

### 4.2 Calculation

From VAR(p) model:
```
y_t = c + A₁y_{t-1} + ... + Aₚy_{t-p} + ε_t
```

Compute MA(∞) representation:
```
y_t = μ + Σ(h=0 to ∞) Ψₕε_{t-h}
```

IRF coefficients: **Ψₕ** matrices

### 4.3 Confidence Intervals

**Bootstrap Method** (500 replications):
1. Estimate VAR, obtain residuals
2. Resample residuals with replacement
3. Generate bootstrap IRFs
4. Compute 90% or 95% percentile bands

### 4.4 Interpretation for Nigeria

**Key IRF to analyze**: MPR shock → Inflation response

Expected pattern:
- **Month 0-3**: Minimal effect (pass-through lag)
- **Month 3-9**: Negative response peaks (tightening reduces inflation)
- **Month 9-24**: Gradual return to baseline

---

## 5. Forecast Error Variance Decomposition (FEVD)

### 5.1 Definition

FEVD answers: *"What percentage of h-step-ahead forecast error variance in variable i is due to shocks in variable j?"*

```
FEVD(i,j,h) = Σ(s=0 to h-1) (e'ᵢΨₛP₀ⱼ)² / MSE_i(h)
```

Where:
- MSE_i(h) = h-step-ahead mean squared error for variable i
- P₀ⱼ = column j of Cholesky factor (structural shock j)

### 5.2 Interpretation

For **Inflation** at horizon h=12 months:

| Shock from | FEVD % | Interpretation |
|------------|--------|----------------|
| MPR | 30% | Direct policy effect |
| Exchange Rate | 40% | Pass-through channel |
| M2 | 15% | Money supply channel |
| Own shock | 15% | Other factors |

**Policy Insight**: If exchange rate dominates, CBN should focus on FX stability.

---

## 6. Policy Simulations

### 6.1 Scenario Design

**Baseline**: +100 basis points (1%) MPR increase

**Simulation Steps**:
1. Estimate VAR on historical data
2. Set initial conditions (last observed values)
3. Apply exogenous shock: MPR_t = MPR_{t-1} + 1%
4. Forecast forward 24 months
5. Compare to counterfactual (no shock)

### 6.2 Output

**Inflation differential path**:
```
Δ Inflation_t = Inflation^{shocked}_t - Inflation^{baseline}_t
```

**Expected result**: Negative for first 12 months (policy works), then converges to zero.

---

## 7. Model Validation

### 7.1 In-Sample Fit
- **R²** for each equation
- **Information criteria** (lower is better)

### 7.2 Out-of-Sample Forecasting
- Reserve last 12 months
- Estimate VAR on training sample (2010-2024)
- Forecast 2024-2025
- Compute RMSE vs. actual

### 7.3 Robustness Checks
1. **Alternative orderings**: Test sensitivity to Cholesky ordering
2. **Sub-sample stability**: Split at structural break (e.g., 2016)
3. **Bootstrap inference**: Verify confidence intervals

---

## 8. Software Implementation

### Python Libraries
```python
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.ardl import ARDL, ardl_select_order
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
```

### Key Functions
- `adfuller()`: ADF test
- `ARDL()`: Estimate ARDL model
- `VAR()`: Estimate VAR
- `.irf()`: Compute impulse responses
- `.fevd()`: Forecast error variance decomposition

---

## References

1. Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. *Journal of the American Statistical Association*, 74(366a), 427-431.

2. Pesaran, M. H., Shin, Y., & Smith, R. J. (2001). Bounds testing approaches to the analysis of level relationships. *Journal of Applied Econometrics*, 16(3), 289-326.

3. Sims, C. A. (1980). Macroeconomics and reality. *Econometrica*, 1-48.

4. Lütkepohl, H. (2005). *New introduction to multiple time series analysis*. Springer.

5. Hamilton, J. D. (1994). *Time series analysis*. Princeton University Press.

---

**Document Status**: Complete for Week 1
**Last Updated**: 2025-02 (Day 1)
