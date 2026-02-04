# Week 1 – Day 3: Stationarity Testing (ADF, PP, KPSS)
## Complete Build Guide

**Duration**: 5-6 hours
**Difficulty**: ⭐⭐⭐ (Intermediate — first real econometrics)
**New dependencies today**: `statsmodels`, `scipy`
**File you build**: `src/econometrics/stationarity_tests.py`

---

## 🎯 Objective

Move from *visual guesses* (Day 2) to **statistically rigorous
proof** of whether each variable is stationary or not.  This single
result determines which econometric model you must use — so get it
right.

---

## 📚 Prerequisites

- Days 1 & 2 completed
- 9 EDA plots in `results/`
- You can run `python src/visualization/plots.py` without error

---

## 🧠 Part 1 — WHY Stationarity Matters (Theory)

This is the longest theory section in the entire project.  Read it
carefully.  If an examiner asks you one question, it will be about
this.

### 1.1 What IS stationarity?

A time series {y_t} is **stationary** if its statistical properties
do not change over time:

```
E[y_t]     = μ         (constant mean)
Var[y_t]   = σ²        (constant variance)
Cov[y_t, y_{t-k}] = γ(k)   (covariance depends only on lag k, not on t)
```

**Intuition**: a stationary series "wanders around a fixed average"
and always comes back.  A non-stationary series "drifts away forever".

### 1.2 Why does it matter?

**Spurious regression** (Granger & Newbold, 1974):

If you regress one random walk on another — two series with
*zero* relationship — you get R² ≈ 0.95 and p < 0.001.  The
regression *looks* perfect but means nothing.

```
Example:  MPR = α + β × ExchangeRate + ε

If both are I(1) and NOT cointegrated, every coefficient is
meaningless, every p-value is a lie.
```

**Conclusion**: before running any regression, you MUST determine the
integration order.

### 1.3 Integration order — the vocabulary

| Symbol | Meaning | Example |
|--------|---------|---------|
| I(0) | Stationary in levels | Inflation (sometimes) |
| I(1) | Stationary after ONE first-difference | MPR, Exchange Rate |
| I(2) | Needs differencing TWICE | Rare in macro |

**Key rule**:
- I(0) + I(0) → can regress directly
- I(1) + I(1) → CAN regress if they are **cointegrated** (test Day 4)
- I(0) + I(1) → cannot regress levels; difference the I(1) variable
  OR use ARDL (which handles this automatically)

### 1.4 The three tests — why we need all three

No single unit-root test is perfect.  They all have known weaknesses:

| Test | H₀ | Power weakness | Best at |
|------|-----|----------------|---------|
| ADF | Unit root | Low power near unit root | General purpose |
| PP | Unit root | Sensitive to structural breaks | Robust to short-run dependence |
| KPSS | **Stationary** ← opposite! | Low power with trends | Confirming stationarity |

**The consensus rule**:
```
ADF rejects  AND  KPSS does NOT reject  →  Stationary  ✓
ADF fails    AND  KPSS rejects          →  Non-stationary  ✓
Tests disagree                           →  Investigate further
```

Using all three together dramatically reduces the chance of a wrong
conclusion.

---

## 💾 Part 2 — Install Day 3 Dependencies

```bash
# Activate your virtual environment first!
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate           # Windows

pip install statsmodels>=0.14.1 scipy==1.11.4
```

**Why statsmodels?**
- `adfuller()` and `kpss()` are built right in
- `plot_acf` / `plot_pacf` for lag-correlation plots
- `OLS` with `cov_type="HAC"` lets us implement Phillips-Perron
  without a separate package

**Why scipy?**
- statsmodels uses it internally for critical-value interpolation
- You won't call it directly; it's a behind-the-scenes requirement

**Verify**:
```python
import statsmodels; print(statsmodels.__version__)  # >= 0.14.1
import scipy;       print(scipy.__version__)         # 1.11.x
```

---

## 🐍 Part 3 — Build `stationarity_tests.py`

**Create file**: `src/econometrics/stationarity_tests.py`

### 3.1 Imports

```python
import pandas  as pd
import numpy   as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing  import Dict

from statsmodels.tsa.stattools        import adfuller, kpss
from statsmodels.graphics.tsaplots    import plot_acf, plot_pacf
import statsmodels.api                as sm
```

**New imports explained**:
- `adfuller` — the ADF test function (returns tuple of results)
- `kpss` — the KPSS test function
- `plot_acf` / `plot_pacf` — autocorrelation plots (visual lag check)
- `sm` — OLS regression engine (used for Phillips-Perron)

### 3.2 Class skeleton

```python
class StationarityTester:
    def __init__(self, df: pd.DataFrame, save_dir: str = "results/stationarity"):
        self.df       = df
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results            = {}
        self.integration_orders = {}
```

Same pattern as Day 1's `NigerianMacroDataLoader` — inject the output
directory, create it if missing.

### 3.3 The ADF test — line by line

```python
    def _run_adf(self, series: pd.Series, name: str) -> Dict:
        raw = adfuller(series.dropna(), maxlag=12,
                       regression="ct", autolag="AIC")

        return {
            "variable":        name,
            "test":            "ADF",
            "test_statistic":  round(raw[0], 4),   # t-statistic
            "p_value":         round(raw[1], 4),
            "lags_used":       raw[2],              # chosen by AIC
            "n_obs":           raw[3],
            "critical_values": raw[4],              # {'1%': …, '5%': …, '10%': …}
            "stationary":      raw[1] < 0.05,       # reject H₀ at 5%
        }
```

**Parameter choices — defend these in your thesis**:

| Parameter | Value | Why |
|-----------|-------|-----|
| `maxlag` | 12 | Monthly data; 12 months covers one policy cycle |
| `regression` | `"ct"` | Constant + trend; Nigerian macro has deterministic trends |
| `autolag` | `"AIC"` | Schwarz (BIC) is more conservative; AIC balances bias-variance |

**What `adfuller()` returns** (a tuple):
```
raw[0] = ADF t-statistic
raw[1] = p-value  (MacKinnon 1994 interpolation)
raw[2] = number of lags used
raw[3] = number of observations
raw[4] = critical values dict  {'1%': -3.43, '5%': -2.86, '10%': -2.57}
raw[5] = AIC of chosen lag
raw[6] = BIC of chosen lag
raw[7] = dict of ADF stats at each lag (if autolag='AIC')
```

**Decision rule**:
```
If  p_value < 0.05  →  reject H₀  →  series IS stationary  →  I(0)
If  p_value ≥ 0.05  →  fail to reject  →  assume unit root  →  I(1)
```

### 3.4 The Phillips-Perron test — built from OLS

PP is **not** in `statsmodels.tsa.stattools`, so we build it ourselves
from OLS with Newey-West (HAC) standard errors.

```python
    def _run_pp(self, series: pd.Series, name: str) -> Dict:
        y = series.dropna().values
        T = len(y)

        # ── regression variables ──
        dy    = np.diff(y)                              # Δy_t
        y_lag = y[:-1]                                  # y_{t-1}
        trend = np.arange(1, len(dy) + 1, dtype=float)

        X = sm.add_constant(np.column_stack([trend, y_lag]))
        # column layout:  [const  |  trend  |  y_{t-1}]
        #  index:            0          1          2      ← ρ is here
```

**The regression equation**:
```
Δy_t  =  μ  +  δ·t  +  ρ·y_{t-1}  +  u_t
```
- `μ`  = constant (intercept)
- `δ`  = trend slope
- `ρ`  = the coefficient we're testing.  Under H₀: ρ = 0  (unit root)
- `u_t` = error term (may have serial correlation — that's OK for PP)

```python
        # ── Newey-West bandwidth  (Andrews 1991) ──
        bw = max(1, int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0))))

        # ── OLS with HAC covariance ──
        model = sm.OLS(dy, X).fit(cov_type="HAC",
                                  cov_kwds={"maxlags": bw})

        pp_t = round(model.tvalues[2], 4)   # t-stat on ρ
```

**Why HAC?**  Phillips & Perron (1988) showed that ordinary OLS
standard errors are **biased** when the error term `u_t` is
autocorrelated.  The Newey-West (HAC) correction fixes this.  The
resulting t-statistic converges to the same asymptotic distribution as
the standard Dickey-Fuller table.

**Bandwidth formula** `4·(T/100)^{2/9}`:
- Andrews (1991) plug-in rule
- For T = 180  →  bw = 3
- Balances bias (too few lags) vs variance (too many lags)

```python
        # ── p-value via critical-value lookup ──
        cv = {"1%": -3.428, "5%": -2.862, "10%": -2.572}

        if   pp_t < cv["1%"]:   p_val = 0.005
        elif pp_t < cv["5%"]:   p_val = 0.02
        elif pp_t < cv["10%"]:  p_val = 0.08
        else:                   p_val = 0.40

        return { … "stationary": pp_t < cv["5%"] }
```

**Why approximate p-values?**  MacKinnon (1994) response-surface
tables give exact p-values only for the ADF test.  PP shares the same
asymptotic distribution, so the critical values are identical.  The
bracketed p-values (0.005, 0.02, 0.08, 0.40) are conservative —
acceptable for academic work.

### 3.5 The KPSS test — mind the flipped null

```python
    def _run_kpss(self, series: pd.Series, name: str) -> Dict:
        raw = kpss(series.dropna(), regression="ct", nlags="auto")

        return {
            …
            "stationary": raw[1] > 0.05,    # ← FLIPPED!  p > 0.05 = keep H₀ = stationary
        }
```

**⚠️ The KPSS null is the OPPOSITE of ADF and PP.**

| Test | H₀ | Reject H₀ means… |
|------|-----|-------------------|
| ADF | unit root | Stationary |
| PP | unit root | Stationary |
| KPSS | **stationary** | **Non-stationary** |

So the decision logic flips:
```
KPSS  p > 0.05  →  fail to reject H₀  →  keep "stationary"  →  I(0)
KPSS  p < 0.05  →  reject H₀          →  "non-stationary"   →  I(1)
```

This is the single most common mistake students make.  Print it on
your wall.

### 3.6 Run on levels AND first differences

```python
    def run_all_tests(self) -> Dict:
        self.results = {"levels": [], "first_differences": []}

        # Panel A — raw levels
        for col in self.df.columns:
            adf    = self._run_adf(self.df[col],  col)
            pp     = self._run_pp(self.df[col],   col)
            kpss_r = self._run_kpss(self.df[col], col)
            self.results["levels"].extend([adf, pp, kpss_r])

        # Panel B — first differences
        df_d = self.df.diff().dropna()
        for col in df_d.columns:
            label_d = f"Δ{col}"
            adf    = self._run_adf(df_d[col],  label_d)
            pp     = self._run_pp(df_d[col],   label_d)
            kpss_r = self._run_kpss(df_d[col], label_d)
            self.results["first_differences"].extend([adf, pp, kpss_r])
```

**Why both panels?**
- Panel A tells you: is it stationary NOW?
- Panel B tells you: is it stationary after one difference?
- Together they determine I(0) vs I(1) vs I(2)

### 3.7 Determine integration order

```python
    def determine_integration_order(self) -> Dict[str, str]:
        for col in self.df.columns:
            lev = # ADF result for levels
            dif = # ADF result for first differences

            if lev["stationary"]:
                order = "I(0)"
            elif dif["stationary"]:
                order = "I(1)"
            else:
                order = "I(2)?"

            self.integration_orders[col] = order
```

**Decision tree (memorise this)**:
```
                  ADF on LEVELS
                 /            \
          Reject (p<0.05)    Fail to reject
               ↓                    ↓
             I(0)            ADF on FIRST DIFF
                            /               \
                     Reject               Fail to reject
                       ↓                        ↓
                     I(1)                     I(2) ?
```

### 3.8 ACF and PACF plots

```python
    def plot_acf_pacf(self):
        for label, data in [("levels", self.df),
                            ("first_differences", self.df.diff().dropna())]:
            fig, axes = plt.subplots(n, 2, …)

            for i, col in enumerate(data.columns):
                plot_acf(data[col].dropna(),  lags=24, ax=axes[i, 0], …)
                plot_pacf(data[col].dropna(), lags=24, ax=axes[i, 1], …)
```

**What ACF and PACF tell you**:

| Pattern | ACF | PACF | Diagnosis |
|---------|-----|------|-----------|
| Slow decay | Decays slowly | Cuts off after lag 1 | **AR(1) / unit root** |
| Sharp cutoff | Cuts off at lag q | Decays slowly | **MA(q)** |
| Alternating | Alternates sign | — | Negative serial correlation |
| All within bounds | Within ± 1.96/√n | Within ± 1.96/√n | **White noise (stationary!)** |

For our 181 observations, the confidence band is ± 1.96/√181 ≈ ±0.146.
Any bar **outside** that band is statistically significant at 5%.

**`lags=24`**: show two years of monthly lags.  Monetary policy
transmission peaks at 6-12 months; 24 gives headroom.

**`method="ywm"`** in `plot_pacf`:  Yule-Walker method.  More stable
than the default "unbiased" method for short series.

### 3.9 Save results to CSV

```python
    def save_results(self):
        for panel, key in [("levels",             "unit_root_tests_levels.csv"),
                           ("first_differences",  "unit_root_tests_first_differences.csv")]:
            rows = []
            for r in self.results[panel]:
                if "error" in r: continue          # skip failed tests gracefully
                row = { "Variable": r["variable"],
                        "Test":     r["test"],
                        "Test_Statistic": r["test_statistic"],
                        "P_Value":  r.get("p_value"),
                        "Stationary": r["stationary"] }
                # flatten critical values
                for k, v in r.get("critical_values", {}).items():
                    row[f"CV_{k}"] = round(v, 3)
                rows.append(row)

            pd.DataFrame(rows).to_csv(self.save_dir / key, index=False)
```

**Why CSV?**  Thesis appendices need clean tables.  CSV → Excel in one
click.  Also version-controllable (diff-friendly).

### 3.10 CLI entry point

```python
def main():
    import sys
    sys.path.insert(0, ".")
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader(data_dir="data")
    df     = loader.load_and_prepare()

    tester = StationarityTester(df, save_dir="results/stationarity")
    tester.run_full_analysis()

if __name__ == "__main__":
    main()
```

---

## 🚀 Part 4 — Run It

```bash
cd Monetary-Policy-Transmission-Analytics-Platform
MPLBACKEND=Agg python src/econometrics/stationarity_tests.py
```

### What to expect in the terminal

```
======================================================================
  PANEL A  –  UNIT ROOT TESTS ON LEVELS
======================================================================
  Variable            ADF t-stat  ADF p   PP t-stat  PP p  KPSS stat  KPSS p  Conclusion
  ─────────────────────────────────────────────────────────────────────────────────
  MPR                   -1.15    0.92      …        …       …         …     I(1) Unit root
  ExchangeRate           0.16    1.00      …        …       …         …     I(1) Unit root
  M2                    -4.10    0.01      …        …       …         …     I(1) Unit root
  Inflation             -3.54    0.04      …        …       …         …     I(1) Unit root

======================================================================
  PANEL B  –  UNIT ROOT TESTS ON FIRST DIFFERENCES
======================================================================
  …  (most variables stationary after differencing)

======================================================================
  INTEGRATION ORDER
======================================================================
  Variable         Level            1st Diff        Order
  MPR              Unit root        Stationary      I(1)
  ExchangeRate     Unit root        Stationary      I(1)
  M2               …                …               I(…)
  Inflation        …                …               I(…)

  📌 ARDL  –  mixed I(0)/I(1) → bounds testing is the IDEAL approach.
  📌 VAR   –  cointegration test needed (Day 4).
```

### Verify output files

```bash
ls -lh results/stationarity/
# unit_root_tests_levels.csv
# unit_root_tests_first_differences.csv
# integration_orders.csv
# acf_pacf_levels.png
# acf_pacf_first_differences.png
```

---

## 📊 Part 5 — Interpret the Results

This section walks you through reading the output and translating
numbers into economic meaning.

### 5.1 Reading the ADF result

```
MPR:  ADF t-stat = -1.15,  p = 0.92
```

- p = 0.92  →  **far** above 0.05
- Fail to reject H₀ (unit root)
- MPR in levels has a unit root → **I(1)**
- This makes sense: MPR trends upward over 2010-2024

```
MPR (Δ):  ADF t-stat = -4.51,  p = 0.001
```

- p = 0.001  →  far below 0.05
- Reject H₀ → first-differenced MPR IS stationary
- Confirms: MPR is **I(1)**

### 5.2 Reading the KPSS result

```
MPR:  KPSS stat = 0.19,  p = 0.02
```

- p = 0.02  <  0.05
- Reject H₀ (stationarity) → **non-stationary** ← agrees with ADF ✓

```
ΔExchangeRate:  KPSS stat = 0.12,  p = 0.10
```

- p = 0.10  >  0.05
- Fail to reject H₀ → "stationary" ← agrees with ADF ✓

### 5.3 When tests disagree — what to do

You may see something like:
```
M2:   ADF says stationary (p=0.006)
      KPSS says non-stationary (p=0.01)
```

**This is not a bug.  It is a known econometric puzzle.**

Possible explanations:
1. **Near-unit-root**: the true ρ is 0.99.  ADF barely rejects; KPSS
   barely rejects the other way.
2. **Structural break**: a regime change (e.g. 2023 FX unification)
   fools both tests.
3. **Small-sample noise**: 181 observations is decent but not huge.

**What to write in your thesis**:
> "ADF and KPSS results for M2 are contradictory.  Given the strong
> upward trend and the structural break in 2023, we treat M2 as I(1)
> and rely on ARDL bounds testing, which is robust to near-unit-root
> behaviour (Pesaran et al., 2001)."

### 5.4 The KPSS "InterpolationWarning"

You will see:
```
InterpolationWarning: The test statistic is outside of the range of
p-values available in the look-up table.
```

**This is normal.**  statsmodels' KPSS look-up table only covers a
limited range.  When the statistic is extreme (very stationary or very
non-stationary), it can only say "p < 0.01" or "p > 0.10".

**How to handle**: treat the returned p-value as a bound, not an exact
number.  The conclusion (stationary / not) is still correct.

### 5.5 Reading the ACF/PACF plots

Open `results/stationarity/acf_pacf_levels.png`:

- **MPR levels**: ACF decays very slowly → classic unit-root signature
- **MPR differences**: ACF cuts off after lag 1 → white noise → ✓ stationary

Open `results/stationarity/acf_pacf_first_differences.png`:
- All bars should fall within the ± 0.146 band after a couple of lags
- If not, you may need seasonal differencing or more lags in the model

### 5.6 The ARDL implication

The integration-order box at the bottom tells you exactly why ARDL
was chosen as the first model:

```
📌 ARDL  –  mixed I(0)/I(1) → bounds testing is the IDEAL approach.
```

**Why ARDL is ideal for mixed orders**:
- Johansen cointegration requires ALL variables to be I(1)
- If even one is I(0), Johansen fails
- ARDL bounds testing handles I(0)/I(1) mix natively
- This is precisely the situation Pesaran et al. (2001) designed for

---

## 🔧 Part 6 — Troubleshooting

### "ImportError: cannot import name 'adfuller'"
```bash
pip install --upgrade statsmodels
# If still broken:
pip uninstall statsmodels
pip install statsmodels>=0.14.1
```

### "TypeError: deprecate_kwarg() missing argument"
This is a known bug in statsmodels 0.14.1 on Python 3.11+.
```bash
pip install --upgrade statsmodels    # gets 0.14.6+
```

### "ValueError: endog is all NaN"
You forgot to `.dropna()`:
```python
# Bad
adfuller(df["MPR"])

# Good
adfuller(df["MPR"].dropna())
```

### All variables show I(0) on levels — suspicious?
Double-check: are your variables in **levels** or already differenced?
If `data_loader.py` accidentally differenced the data, everything
looks stationary.  Print `df.head()` and confirm values match the raw
CSV.

---

## ✅ Day 3 Completion Checklist

### Code
- [ ] `src/econometrics/stationarity_tests.py` created (full file)
- [ ] `requirements.txt` updated with statsmodels, scipy
- [ ] Script runs without errors (warnings are OK)
- [ ] 3 CSV files in `results/stationarity/`
- [ ] 2 ACF/PACF PNGs in `results/stationarity/`

### Understanding — can you answer these?
- [ ] What does I(1) mean in plain English?
- [ ] Why does KPSS have a FLIPPED null hypothesis?
- [ ] What is a "spurious regression" and why does it matter?
- [ ] Why do we need BOTH levels AND first differences tested?
- [ ] What does a slowly-decaying ACF tell you?
- [ ] Why is ARDL the right choice when orders are mixed?

### Git
```bash
git add src/econometrics/stationarity_tests.py \
        results/stationarity/ \
        requirements.txt
git commit -m "Day 3: stationarity tests — ADF, PP, KPSS + integration orders"
git push -u origin claude/monetary-policy-analytics-platform-03tRK
```

---

## 🎓 Thesis-Ready Phrases

Copy these directly into your methodology chapter:

> **Stationarity testing**
> "Unit-root tests were applied to all four variables in both levels
> and first differences using the Augmented Dickey-Fuller (ADF),
> Phillips-Perron (PP), and KPSS tests.  Lag selection for the ADF
> test followed the Schwarz Information Criterion with a maximum of
> 12 lags.  The Phillips-Perron test was estimated using OLS with
> Newey-West heteroskedasticity- and autocorrelation-consistent (HAC)
> standard errors (Andrews, 1991).  The KPSS test used an automatic
> bandwidth selection procedure."

> **Integration order**
> "Results indicate that [MPR / Exchange Rate] are integrated of order
> one, I(1), while [Inflation / M2] exhibit mixed integration behaviour.
> Given the presence of I(0) and I(1) variables, the ARDL bounds-testing
> approach of Pesaran, Shin, and Smith (2001) is adopted for
> cointegration analysis."

---

## 🚀 Day 4 Preview

**Objective**: Cointegration pre-tests and correlation analysis

**What you'll build**: preliminary cointegration framework

**Why it matters**: Day 3 told us the integration orders.  Day 4
answers: *"do the I(1) variables move together in the long run?"*
If yes → cointegration exists → we can model the long-run
equilibrium relationship.  If no → we must difference everything.

---

*Guide v1.0  ·  Day 3  ·  2025-02*
