# Week 1 - Day 3: Stationarity Testing

**Time Estimate:** 8 hours (full day)
**What You'll Build:** `src/econometrics/stationarity_tests.py` (~430 lines)
**End Goal:** Determine whether each variable is I(0) or I(1) — critical for choosing ARDL vs VAR vs VECM

**Pre-requisite:** Day 1 & 2 completed. Install new packages:
```bash
pip install statsmodels==0.14.1 scipy==1.11.4
```

---

## What You're Building Today

By 5 PM today, you will have:

- ✅ `src/econometrics/stationarity_tests.py` (~430 lines)
- ✅ ADF test (Augmented Dickey-Fuller) — standard unit root test
- ✅ PP test (Phillips-Perron) — robust to serial correlation
- ✅ KPSS test — opposite null: tests if series IS stationary
- ✅ All three tests on LEVELS and FIRST DIFFERENCES for every variable
- ✅ Integration order table: `{'MPR': 'I(1)', 'Inflation': 'I(0)', ...}`
- ✅ ACF/PACF plots saved to `results/stationarity/`

**Why three tests?** Each has different assumptions. Agreement across all three gives a confident conclusion.

---

## Hour 1 (9 AM - 10 AM): Understand Stationarity & Module Setup

### Step 1.1: What is stationarity?

**Stationary series:** Constant mean, variance, and autocorrelation over time.
- Example: Inflation around a mean of ~12% roughly stationary = I(0)

**Non-stationary series (unit root):** Mean and variance change with time.
- Example: Exchange Rate keeps trending upward, no mean reversion = I(1)

**Why it matters for your thesis:**
- If MPR and ExchangeRate are both I(1) → need cointegration test (Day 4)
- Mixed I(0) and I(1) → ARDL bounds test is the right approach (Day 5)
- All I(0) → simple OLS is valid

**Three complementary tests:**

| Test | Null Hypothesis | Stationary when |
|------|----------------|-----------------|
| ADF  | Unit root exists (non-stationary) | p < 0.05 |
| PP   | Unit root exists (non-stationary) | p < 0.05 |
| KPSS | Series IS stationary | p > 0.05 (opposite!) |

**Note:** KPSS is the OPPOSITE of ADF/PP. All three agree when:
- ADF: p < 0.05 AND PP: p < 0.05 AND KPSS: p > 0.05

---

### Step 1.2: Create module — imports & colour palette

Create `src/econometrics/stationarity_tests.py` and **write this first chunk:**

```python
"""
Stationarity-Testing Module — Nigerian Monetary Policy Transmission Analysis

Implements three complementary unit-root tests:

    ADF   - Augmented Dickey-Fuller        (statsmodels)
    PP    - Phillips-Perron                (statsmodels OLS + HAC covariance)
    KPSS  - Kwiatkowski-Phillips-Schmidt-Shin  (statsmodels)

Each test is run on LEVELS and on FIRST DIFFERENCES.  The module then
determines the integration order I(0)/I(1) for every variable.

Day 3 deliverable.  New dependencies: statsmodels, scipy.

Academic references
-------------------
Phillips & Perron (1988)  - Econometrica 56(2)
Kwiatkowski et al. (1992) - Journal of Econometrics 54
Hamilton (1994) Ch.15     - Time-Series Analysis (textbook)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm


# ─────────────────────────────────────────────────────────────────────
# COLOUR PALETTE  (kept in sync with plots.py)
# ─────────────────────────────────────────────────────────────────────

_COLORS = {
    "MPR":            "#2E86AB",
    "Inflation":      "#A23B72",
    "ExchangeRate":   "#F18F01",
    "M2":             "#6A994E",
}


class StationarityTester:
    """
    Run ADF / PP / KPSS on every column of df, on both levels and
    first differences.  Persist results as CSV and plots to save_dir.

    Parameters
    ----------
    df       : pd.DataFrame  - columns in VAR order, DatetimeIndex
    save_dir : str | Path    - e.g. 'results/stationarity'
    """

    def __init__(self, df: pd.DataFrame, save_dir: str = "results/stationarity"):
        self.df       = df
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.results            = {}   # filled by run_all_tests()
        self.integration_orders = {}   # filled by determine_integration_order()
        print(f"StationarityTester initialized. Results: {self.save_dir}")


# Test code
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader(data_dir="data")
    df = loader.load_and_prepare()

    tester = StationarityTester(df, save_dir="results/stationarity")
    print("StationarityTester initialized successfully!")
```

**Save and test:**

```bash
python src/econometrics/stationarity_tests.py
```

---

## Hour 2 (10 AM - 11 AM): Implement ADF Test

### Step 2.1: Add the ADF test method

**Add this method to the class** (after `__init__`):

```python
    # ────────────────────────────────────────────────────────────────
    # 1.  ADF  (exact - straight from statsmodels)
    # ────────────────────────────────────────────────────────────────
    def _run_adf(self, series: pd.Series, name: str) -> Dict:
        """
        Augmented Dickey-Fuller test.

        H0 : unit root exists  (non-stationary)
        H1 : no unit root      (stationary)

        Decision : reject H0 means stationary when p < 0.05

        Parameters
        ----------
        regression="ct" : includes constant + trend (appropriate for macro data)
        maxlag=12       : up to 12 lags for monthly data
        autolag="AIC"   : automatically selects optimal lag length
        """
        raw = adfuller(series.dropna(), maxlag=12, regression="ct", autolag="AIC")

        return {
            "variable":        name,
            "test":            "ADF",
            "test_statistic":  round(raw[0], 4),
            "p_value":         round(raw[1], 4),
            "lags_used":       raw[2],
            "n_obs":           raw[3],
            "critical_values": raw[4],   # dict '1%', '5%', '10%'
            "stationary":      raw[1] < 0.05,
        }
```

**Update test code:**

```python
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader(data_dir="data")
    df = loader.load_and_prepare()

    tester = StationarityTester(df, save_dir="results/stationarity")

    print("\nRunning ADF test on MPR...")
    result = tester._run_adf(df['MPR'], 'MPR')
    print(f"  Statistic: {result['test_statistic']}")
    print(f"  p-value:   {result['p_value']}")
    print(f"  Stationary: {result['stationary']}")
```

**Save and test:**

```bash
python src/econometrics/stationarity_tests.py
```

**Expected:** p-value > 0.05 for MPR (it has a unit root = non-stationary = I(1))

---

## Hour 3 (11 AM - 12 PM): Implement PP & KPSS Tests

### Step 3.1: Add the Phillips-Perron test

**Add this method after `_run_adf`:**

```python
    # ────────────────────────────────────────────────────────────────
    # 2.  PHILLIPS-PERRON  (statsmodels OLS + Newey-West / HAC)
    # ────────────────────────────────────────────────────────────────
    def _run_pp(self, series: pd.Series, name: str) -> Dict:
        """
        Phillips-Perron test via OLS with HAC-corrected standard errors.

        The regression  dy_t = mu + delta*t + rho*y_{t-1} + u_t  is estimated
        by OLS with Newey-West (Bartlett) kernel, automatic bandwidth
        following Andrews (1991).

        Under H0 : rho = 0  the t-statistic follows the Dickey-Fuller
        asymptotic distribution — MacKinnon (1994) tables apply.

        H0 : unit root  (non-stationary)
        H1 : no unit root  (stationary)
        """
        y = series.dropna().values
        T = len(y)

        # Regression variables
        dy    = np.diff(y)
        y_lag = y[:-1]
        trend = np.arange(1, len(dy) + 1, dtype=float)

        X = sm.add_constant(np.column_stack([trend, y_lag]))

        # Newey-West bandwidth
        bw = max(1, int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0))))

        # OLS with HAC covariance
        model = sm.OLS(dy, X).fit(cov_type="HAC", cov_kwds={"maxlags": bw})

        pp_t = round(model.tvalues[2], 4)

        # Approximate p-value via MacKinnon critical-value table
        cv = {"1%": -3.428, "5%": -2.862, "10%": -2.572}

        if   pp_t < cv["1%"]:   p_val = 0.005
        elif pp_t < cv["5%"]:   p_val = 0.02
        elif pp_t < cv["10%"]:  p_val = 0.08
        else:                   p_val = 0.40

        return {
            "variable":        name,
            "test":            "PP",
            "test_statistic":  pp_t,
            "p_value":         p_val,
            "critical_values": cv,
            "stationary":      pp_t < cv["5%"],
        }
```

### Step 3.2: Add the KPSS test

**Add this method after `_run_pp`:**

```python
    # ────────────────────────────────────────────────────────────────
    # 3.  KPSS  (note the OPPOSITE null hypothesis!)
    # ────────────────────────────────────────────────────────────────
    def _run_kpss(self, series: pd.Series, name: str) -> Dict:
        """
        KPSS test - OPPOSITE null hypothesis to ADF/PP.

        H0 : series IS stationary
        H1 : unit root exists  (non-stationary)

        Decision : reject H0 means NON-stationary when p < 0.05
        So  stationary = (p > 0.05) -- FLIPPED vs ADF/PP.

        This is why we need all three tests: they complement each other.
        """
        try:
            raw = kpss(series.dropna(), regression="ct", nlags="auto")
            return {
                "variable":        name,
                "test":            "KPSS",
                "test_statistic":  round(raw[0], 4),
                "p_value":         round(raw[1], 4),
                "lags_used":       raw[2],
                "critical_values": raw[3],
                "stationary":      raw[1] > 0.05,   # OPPOSITE to ADF/PP
            }
        except Exception as exc:
            return {"variable": name, "test": "KPSS", "error": str(exc), "stationary": None}
```

**Update test code:**

```python
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader(data_dir="data")
    df = loader.load_and_prepare()

    tester = StationarityTester(df, save_dir="results/stationarity")

    for var in ['MPR', 'Inflation']:
        print(f"\n--- {var} ---")
        adf = tester._run_adf(df[var], var)
        pp  = tester._run_pp(df[var], var)
        kps = tester._run_kpss(df[var], var)
        print(f"  ADF:  stat={adf['test_statistic']:.4f}  p={adf['p_value']:.4f}  stationary={adf['stationary']}")
        print(f"  PP:   stat={pp['test_statistic']:.4f}  p={pp['p_value']:.3f}  stationary={pp['stationary']}")
        print(f"  KPSS: stat={kps['test_statistic']:.4f}  p={kps['p_value']:.4f}  stationary={kps['stationary']}")
```

**Save and test:**

```bash
python src/econometrics/stationarity_tests.py
```

---

## Hour 4 (12 PM - 1 PM): LUNCH BREAK

Take a break! You have all three unit-root tests working.

---

## Hour 5 (1 PM - 2 PM): Run All Tests on Levels + First Differences

### Step 5.1: Add the master test runner

**Add this method after `_run_kpss`:**

```python
    # ────────────────────────────────────────────────────────────────
    # 4.  RUN ALL  (levels + first differences)
    # ────────────────────────────────────────────────────────────────
    def run_all_tests(self) -> Dict:
        """
        Execute the full battery on raw levels AND on first differences.

        Returns a nested dict:
            {
              "levels":            [list of result dicts],
              "first_differences": [list of result dicts],
            }
        """
        self.results = {"levels": [], "first_differences": []}

        # PANEL A - levels
        print("\n" + "=" * 70)
        print("  PANEL A  -  UNIT ROOT TESTS ON LEVELS")
        print("=" * 70)
        print(f"  {'Variable':<18} {'ADF t-stat':>11} {'ADF p':>7}"
              f" {'PP t-stat':>10} {'PP p':>6}"
              f" {'KPSS stat':>10} {'KPSS p':>7}  Conclusion")
        print("  " + "-" * 95)

        for col in self.df.columns:
            adf    = self._run_adf(self.df[col],  col)
            pp     = self._run_pp(self.df[col],   col)
            kpss_r = self._run_kpss(self.df[col], col)
            self.results["levels"].extend([adf, pp, kpss_r])

            # Consensus: stationary only if ALL three agree
            agree_stat = (adf["stationary"] and pp["stationary"]
                          and (kpss_r.get("stationary", False)))
            label = "I(0) Stationary" if agree_stat else "I(1) Unit root"

            kpss_p = kpss_r.get("p_value", "-")
            print(f"  {col:<18}"
                  f" {adf['test_statistic']:>10.4f} {adf['p_value']:>7.4f}"
                  f" {pp['test_statistic']:>10.4f} {pp['p_value']:>6.3f}"
                  f" {kpss_r.get('test_statistic', 0):>10.4f} {kpss_p:>7}"
                  f"  {label}")

        # PANEL B - first differences
        print("\n" + "=" * 70)
        print("  PANEL B  -  UNIT ROOT TESTS ON FIRST DIFFERENCES (Delta y)")
        print("=" * 70)
        print(f"  {'Variable':<18} {'ADF t-stat':>11} {'ADF p':>7}"
              f" {'PP t-stat':>10} {'PP p':>6}"
              f" {'KPSS stat':>10} {'KPSS p':>7}  Conclusion")
        print("  " + "-" * 95)

        df_d = self.df.diff().dropna()

        for col in df_d.columns:
            label_d = f"Delta{col}"
            adf    = self._run_adf(df_d[col],  label_d)
            pp     = self._run_pp(df_d[col],   label_d)
            kpss_r = self._run_kpss(df_d[col], label_d)
            self.results["first_differences"].extend([adf, pp, kpss_r])

            agree_stat = (adf["stationary"] and pp["stationary"]
                          and (kpss_r.get("stationary", False)))
            flag = "Stationary" if agree_stat else "Non-stat"

            kpss_p = kpss_r.get("p_value", "-")
            print(f"  {label_d:<18}"
                  f" {adf['test_statistic']:>10.4f} {adf['p_value']:>7.4f}"
                  f" {pp['test_statistic']:>10.4f} {pp['p_value']:>6.3f}"
                  f" {kpss_r.get('test_statistic', 0):>10.4f} {kpss_p:>7}"
                  f"  {flag}")

        return self.results
```

**Update test code:**

```python
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader(data_dir="data")
    df = loader.load_and_prepare()

    tester = StationarityTester(df, save_dir="results/stationarity")
    results = tester.run_all_tests()
    print("\nAll tests complete!")
```

**Save and test:**

```bash
python src/econometrics/stationarity_tests.py
```

**You should see Panel A (levels) and Panel B (first differences) in the console.**

---

## Hour 6 (2 PM - 3 PM): Determine Integration Orders

### Step 6.1: Add integration order determination

**Add this method after `run_all_tests`:**

```python
    # ────────────────────────────────────────────────────────────────
    # 5.  INTEGRATION-ORDER DETERMINATION
    # ────────────────────────────────────────────────────────────────
    def determine_integration_order(self) -> Dict[str, str]:
        """
        Rule:
            - I(0)  if ADF rejects at 5% on levels
            - I(1)  if ADF does NOT reject on levels  BUT  rejects on Delta y
            - I(2)? if neither level nor Delta y rejects  (flag for review)

        Returns  {'MPR': 'I(1)', 'Inflation': 'I(0)', ...}
        """
        if not self.results:
            raise RuntimeError("Call run_all_tests() first.")

        print("\n" + "=" * 70)
        print("  INTEGRATION ORDER DETERMINATION")
        print("=" * 70)
        print(f"\n  {'Variable':<18} {'Level':<18} {'First Diff':<18} {'Order'}")
        print("  " + "-" * 60)

        for col in self.df.columns:
            # Pull the ADF dicts for this variable
            lev = next((r for r in self.results["levels"]
                        if r.get("variable") == col and r["test"] == "ADF"), None)
            dif = next((r for r in self.results["first_differences"]
                        if r.get("variable") == f"Delta{col}" and r["test"] == "ADF"), None)

            if lev and lev["stationary"]:
                order, lev_txt, dif_txt = "I(0)", "Stationary", "-"
            elif dif and dif["stationary"]:
                order, lev_txt, dif_txt = "I(1)", "Unit root",  "Stationary"
            else:
                order, lev_txt, dif_txt = "I(2)?", "Unit root", "Unit root"

            self.integration_orders[col] = order
            print(f"  {col:<18} {lev_txt:<18} {dif_txt:<18} {order}")

        # ARDL / VAR implications
        i0 = [v for v, o in self.integration_orders.items() if o == "I(0)"]
        i1 = [v for v, o in self.integration_orders.items() if o == "I(1)"]

        print("\n  " + "-" * 60)
        print(f"  I(0) variables : {i0 if i0 else 'none'}")
        print(f"  I(1) variables : {i1 if i1 else 'none'}")
        print()

        if i0 and i1:
            print("  ARDL  - mixed I(0)/I(1)  -> bounds testing is the IDEAL approach.")
            print("  VAR   - cointegration test needed (Day 4) before deciding levels vs Delta.")
        elif not i0:
            print("  ARDL  - all I(1)  -> bounds testing still valid; Johansen also possible.")
            print("  VAR   - test for cointegration; if found, use VECM.")
        else:
            print("  ARDL  - all I(0)  -> simple OLS valid; ARDL can still capture dynamics.")

        return self.integration_orders
```

**Update test code:**

```python
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader(data_dir="data")
    df = loader.load_and_prepare()

    tester = StationarityTester(df, save_dir="results/stationarity")
    tester.run_all_tests()
    orders = tester.determine_integration_order()
    print(f"\nIntegration orders: {orders}")
```

**Save and test:**

```bash
python src/econometrics/stationarity_tests.py
```

---

## Hour 7 (3 PM - 4 PM): Add ACF/PACF Plots & Save CSV Results

### Step 7.1: Add ACF/PACF plot method

**Add this method after `determine_integration_order`:**

```python
    # ────────────────────────────────────────────────────────────────
    # 6.  ACF / PACF PLOTS
    # ────────────────────────────────────────────────────────────────
    def plot_acf_pacf(self):
        """Two figure-grids: one for levels, one for first differences."""
        n = len(self.df.columns)

        for label, data in [("levels", self.df),
                             ("first_differences", self.df.diff().dropna())]:
            fig, axes = plt.subplots(n, 2, figsize=(14, 3.8 * n))
            fig.suptitle(
                f"ACF & PACF - {'Levels' if label == 'levels' else 'First Differences'}",
                fontsize=15, fontweight="bold", y=1.02,
            )

            for i, col in enumerate(data.columns):
                col_name = col if label == "levels" else f"Delta{col}"
                c = _COLORS.get(col, "#333333")

                plot_acf(data[col].dropna(),  lags=24, ax=axes[i, 0],
                         alpha=0.05, color=c)
                axes[i, 0].set_title(f"{col_name} - ACF",  fontsize=11, fontweight="bold")
                axes[i, 0].set_xlabel("Lag (months)")
                axes[i, 0].grid(True, alpha=0.2)

                plot_pacf(data[col].dropna(), lags=24, ax=axes[i, 1],
                          alpha=0.05, color=c, method="ywm")
                axes[i, 1].set_title(f"{col_name} - PACF", fontsize=11, fontweight="bold")
                axes[i, 1].set_xlabel("Lag (months)")
                axes[i, 1].grid(True, alpha=0.2)

            fig.tight_layout()
            path = self.save_dir / f"acf_pacf_{label}.png"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"  saved  {path}")
            plt.close(fig)
```

### Step 7.2: Add save results method

**Add this method after `plot_acf_pacf`:**

```python
    # ────────────────────────────────────────────────────────────────
    # 7.  SAVE RESULTS TO CSV
    # ────────────────────────────────────────────────────────────────
    def save_results(self):
        """Persist every result dict as flat CSV rows."""
        if not self.results:
            print("  Nothing to save - run run_all_tests() first.")
            return

        for panel, key in [("levels", "unit_root_tests_levels.csv"),
                            ("first_differences", "unit_root_tests_first_differences.csv")]:
            rows = []
            for r in self.results[panel]:
                if "error" in r:
                    continue   # skip failed KPSS gracefully
                row = {
                    "Variable":       r["variable"],
                    "Test":           r["test"],
                    "Test_Statistic": r["test_statistic"],
                    "P_Value":        r.get("p_value", "approx"),
                    "Stationary":     r.get("stationary"),
                }
                cv = r.get("critical_values", {})
                if isinstance(cv, dict):
                    for k, v in cv.items():
                        row[f"CV_{k}"] = round(v, 3) if isinstance(v, (int, float)) else v
                rows.append(row)

            if rows:
                path = self.save_dir / key
                pd.DataFrame(rows).to_csv(path, index=False)
                print(f"  saved  {path}")

        # Integration-order summary
        if self.integration_orders:
            path = self.save_dir / "integration_orders.csv"
            pd.DataFrame(
                [{"Variable": v, "Order": o}
                 for v, o in self.integration_orders.items()]
            ).to_csv(path, index=False)
            print(f"  saved  {path}")
```

**Save and test:**

```bash
python src/econometrics/stationarity_tests.py
```

---

## Hour 8 (4 PM - 5 PM): Add Full Pipeline & Commit

### Step 8.1: Add run_full_analysis() and main()

**Add this final section:**

```python
    # ────────────────────────────────────────────────────────────────
    # 8.  FULL PIPELINE
    # ────────────────────────────────────────────────────────────────
    def run_full_analysis(self):
        """One call does everything: test -> order -> plot -> save."""
        print("\n" + "=" * 70)
        print("  STATIONARITY ANALYSIS - FULL PIPELINE")
        print("=" * 70)

        self.run_all_tests()
        self.determine_integration_order()

        print("\n  [Plotting] ACF / PACF ...")
        self.plot_acf_pacf()

        print("\n  [Saving]   CSV results ...")
        self.save_results()

        print("\n" + "=" * 70)
        print("  STATIONARITY ANALYSIS COMPLETE")
        print("=" * 70)

        return self.results, self.integration_orders


# ─────────────────────────────────────────────────────────────────────
# CLI ENTRY-POINT
# ─────────────────────────────────────────────────────────────────────

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

**Save and run full pipeline:**

```bash
python src/econometrics/stationarity_tests.py
```

**Verify outputs:**

```bash
wc -l src/econometrics/stationarity_tests.py
# Should show: ~430 lines

ls results/stationarity/
# unit_root_tests_levels.csv
# unit_root_tests_first_differences.csv
# integration_orders.csv
# acf_pacf_levels.png
# acf_pacf_first_differences.png
```

---

### Step 8.2: Commit everything

```bash
git add .
git status
```

**Commit:**

```bash
git commit -m "Day 3: Stationarity testing (ADF + PP + KPSS)

- src/econometrics/stationarity_tests.py (430 lines)
  - StationarityTester class
  - ADF, Phillips-Perron, KPSS on levels and first differences
  - Integration order determination
  - ACF/PACF plots
  - CSV results

Key results (fill in actual findings):
  - MPR:          I(?)
  - Inflation:    I(?)
  - ExchangeRate: I(?)
  - M2:           I(?)

Outputs: results/stationarity/ (3 CSVs + 2 PNG plots)

https://claude.ai/code/session_019oWdezYCv1NxdFa4QYPPhs"
```

**Push:**

```bash
git push -u origin claude/monetary-policy-analytics-platform-03tRK
```

---

## Final Code Summary

**File**: `src/econometrics/stationarity_tests.py`

```python
# Module-level: _COLORS dict

class StationarityTester:
    def __init__(df, save_dir="results/stationarity")
    def _run_adf(series, name)
    def _run_pp(series, name)
    def _run_kpss(series, name)
    def run_all_tests()
    def determine_integration_order()
    def plot_acf_pacf()
    def save_results()
    def run_full_analysis()

def main()
```

**Verify:**
```bash
python -c "
from src.econometrics.stationarity_tests import StationarityTester
methods = [m for m in dir(StationarityTester) if not m.startswith('__')]
print('Methods:', methods)
"
```

---

## End of Day 3 Checklist

- [ ] `stationarity_tests.py` runs without errors
- [ ] Panel A (levels) and Panel B (first differences) printed
- [ ] `results/stationarity/integration_orders.csv` created
- [ ] 2 ACF/PACF PNG plots saved
- [ ] You understand KPSS null is opposite to ADF/PP
- [ ] Git commit successful

**If all boxes checked -> DAY 3 COMPLETE!**

---

## What You Built Today

**File created:** 1 (`stationarity_tests.py`)
**Lines of code:** ~430
**Tests implemented:** 3 (ADF, PP, KPSS) on both levels and first differences

**Key result for your thesis:**
- Typical finding: MPR = I(1), ExchangeRate = I(1), M2 = I(0), Inflation = I(0)
- Mixed orders means ARDL is the right model (Day 5)
- I(1) pairs need cointegration testing (Day 4)

---

## Tomorrow (Day 4): Cointegration Testing

**What you'll build:** `src/econometrics/cointegration_tests.py`
- Engle-Granger pairwise tests using `statsmodels.tsa.stattools.coint`
- Johansen multivariate test (trace + max-eigenvalue statistics)
- Residual plots for visual confirmation

---

## Troubleshooting

**"KPSS InterpolationWarning"**
```python
# Normal — p-value is bounded between 0.01 and 0.10
# Suppress: import warnings; warnings.filterwarnings('ignore')
```

**"adfuller returns large positive statistic"**
```python
# Large positive = strong evidence of unit root (non-stationary)
# Need VERY NEGATIVE (< -3.0) to reject unit root
```

**"PP test gives different result than ADF"**
```python
# This can happen — use majority result (2 out of 3)
# Look at ACF plots for visual confirmation
```

---

**Great work! See you tomorrow for Day 4!**
