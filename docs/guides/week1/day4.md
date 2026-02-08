# Week 1 - Day 4: Cointegration Testing

**Time Estimate:** 8 hours (full day)
**What You'll Build:** `src/econometrics/cointegration_tests.py` (~400 lines)
**End Goal:** Determine if I(1) variables share a long-run equilibrium relationship

**Pre-requisite:** Day 3 completed — you know which variables are I(1)

---

## What You're Building Today

By 5 PM today, you will have:

- ✅ Engle-Granger pairwise tests using `statsmodels.tsa.stattools.coint` (proper statistical function)
- ✅ Tests run on ALL ordered pairs — not just I(1) variables
- ✅ Johansen test with BOTH trace AND max-eigenvalue statistics
- ✅ `plot_eg_residuals()` — visual inspection of cointegrating relationships
- ✅ I(1)-only robustness run of Johansen
- ✅ Results saved to `results/cointegration/`

---

## Hour 1 (9 AM - 10 AM): Understand Cointegration & Module Setup

### Step 1.1: What is cointegration?

**Cointegration:** Two or more I(1) series have a long-run equilibrium relationship when their linear combination is I(0).

**Example:**
- MPR and ExchangeRate are both I(1) (random walk, trending)
- But their relationship (via OLS residual) might be I(0) = stationary
- If so, they "move together" in the long run despite short-run deviations

**Why it matters:**
- I(1) variables in VAR → risk of spurious regression
- **But** if they're cointegrated → VAR in levels IS valid
- Or use VECM (Day 6B) to model short-run adjustment + long-run relationship

**Two tests today:**
1. **Engle-Granger (1987):** Pairwise — tests one pair at a time
2. **Johansen (1988):** Multivariate — tests all variables together, finds the RANK

---

### Step 1.2: Create module — imports and setup

Create `src/econometrics/cointegration_tests.py` and **write this first chunk:**

```python
"""
Cointegration Tests — Nigerian Monetary Policy Transmission Analysis

Implements:
    - Engle-Granger two-step test  (statsmodels.tsa.stattools.coint)
    - Johansen trace + max-eigenvalue test  (statsmodels)
    - Residual plots for visual confirmation

Day 4 deliverable.

Academic references
-------------------
Engle & Granger (1987)  - Econometrica 55(2)
Johansen (1988)         - Journal of Economic Dynamics and Control 12
MacKinnon (2010)        - Journal of Business & Economic Statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm


class CointegrationTester:
    """
    Test for cointegration among time series.

    Parameters
    ----------
    df        : pd.DataFrame  - full data (all variables), DatetimeIndex
    int_orders: dict          - integration orders from Day 3, e.g.
                                {'MPR': 'I(1)', 'Inflation': 'I(0)', ...}
    save_dir  : str | Path    - e.g. 'results/cointegration'
    """

    def __init__(self, df: pd.DataFrame,
                 int_orders: Dict[str, str] = None,
                 save_dir: str = "results/cointegration"):
        self.df         = df
        self.int_orders = int_orders or {}
        self.save_dir   = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results    = {}
        print(f"CointegrationTester initialized. Results: {self.save_dir}")


# Test code
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader(data_dir="data")
    df = loader.load_and_prepare()

    tester = CointegrationTester(df, save_dir="results/cointegration")
    print("CointegrationTester initialized successfully!")
```

**Save and test:**

```bash
python src/econometrics/cointegration_tests.py
```

---

## Hour 2 (10 AM - 11 AM): Implement Engle-Granger Test (statsmodels)

### Step 2.1: Add the Engle-Granger method

**Why use `statsmodels.tsa.stattools.coint` instead of manual OLS?**
- Uses MacKinnon (2010) response surface critical values — more accurate for any sample size
- Automatically handles the regression and residual ADF
- No need for hardcoded critical values like `-3.37`

**Add this method to the class** (after `__init__`):

```python
    def _run_eg(self, y: pd.Series, x: pd.Series,
                y_name: str, x_name: str) -> Dict:
        """
        Engle-Granger two-step cointegration test.

        Uses statsmodels.tsa.stattools.coint which:
        1. Regresses y on x using OLS
        2. Tests residuals with ADF using MacKinnon (2010) critical values

        Parameters
        ----------
        y, x      : I(1) series (though coint() tests all pairs)
        y_name, x_name : variable labels

        Returns
        -------
        dict with test results
        """
        # Align series
        data = pd.concat([y, x], axis=1).dropna()
        y_clean = data.iloc[:, 0]
        x_clean = data.iloc[:, 1]

        # coint() returns (t_stat, p_value, crit_values)
        t_stat, p_value, crit_values = coint(y_clean, x_clean)

        return {
            "pair":        f"{y_name} ~ {x_name}",
            "t_statistic": round(t_stat, 4),
            "p_value":     round(p_value, 4),
            "crit_1pct":   round(crit_values[0], 3),
            "crit_5pct":   round(crit_values[1], 3),
            "crit_10pct":  round(crit_values[2], 3),
            "cointegrated": p_value < 0.05,
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

    tester = CointegrationTester(df, save_dir="results/cointegration")

    print("\nEngle-Granger: MPR ~ ExchangeRate")
    result = tester._run_eg(df['MPR'], df['ExchangeRate'], 'MPR', 'ExchangeRate')
    print(f"  t-stat:     {result['t_statistic']}")
    print(f"  p-value:    {result['p_value']}")
    print(f"  Crit (5%):  {result['crit_5pct']}")
    print(f"  Cointegrated: {result['cointegrated']}")
```

**Save and test:**

```bash
python src/econometrics/cointegration_tests.py
```

---

## Hour 3 (11 AM - 12 PM): Run All Pairwise EG Tests

### Step 3.1: Add method to test all pairs

**Add this method after `_run_eg`:**

```python
    def run_all_pairwise_eg(self) -> pd.DataFrame:
        """
        Run Engle-Granger on ALL ordered pairs of variables (both directions).

        Testing all pairs (not just I(1)) is thorough — it lets the data
        speak rather than pre-filtering on integration order.
        """
        print("\n" + "=" * 70)
        print("  ENGLE-GRANGER PAIRWISE COINTEGRATION TESTS")
        print("=" * 70)
        print(f"  {'Pair':<30} {'t-stat':>10} {'p-value':>10} {'Cointegrated':>14}")
        print("  " + "-" * 68)

        results = []
        cols = list(self.df.columns)

        # All ordered pairs (y~x and x~y separately — direction matters)
        for y_name in cols:
            for x_name in cols:
                if y_name == x_name:
                    continue
                r = self._run_eg(self.df[y_name], self.df[x_name],
                                 y_name, x_name)
                results.append(r)
                flag = "YES" if r["cointegrated"] else "no"
                print(f"  {r['pair']:<30} {r['t_statistic']:>10.4f}"
                      f" {r['p_value']:>10.4f}  {flag:>14}")

        print("  " + "=" * 68)

        results_df = pd.DataFrame(results)
        self.results["engle_granger"] = results_df
        return results_df
```

**Update test code:**

```python
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader(data_dir="data")
    df = loader.load_and_prepare()

    tester = CointegrationTester(df, save_dir="results/cointegration")
    eg_df = tester.run_all_pairwise_eg()
    print(f"\nTotal pairs tested: {len(eg_df)}")
    print(f"Cointegrated pairs: {eg_df['cointegrated'].sum()}")
```

**Save and test:**

```bash
python src/econometrics/cointegration_tests.py
```

**You should see 12 pairs tested (4×3=12 ordered pairs).**

---

## Hour 4 (12 PM - 1 PM): LUNCH BREAK

Take a break! Engle-Granger is done.

---

## Hour 5 (1 PM - 2 PM): Implement Johansen Test (Trace + Max-Eigenvalue)

### Step 5.1: Add Johansen test — why two statistics?

**Trace statistic:** Tests H0 = at most r cointegrating vectors
**Max-eigenvalue statistic:** Tests H0 = exactly r cointegrating vectors

They are complementary. Your thesis should report BOTH.

**Add this method after `run_all_pairwise_eg`:**

```python
    def _run_johansen(self, data: pd.DataFrame, label: str = "") -> Dict:
        """
        Johansen cointegration test — returns both trace and max-eigenvalue.

        Parameters
        ----------
        data  : DataFrame with the variables to test
        label : Description string for printing

        Returns
        -------
        dict with rank, trace stats, max-eigen stats, critical values
        """
        print(f"\n  Johansen test: {label}  ({data.shape[1]} variables)")
        print(f"  Variables: {list(data.columns)}")

        # det_order=0: restricted constant (intercept in CE, no trend)
        # k_ar_diff=1: one lag in the VECM representation
        result = coint_johansen(data.dropna(), det_order=0, k_ar_diff=1)

        # --- Trace test ---
        trace_stats = result.lr1        # test statistics
        trace_crit  = result.cvt        # critical values [90%, 95%, 99%]

        # --- Max-eigenvalue test ---
        max_stats = result.lr2          # test statistics
        max_crit  = result.cvm          # critical values [90%, 95%, 99%]

        # Determine rank from trace statistic at 5% level
        rank = 0
        for i in range(len(trace_stats)):
            if trace_stats[i] > trace_crit[i, 1]:  # column 1 = 95% CV
                rank = i + 1

        print(f"\n  Trace Statistics:")
        print(f"  {'H0: rank<=':>12} {'Trace':>10} {'CV 5%':>10} {'Reject?':>10}")
        for i, (stat, cv) in enumerate(zip(trace_stats, trace_crit[:, 1])):
            reject = "Yes" if stat > cv else "No"
            print(f"  {'r <= '+str(i):>12} {stat:>10.4f} {cv:>10.4f} {reject:>10}")

        print(f"\n  Max-Eigenvalue Statistics:")
        print(f"  {'H0: rank=':>12} {'Max-Eig':>10} {'CV 5%':>10} {'Reject?':>10}")
        for i, (stat, cv) in enumerate(zip(max_stats, max_crit[:, 1])):
            reject = "Yes" if stat > cv else "No"
            print(f"  {'r = '+str(i):>12} {stat:>10.4f} {cv:>10.4f} {reject:>10}")

        print(f"\n  => Cointegration rank: {rank}")

        return {
            "label":            label,
            "variables":        list(data.columns),
            "n_variables":      data.shape[1],
            "rank":             rank,
            "trace_statistics": trace_stats.tolist(),
            "trace_cv_5pct":    trace_crit[:, 1].tolist(),
            "max_statistics":   max_stats.tolist(),
            "max_cv_5pct":      max_crit[:, 1].tolist(),
            "interpretation":   self._interpret_rank(rank, data.shape[1]),
        }

    def _interpret_rank(self, rank: int, n_vars: int) -> str:
        if rank == 0:
            return "No cointegration detected"
        elif rank >= n_vars:
            return f"All {n_vars} variables are cointegrated — check specification"
        else:
            return f"{rank} cointegrating relationship(s) detected"
```

**Update test code:**

```python
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader(data_dir="data")
    df = loader.load_and_prepare()

    tester = CointegrationTester(df, save_dir="results/cointegration")

    print("\nRunning Johansen on full system...")
    j_result = tester._run_johansen(df, "Full system")
    print(f"\nInterpretation: {j_result['interpretation']}")
```

**Save and test:**

```bash
python src/econometrics/cointegration_tests.py
```

---

## Hour 6 (2 PM - 3 PM): Add Master Johansen Runner & Residual Plots

### Step 6.1: Add full Johansen analysis (full system + I(1)-only robustness)

**Add this method after `_interpret_rank`:**

```python
    def run_johansen_full(self) -> Dict:
        """
        Run Johansen twice:
        1. Full system (all 4 variables)
        2. I(1)-only subset (robustness check)

        Two runs cross-validate the result.
        """
        print("\n" + "=" * 70)
        print("  JOHANSEN MULTIVARIATE COINTEGRATION TEST")
        print("=" * 70)

        johansen_results = {}

        # Run 1: All variables
        j1 = self._run_johansen(self.df, "Full system (all variables)")
        johansen_results["full_system"] = j1

        # Run 2: I(1)-only subset (if we know orders)
        if self.int_orders:
            i1_vars = [v for v, o in self.int_orders.items() if o == "I(1)"]
            if len(i1_vars) >= 2:
                print(f"\n  --- Robustness: I(1)-only subset ---")
                j2 = self._run_johansen(self.df[i1_vars],
                                         f"I(1)-only: {i1_vars}")
                johansen_results["i1_only"] = j2
            else:
                print(f"\n  Note: fewer than 2 I(1) variables — skipping I(1)-only run.")

        self.results["johansen"] = johansen_results
        return johansen_results
```

### Step 6.2: Add residual plots method

**Add this method after `run_johansen_full`:**

```python
    def plot_eg_residuals(self):
        """
        For each cointegrated pair, plot the OLS residuals (equilibrium error).

        A stationary residual confirms cointegration visually —
        it should revert to zero rather than wander indefinitely.
        """
        if "engle_granger" not in self.results:
            print("  Run run_all_pairwise_eg() first.")
            return

        # Only plot cointegrated pairs
        coint_pairs = self.results["engle_granger"][
            self.results["engle_granger"]["cointegrated"] == True
        ]

        if coint_pairs.empty:
            print("  No cointegrated pairs to plot.")
            return

        n = len(coint_pairs)
        fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
        if n == 1:
            axes = [axes]

        fig.suptitle("Cointegrating Residuals (Equilibrium Errors)",
                     fontsize=14, fontweight="bold")

        for ax, (_, row) in zip(axes, coint_pairs.iterrows()):
            pair = row["pair"]
            y_name, x_name = pair.split(" ~ ")

            # Recompute OLS residual
            data = pd.concat([self.df[y_name], self.df[x_name]], axis=1).dropna()
            X = sm.add_constant(data.iloc[:, 1])
            model = sm.OLS(data.iloc[:, 0], X).fit()
            residuals = model.resid

            ax.plot(residuals.index, residuals, color="#2E86AB", linewidth=1.5)
            ax.axhline(0, color="red", linestyle="--", alpha=0.5)
            ax.set_title(f"Equilibrium Error: {pair}  (p={row['p_value']:.4f})",
                         fontsize=12, fontweight="bold")
            ax.set_ylabel("Residual", fontsize=10)
            ax.grid(True, alpha=0.25)

        fig.tight_layout()
        path = self.save_dir / "eg_residuals.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  saved  {path}")
        plt.close(fig)
```

**Update test code:**

```python
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader(data_dir="data")
    df = loader.load_and_prepare()

    tester = CointegrationTester(df, save_dir="results/cointegration")

    print("\n[1/3] Engle-Granger pairwise...")
    tester.run_all_pairwise_eg()

    print("\n[2/3] Johansen multivariate...")
    tester.run_johansen_full()

    print("\n[3/3] Plotting residuals...")
    tester.plot_eg_residuals()
```

**Save and test:**

```bash
python src/econometrics/cointegration_tests.py
```

---

## Hour 7 (3 PM - 4 PM): Add Printing & Save Methods

### Step 7.1: Add print and save methods

**Add these methods after `plot_eg_residuals`:**

```python
    def _print_eg(self, results_df: pd.DataFrame):
        """Print formatted Engle-Granger summary."""
        print("\n  Summary:")
        n_coint = results_df["cointegrated"].sum()
        print(f"  {n_coint} / {len(results_df)} pairs cointegrated at 5%")
        if n_coint > 0:
            coint_pairs = results_df[results_df["cointegrated"]]["pair"].tolist()
            for p in coint_pairs:
                print(f"    - {p}")

    def _print_johansen(self, johansen_results: Dict):
        """Print formatted Johansen summary."""
        for run_name, r in johansen_results.items():
            print(f"\n  [{run_name}] rank = {r['rank']}: {r['interpretation']}")

    def save_results(self):
        """Save all results to CSV."""
        if "engle_granger" in self.results:
            path = self.save_dir / "engle_granger_all_pairs.csv"
            self.results["engle_granger"].to_csv(path, index=False)
            print(f"  saved  {path}")

        if "johansen" in self.results:
            rows = []
            for run_name, r in self.results["johansen"].items():
                rows.append({
                    "run":              run_name,
                    "variables":        ", ".join(r["variables"]),
                    "rank":             r["rank"],
                    "interpretation":   r["interpretation"],
                })
            path = self.save_dir / "johansen_summary.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"  saved  {path}")
```

---

## Hour 8 (4 PM - 5 PM): Add Full Pipeline & Commit

### Step 8.1: Add run_full_analysis() and main()

**Add this final section:**

```python
    def run_full_analysis(self):
        """Complete cointegration analysis pipeline."""
        print("\n" + "=" * 70)
        print("  COINTEGRATION ANALYSIS - FULL PIPELINE")
        print("=" * 70)

        # 1. Engle-Granger pairwise
        print("\n[1/4] Engle-Granger pairwise tests...")
        eg_df = self.run_all_pairwise_eg()
        self._print_eg(eg_df)

        # 2. Johansen multivariate
        print("\n[2/4] Johansen multivariate test...")
        johansen_results = self.run_johansen_full()
        self._print_johansen(johansen_results)

        # 3. Residual plots
        print("\n[3/4] Plotting cointegrating residuals...")
        self.plot_eg_residuals()

        # 4. Save
        print("\n[4/4] Saving results...")
        self.save_results()

        print("\n" + "=" * 70)
        print("  COINTEGRATION ANALYSIS COMPLETE")
        print("=" * 70)

        return self.results


def main():
    import sys
    sys.path.insert(0, ".")
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader(data_dir="data")
    df = loader.load_and_prepare()

    # Load integration orders if available
    try:
        orders_df = pd.read_csv("results/stationarity/integration_orders.csv")
        int_orders = dict(zip(orders_df["Variable"], orders_df["Order"]))
        print(f"Loaded integration orders: {int_orders}")
    except FileNotFoundError:
        int_orders = {}
        print("Note: integration_orders.csv not found — run Day 3 first")

    tester = CointegrationTester(df, int_orders=int_orders,
                                  save_dir="results/cointegration")
    tester.run_full_analysis()


if __name__ == "__main__":
    main()
```

**Save and run:**

```bash
python src/econometrics/cointegration_tests.py
```

**Verify outputs:**

```bash
wc -l src/econometrics/cointegration_tests.py
# Should show: ~400 lines

ls results/cointegration/
# engle_granger_all_pairs.csv
# johansen_summary.csv
# eg_residuals.png  (only if cointegrated pairs found)
```

---

### Step 8.2: Commit everything

```bash
git add .
git status
```

**Commit:**

```bash
git commit -m "Day 4: Cointegration testing (Engle-Granger + Johansen)

- src/econometrics/cointegration_tests.py (~400 lines)
  - CointegrationTester class
  - Engle-Granger using statsmodels.coint (MacKinnon 2010 critical values)
  - All ordered pairs tested (not just I(1))
  - Johansen trace + max-eigenvalue statistics
  - I(1)-only robustness run
  - plot_eg_residuals() for visual confirmation

Key results (fill in actual findings):
  - EG cointegrated pairs: [list them]
  - Johansen rank: [value]
  - Interpretation: [e.g., '1 long-run relationship detected']

Outputs: results/cointegration/ (2 CSVs + 1 PNG)

https://claude.ai/code/session_019oWdezYCv1NxdFa4QYPPhs"
```

**Push:**

```bash
git push -u origin claude/monetary-policy-analytics-platform-03tRK
```

---

## Final Code Summary

**File**: `src/econometrics/cointegration_tests.py`

```python
class CointegrationTester:
    def __init__(df, int_orders=None, save_dir="results/cointegration")
    def _run_eg(y, x, y_name, x_name)
    def run_all_pairwise_eg()
    def _run_johansen(data, label="")
    def _interpret_rank(rank, n_vars)
    def run_johansen_full()
    def plot_eg_residuals()
    def _print_eg(results_df)
    def _print_johansen(johansen_results)
    def save_results()
    def run_full_analysis()

def main()
```

**Verify:**
```bash
python -c "
from src.econometrics.cointegration_tests import CointegrationTester
methods = [m for m in dir(CointegrationTester) if not m.startswith('__')]
print('Methods:', methods)
"
```

---

## End of Day 4 Checklist

- [ ] `cointegration_tests.py` runs without errors
- [ ] 12 Engle-Granger pairs printed (4 variables × 3 directions)
- [ ] Johansen shows both trace AND max-eigenvalue statistics
- [ ] `results/cointegration/johansen_summary.csv` created
- [ ] You understand cointegration rank and what it means for VAR vs VECM
- [ ] Git commit successful

**If all boxes checked -> DAY 4 COMPLETE!**

---

## What You Built Today

**Key difference from simpler implementations:**
- Used `statsmodels.coint` (proper statistical function) instead of manual OLS + hardcoded `-3.37` critical value
- Reports max-eigenvalue alongside trace — both required for a rigorous thesis
- Tests ALL pairs, not just pre-filtered I(1) combinations
- Visual residual plots confirm cointegration intuitively

**Key result for your thesis:**
- Rank 0: No cointegration → use differenced VAR or ARDL
- Rank 1+: Cointegration found → VAR in levels valid, or use VECM for error-correction

---

## Tomorrow (Day 5): ARDL Bounds Testing

**What you'll build:** `src/econometrics/ardl.py`
- Pesaran et al. (2001) bounds test for mixed I(0)/I(1) systems
- Optimal lag selection via AIC/BIC
- Long-run coefficient extraction

---

## Troubleshooting

**"coint() returns different results than ADF on residuals"**
```python
# statsmodels.coint uses response surface critical values
# which adjust for sample size — they are more accurate
# This is the CORRECT version to use
```

**"Johansen rank = 0 despite EG finding cointegration"**
```python
# Possible — EG is pairwise, Johansen is multivariate
# Different tests can disagree; report both and explain
# Check lag order: try k_ar_diff=2 or k_ar_diff=3
```

**"plot_eg_residuals() says 'No cointegrated pairs to plot'"**
```python
# Run run_all_pairwise_eg() first
# If genuinely no cointegration: ARDL is your primary model
```

---

**Great work finishing Week 1! See you for Day 5!**
